#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""11-02 重训 S1：在 S0 全航司主表上构造候选特征（剔禁止入模42 + seats红线）。
只构造、同义去重记账、覆盖率；不做有用性分析、不训练。
红线：不碰线上推理服务 / [REDACTED_SERVICE] / 写库 / ana_extract / pull8001 / store.json / AF。
口径复用 v2.1 / fb01 / v2_01 / s03_batch2 / 淡旺季 v3，不新造泄露特征。
"""
from __future__ import annotations
import json, time, hashlib, os, sys, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

ROOT = Path("${PROJECT_ROOT}/S1_特征")
S0 = Path("${PROJECT_ROOT}/S0_底表/数据")
MAIN = S0 / "主表_全航司日价_dtd0_30.parquet"
EXP = S0 / "扩样本表_航管2025与2026新_对齐_dtd0_30_含子来源.parquet"
CAL = ROOT / "核对" / "淡旺季日历_2025_2027.csv"
POOL = Path("${PRED_ROOT}/特征候选池_11-02_20261028/淡旺季补充_20261102/特征候选池_去重后_v3_20261102_淡旺季补充.csv")
MID = ROOT / "中间"
DATA = ROOT / "数据"
CHK = ROOT / "核对"
LOG = ROOT / "日志"
for p in (MID, DATA, CHK, LOG, ROOT / "脚本"):
    p.mkdir(parents=True, exist_ok=True)

MEM = os.environ.get("S1_DUCK_MEM", "8GB")
THREADS = int(os.environ.get("S1_THREADS", "4"))
廉航 = ("9C", "AQ")
切换日 = "2026-10-21"
邻时段分钟 = 180
B最少历史天数 = 2
上一观察最大间隔天 = 3
FIT_END_TRAIN = "2025-12-30"  # 训练集分位参考：只用航管2025（全部早于主表观察日）

def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG / "构建.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def md5file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def con_new():
    c = duckdb.connect()
    tmp = MID / "duck_tmp"
    tmp.mkdir(exist_ok=True)
    c.execute(f"SET memory_limit='{MEM}'; SET threads={THREADS}; SET temp_directory='{tmp}'; SET preserve_insertion_order=false")
    return c

def phase_done(name: str) -> bool:
    return (MID / f"_DONE_{name}").exists()

def mark_done(name: str, info: dict | None = None):
    (MID / f"_DONE_{name}").write_text(json.dumps(info or {}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# ---------------- 0 清单 ----------------
def phase0():
    if phase_done("00_清单"):
        log("skip 00_清单"); return
    raw = POOL.read_bytes()
    assert hashlib.md5(raw).hexdigest().startswith("b4072fc8")
    df = pd.read_csv(POOL)
    banned = df[df["泄露风险分类"] != "未涉及（非v2.1禁止入模列）"]
    cands = df[df["泄露风险分类"] == "未涉及（非v2.1禁止入模列）"]
    seats = cands[cands["特征名"].astype(str).str.contains("seats|余座|余票", case=False, regex=True)]
    keep = cands[~cands["特征名"].isin(seats["特征名"])]
    assert len(cands) == 216 and len(banned) == 42
    keep.to_csv(CHK / "待构造特征清单_剔禁止与seats.csv", index=False)
    mark_done("00_清单", {"待构造": len(keep), "seats跳过": seats["特征名"].tolist()})
    log(f"00_清单 待构造={len(keep)}")

# ---------------- 1 基础+日历+淡旺季日历 ----------------
def phase1():
    if phase_done("01_基础日历"):
        log("skip 01_基础日历"); return
    t0 = time.time()
    c = con_new()
    # 主表键+基础
    c.execute(f"""
    CREATE TABLE base AS
    SELECT
      CAST(观察日 AS DATE) 观察日, 航班号, 航线, 出发机场, 到达机场,
      CAST(起飞日 AS DATE) 起飞日, 起飞时刻, 航司, 距起飞天数, CAST(价格 AS DOUBLE) 价格,
      来源批次, 数据源, 是否回补键, 混批标记, 批次事故标记, 批1117空日标记, 是否满7天窗口, 采集窗,
      CASE WHEN regexp_matches(起飞时刻, '^[0-9]{{1,2}}:[0-9]{{2}}$')
           THEN CAST(split_part(起飞时刻,':',1) AS INT)*60 + CAST(split_part(起飞时刻,':',2) AS INT) END mm,
      CAST(起飞日 AS DATE) - DATE '2026-01-01' AS dd
    FROM read_parquet('{MAIN}')
    WHERE 价格 > 0 AND length(航班号) <> 7
    """)
    c.execute("ALTER TABLE base ADD COLUMN abs_min BIGINT")
    c.execute("UPDATE base SET abs_min = CASE WHEN mm IS NOT NULL THEN dd::BIGINT*1440 + mm END")
    n = c.execute("SELECT count(*) FROM base").fetchone()[0]
    log(f"base rows={n}")

    # 日历（chinese_calendar + 淡旺季表）
    import chinese_calendar as cc
    cal_season = pd.read_csv(CAL)
    cal_season["d"] = pd.to_datetime(cal_season["d"]).dt.date
    # 扩展观察/起飞覆盖到 2025-2027
    days = pd.date_range("2025-01-01", "2027-03-15").date
    hol_map = {}
    tiaoxiu = []
    W = {"Spring Festival": 3.0, "National Day": 3.0, "Labour Day": 2.0, "Mid-autumn Festival": 1.5,
         "Dragon Boat Festival": 1.5, "Tomb-sweeping Day": 1.5, "New Year's Day": 1.0}
    TYPE = {"New Year's Day": 1, "Spring Festival": 2, "Tomb-sweeping Day": 3, "Labour Day": 4,
            "Dragon Boat Festival": 5, "Mid-autumn Festival": 6, "National Day": 7}
    for d in days:
        if d.year <= 2026:
            try:
                on, name = cc.get_holiday_detail(d)
                if on and name is not None:
                    hol_map[d] = name
                if (not on) and name is not None:
                    tiaoxiu.append(d)
            except Exception:
                pass
        else:
            # 2027 元旦补录；春运/其余用淡旺季日历 is_holiday_dep_ref
            if d == dt.date(2027, 1, 1):
                hol_map[d] = "New Year's Day"
    # 从淡旺季日历补 2027 假日名
    for _, r in cal_season.iterrows():
        d = r["d"]
        if d.year >= 2027 and int(r.get("is_holiday_dep_ref") or 0) == 1 and d not in hol_map:
            hol_map[d] = str(r.get("hol_name") or "Holiday")
    SF = [dt.date(2025, 1, 29), dt.date(2026, 2, 17), dt.date(2027, 2, 6)]
    hd = np.array(sorted(hol_map), dtype="datetime64[D]")
    td = np.array(sorted(tiaoxiu), dtype="datetime64[D]") if tiaoxiu else np.array([], dtype="datetime64[D]")
    sf = np.array(SF, dtype="datetime64[D]")
    seg_len = {}
    hs = sorted(hol_map)
    i = 0
    while i < len(hs):
        j = i
        while j + 1 < len(hs) and (hs[j + 1] - hs[j]).days == 1:
            j += 1
        for k in range(i, j + 1):
            seg_len[hs[k]] = j - i + 1
        i = j + 1

    def cal_feats(dd):
        d = np.array(dd, dtype="datetime64[D]")
        if len(hd) == 0:
            z = np.full(len(d), np.nan)
            return z, z, z, z, z, z
        nx = np.searchsorted(hd, d, "left")
        pv = np.searchsorted(hd, d, "right") - 1
        to_next = (hd[np.minimum(nx, len(hd) - 1)] - d).astype(int).astype(float)
        to_next[nx >= len(hd)] = np.nan
        since = (d - hd[np.maximum(pv, 0)]).astype(int).astype(float)
        since[pv < 0] = np.nan
        if len(td):
            ti = np.searchsorted(td, d)
            a1 = np.abs((td[np.minimum(ti, len(td) - 1)] - d).astype(int))
            a0 = np.abs((td[np.maximum(ti - 1, 0)] - d).astype(int))
            tia = np.minimum(a0, a1).astype(float)
        else:
            tia = np.full(len(d), np.nan)
        si = np.searchsorted(sf, d)
        s1 = (sf[np.minimum(si, len(sf) - 1)] - d).astype(int)
        s0 = (sf[np.maximum(si - 1, 0)] - d).astype(int)
        sfd = np.where(np.abs(s0) < np.abs(s1), s0, s1).astype(float)
        near = np.where(np.nan_to_num(to_next, nan=1e9) <= np.nan_to_num(since, nan=1e9),
                        np.minimum(nx, len(hd) - 1), np.maximum(pv, 0))
        names = [hol_map[pd.Timestamp(x).date()] for x in hd[near]]
        ht = np.array([TYPE.get(n, 0) for n in names], float)
        hwt = np.array([W.get(n, 1.0) * seg_len.get(pd.Timestamp(x).date(), 1) / 3.0 for n, x in zip(names, hd[near])])
        dist = np.minimum(np.nan_to_num(to_next, nan=1e9), np.nan_to_num(since, nan=1e9))
        ht[dist > 30] = 0
        hwt = np.where(dist > 30, 0.0, hwt / (1.0 + dist))
        return to_next, since, tia, sfd, ht, hwt

    # days_to_hol / days_from_hol / is_gw 按 chinese_calendar is_holiday（含周末）
    def days_hol_cc(dd):
        # cc.is_holiday includes weekends
        hol_days = []
        for d in pd.date_range("2025-01-01", "2027-03-15").date:
            try:
                if cc.is_holiday(d):
                    hol_days.append(d)
            except Exception:
                if d.weekday() >= 5:
                    hol_days.append(d)
        hd2 = np.array(hol_days, dtype="datetime64[D]")
        d = np.array(dd, dtype="datetime64[D]")
        nx = np.searchsorted(hd2, d, "left")
        pv = np.searchsorted(hd2, d, "right") - 1
        to_next = (hd2[np.minimum(nx, len(hd2) - 1)] - d).astype(int).astype(float)
        to_next[nx >= len(hd2)] = 999
        since = (d - hd2[np.maximum(pv, 0)]).astype(int).astype(float)
        since[pv < 0] = 999
        return to_next, since

    all_days = pd.date_range("2025-01-01", "2027-03-15")
    g1 = cal_feats(all_days.date)
    d2h, dfh = days_hol_cc(all_days.date)
    cal_df = pd.DataFrame({
        "d": all_days.strftime("%Y-%m-%d"),
        "g1_days_to_next_hol": g1[0],
        "g1_days_since_last_hol": g1[1],
        "g1_days_to_tiaoxiu": g1[2],
        "g1_days_to_spring_festival": g1[3],
        "g1_holiday_type": g1[4],
        "g1_holiday_weight": g1[5],
        "days_to_hol": d2h,
        "days_from_hol": dfh,
        "is_gw": [1 if (x.month == 10 and 1 <= x.day <= 7) else 0 for x in all_days.date],
    })
    # merge season calendar
    season = cal_season.copy()
    season["d"] = pd.to_datetime(season["d"]).dt.strftime("%Y-%m-%d")
    cal_df = cal_df.merge(season[["d", "dep_month", "dep_is_shuyun", "dep_is_chunyun", "dep_golden_week_phase",
                                   "dep_short_hol_peak", "dep_is_off_season", "dep_season_level"]],
                          on="d", how="left")
    # weekend / holiday flags via cc
    flags = []
    for x in all_days.date:
        try:
            is_hol = int(cc.is_holiday(x))
            is_work = int(cc.is_workday(x))
        except Exception:
            is_hol = int(x.weekday() >= 5)
            is_work = int(x.weekday() < 5)
        flags.append((is_hol, is_work, int(x.weekday() >= 5)))
    cal_df["is_holiday"] = [f[0] for f in flags]
    cal_df["is_workday"] = [f[1] for f in flags]
    cal_df["is_weekend"] = [f[2] for f in flags]
    cal_path = MID / "日历表.parquet"
    cal_df.to_parquet(cal_path, index=False)
    c.execute(f"CREATE TABLE cal AS SELECT * FROM read_parquet('{cal_path}')")

    # 基础特征块
    out1 = MID / "块01_基础日历.parquet"
    c.execute(f"""
    COPY (
      SELECT
        b.观察日, b.航班号, b.航线, b.起飞日, b.航司, b.价格, b.距起飞天数, b.出发机场, b.到达机场,
        CASE WHEN b.mm IS NOT NULL THEN b.mm // 60 END AS 起飞小时,
        dayofweek(b.起飞日) AS 起飞日星期,
        dayofweek(b.观察日) AS 观察日星期,
        CASE WHEN b.观察日 >= DATE '{切换日}' THEN 1 ELSE 0 END AS 采集期,
        -- 现网日历
        cd.is_weekend AS is_weekend_dep,
        co.is_weekend AS is_weekend_obs,
        cd.is_holiday AS is_holiday_dep,
        co.is_holiday AS is_holiday_obs,
        co.is_workday AS is_workday_obs,
        CASE WHEN co.is_weekend=1 OR co.is_holiday=1 THEN 1 ELSE 0 END AS is_weekend_or_hol_obs,
        month(b.观察日) AS month,
        month(b.起飞日) AS dep_month,
        cd.dep_is_shuyun, cd.dep_is_chunyun, cd.dep_golden_week_phase, cd.dep_short_hol_peak,
        cd.dep_is_off_season, cd.dep_season_level,
        -- G1
        cd.g1_days_to_next_hol, cd.g1_days_since_last_hol, cd.g1_days_to_tiaoxiu,
        cd.g1_days_to_spring_festival, cd.g1_holiday_type, cd.g1_holiday_weight,
        co.g1_days_to_next_hol AS g1_obs_days_to_next_hol,
        -- batch2 日历
        co.days_to_hol, co.days_from_hol, co.is_gw,
        CASE WHEN co.is_gw=1 AND b.距起飞天数 <= 10 THEN 1 ELSE 0 END AS gw_x_near,
        CASE WHEN cd.is_weekend=1 AND b.距起飞天数 <= 7 THEN 1 ELSE 0 END AS weekend_dep_x_near,
        -- 四族
        cd.dep_golden_week_phase AS hd_dep_hol_phase,
        cd.g1_days_to_next_hol AS hd_dep_days_to_next_hol,
        -- gap（观察间隔占位，后面自身历史再填真正上一观察间隔；这里先放距起飞相关）
        b.距起飞天数 AS gap,
        CASE WHEN b.距起飞天数 >= 3 THEN 1 ELSE 0 END AS gap_ge3,
        -- 通程腿类型：主表已剔 len=7；同航班多 OD 另算
        CASE WHEN length(b.航班号)=6 THEN 'direct' ELSE 'other' END AS leg_type,
        CASE WHEN length(b.航班号)=6 THEN 1 ELSE 0 END AS is_direct,
        0 AS is_thru_leg,
        0 AS is_thru_total
      FROM base b
      LEFT JOIN cal cd ON cd.d = strftime(b.起飞日, '%Y-%m-%d')
      LEFT JOIN cal co ON co.d = strftime(b.观察日, '%Y-%m-%d')
    ) TO '{out1}' (FORMAT PARQUET)
    """)
    log(f"01_基础日历 done {time.time()-t0:.0f}s -> {out1}")
    mark_done("01_基础日历", {"rows": n, "secs": round(time.time()-t0)})
    c.close()

# ---------------- 2 自身历史 / 价格窗口 / N / G4 / roll ----------------
def phase2():
    if phase_done("02_自身历史"):
        log("skip 02_自身历史"); return
    t0 = time.time()
    c = con_new()
    c.execute(f"""
    CREATE TABLE base AS
    SELECT CAST(观察日 AS DATE) 观察日, 航班号, 航线, CAST(起飞日 AS DATE) 起飞日, 航司,
           距起飞天数, CAST(价格 AS DOUBLE) 价格, 出发机场, 到达机场
    FROM read_parquet('{MAIN}') WHERE 价格>0 AND length(航班号)<>7
    """)
    # 序列事件
    c.execute("""
    CREATE TABLE seq AS
    WITH t AS (
      SELECT *, lag(价格) OVER w pp, lag(价格,2) OVER w pp2,
             lag(观察日) OVER w po, lead(价格) OVER w np,
             date_diff('day', lag(观察日) OVER w, 观察日) gap_obs,
             row_number() OVER w AS obs_idx
      FROM base
      WINDOW w AS (PARTITION BY 航班号, 航线, 起飞日 ORDER BY 观察日)
    )
    SELECT *,
      (pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3) AS hc,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp <= -0.15 OR 价格-pp <= -100) THEN 1 ELSE 0 END AS ev_dn,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp >= 0.10 OR 价格-pp >= 80) THEN 1 ELSE 0 END AS ev_up,
      CASE WHEN pp IS NOT NULL AND 价格>pp THEN 1 ELSE 0 END AS up1,
      CASE WHEN pp IS NOT NULL AND 价格<pp THEN 1 ELSE 0 END AS dn1,
      CASE WHEN pp IS NOT NULL THEN 价格/pp - 1 END AS pct
    FROM t
    """)
    # 滚动窗口（按观察次数，不含未来；含当天）
    c.execute("""
    CREATE TABLE roll AS
    SELECT *,
      -- 历史窗（不含当天）用于特征B
      count(价格) OVER wh AS 特征B_历史天数,
      avg(价格) OVER wh AS b_mean_raw,
      -- 含当天的滚动分位/极值（近7/14次观测）
      quantile_cont(价格, 0.2) OVER w7 AS roll7_p20,
      quantile_cont(价格, 0.8) OVER w7 AS roll7_p80,
      max(价格) OVER w7 AS roll7_max,
      quantile_cont(价格, 0.2) OVER w14 AS roll14_p20,
      quantile_cont(价格, 0.8) OVER w14 AS roll14_p80,
      max(价格) OVER w14 AS roll14_max,
      min(价格) OVER w_all AS own_min_h,
      max(价格) OVER w_all AS own_max_h,
      avg(价格) OVER w_all AS own_mean_h,
      stddev_samp(价格) OVER w_all AS p_std,
      sum(ev_up) OVER w_all AS n_bigup_hist,
      sum(CASE WHEN hc THEN 1 ELSE 0 END) OVER w_all AS n_hc_h,
      sum(ev_dn) OVER w_all AS n_bigdrop_hist,
      sum(ev_dn) OVER w7d AS g4_n_bigdrop_7d,
      sum(up1) OVER w3 AS up_cnt_3obs,
      sum(dn1) OVER w3 AS dn_cnt_3obs,
      last_value(CASE WHEN ev_up=1 THEN pct END IGNORE NULLS) OVER w_all AS own_last_bigup_pct,
      max(CASE WHEN ev_dn=1 THEN 观察日 END) OVER w_all AS day_bd,
      max(CASE WHEN ev_up=1 THEN 观察日 END) OVER w_all AS day_rise,
      -- 过去1/3/7/14 天日历窗均价（不含当天）
      avg(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS mean_1d,
      avg(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS mean_3d,
      avg(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS mean_7d,
      avg(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 14 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS mean_14d,
      min(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 14 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS min_14d,
      -- 斜率：近3/7天 (末-首)/天数
      first(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS p_first_3,
      last(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS p_last_3,
      first(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS p_first_7,
      last(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS p_last_7,
      count(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS n_3d,
      count(价格) OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING) AS n_7d
    FROM seq
    WINDOW
      wh AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
      w7 AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
      w14 AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
      w_all AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      w7d AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW),
      w3 AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    """)
    # 跳变次数（阶梯：相邻价差绝对值≥50）
    c.execute("""
    CREATE TABLE jump AS
    SELECT 观察日,航班号,航线,起飞日,
      sum(CASE WHEN pp IS NOT NULL AND abs(价格-pp)>=50 THEN 1 ELSE 0 END) OVER w AS 价格跳变_阶梯次数,
      max(CASE WHEN pp IS NOT NULL AND abs(价格-pp)>=50 THEN 观察日 END) OVER w AS last_jump_day,
      sum(CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND 价格>pp THEN 1 ELSE 0 END) OVER w7 AS 近期涨价次数
    FROM seq
    WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
           w7 AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)
    """)
    # G4 大降细节：关联大降日价格
    c.execute("""
    CREATE TABLE g4 AS
    SELECT a.观察日, a.航班号, a.航线, a.起飞日, a.价格, a.距起飞天数, a.day_bd,
      date_diff('day', a.day_bd, a.观察日) AS g4_days_since_bigdrop,
      CASE WHEN b.pp IS NOT NULL THEN (b.价格-b.pp)/b.pp END AS g4_bigdrop_pct,
      CASE WHEN b.pp IS NOT NULL THEN b.价格-b.pp END AS g4_bigdrop_yuan,
      CASE WHEN b.价格 IS NOT NULL AND b.价格>0 THEN a.价格/b.价格 - 1 END AS g4_rebound_since,
      CASE WHEN b.pp IS NOT NULL AND b.pp>0 THEN a.价格/b.pp - 1 END AS g4_prev_vs_prebd,
      CASE WHEN a.day_bd = a.观察日 THEN 1 ELSE 0 END AS g4_last_is_bigdrop,
      a.g4_n_bigdrop_7d, a.n_bigdrop_hist AS g4_n_bigdrop_hist
    FROM roll a
    LEFT JOIN seq b ON a.航班号=b.航班号 AND a.航线=b.航线 AND a.起飞日=b.起飞日 AND b.观察日=a.day_bd
    """)
    # 去尖峰版（简化：大降后下一次观测回到≥0.97×降前价则剔除该事件）
    c.execute("""
    CREATE TABLE g4b AS
    WITH s2 AS (
      SELECT *, (pp IS NOT NULL AND ((价格-pp)/pp <= -0.15 OR 价格-pp <= -100))::INT AS bd,
             coalesce(np >= 0.97*pp, false) AS rev
      FROM seq
    ),
    s3 AS (
      SELECT *, sum(bd) OVER w0 AS n_hist,
        max(CASE WHEN bd=1 AND NOT rev THEN 观察日 END)
          OVER (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS lc
      FROM s2 WINDOW w0 AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ),
    s4 AS (SELECT *, CASE WHEN bd=1 THEN 观察日 ELSE lc END AS day_dn FROM s3)
    SELECT a.观察日, a.航班号, a.航线, a.起飞日,
      date_diff('day', a.day_dn, a.观察日) AS g4_dn_days_since_bigdrop,
      CASE WHEN b.pp IS NOT NULL THEN (b.价格-b.pp)/b.pp END AS g4_dn_bigdrop_pct
    FROM s4 a
    LEFT JOIN s4 b ON a.航班号=b.航班号 AND a.航线=b.航线 AND a.起飞日=b.起飞日 AND b.观察日=a.day_dn
    """)

    out2 = MID / "块02_自身历史.parquet"
    c.execute(f"""
    COPY (
      SELECT
        r.观察日, r.航班号, r.航线, r.起飞日,
        -- 上一观察
        CASE WHEN r.po IS NOT NULL AND r.gap_obs <= {上一观察最大间隔天} THEN r.价格 - r.pp END AS 上一观察价差_元,
        CASE WHEN r.po IS NOT NULL AND r.gap_obs <= {上一观察最大间隔天} AND r.pp>0 THEN r.价格 / r.pp END AS 上一观察价比值,
        CASE WHEN r.po IS NOT NULL AND r.gap_obs <= {上一观察最大间隔天} THEN r.gap_obs END AS 上一观察间隔天数,
        CASE WHEN r.pp IS NOT NULL THEN ln(1 + r.pp) END AS log_price_prev,
        CASE WHEN r.p_std IS NOT NULL AND r.p_std>0 AND r.pp IS NOT NULL THEN (r.pp - r.own_mean_h)/r.p_std END AS price_prev_z,
        r.gap_obs AS gap_obs_real,
        r.obs_idx,
        -- B
        coalesce(r.特征B_历史天数, 0) AS 特征B_历史天数,
        CASE WHEN r.特征B_历史天数 >= {B最少历史天数} THEN r.b_mean_raw END AS 特征B_历史均价,
        CASE WHEN r.特征B_历史天数 >= {B最少历史天数} THEN r.价格 - r.b_mean_raw END AS 特征B_差值_元,
        CASE WHEN r.特征B_历史天数 >= {B最少历史天数} AND r.b_mean_raw>0 THEN r.价格 / r.b_mean_raw END AS 特征B_比值,
        CASE WHEN r.特征B_历史天数 >= 1 AND r.b_mean_raw>0 THEN r.价格 / r.b_mean_raw END AS prev_vs_own_mean,
        -- 价格窗口口述
        r.mean_1d AS 过去1天均价,
        r.mean_3d AS 过去3天均价,
        r.mean_7d AS 过去7天均价,
        r.mean_14d AS 过去14天均价,
        r.min_14d AS 近N天最低价,
        j.近期涨价次数,
        j.价格跳变_阶梯次数,
        date_diff('day', j.last_jump_day, r.观察日) AS 上次跳变距今天数,
        CASE WHEN r.n_3d>=2 THEN (r.p_last_3 - r.p_first_3)/3.0 END AS 近3天价斜率,
        CASE WHEN r.n_7d>=2 THEN (r.p_last_7 - r.p_first_7)/7.0 END AS 近7天价斜率,
        -- roll
        r.roll7_p20, r.roll7_p80, r.roll7_max,
        CASE WHEN r.roll7_max>0 THEN 1 - r.价格/r.roll7_max END AS roll7_drawdown,
        r.roll14_p20, r.roll14_p80, r.roll14_max,
        CASE WHEN r.roll14_max>0 THEN 1 - r.价格/r.roll14_max END AS roll14_drawdown,
        CASE WHEN r.roll14_max>r.roll14_p20 THEN
          (r.价格 - r.roll14_p20) / nullif(r.roll14_max - r.roll14_p20, 0) END AS roll14_rank,
        CASE WHEN r.roll7_p80>0 THEN r.价格 / r.roll7_p80 END AS prev_vs_roll7_p80,
        CASE WHEN r.roll14_p20>0 THEN r.价格 / r.roll14_p20 END AS prev_vs_roll14_p20,
        -- N 自身涨势
        r.n_bigup_hist,
        CASE WHEN r.n_hc_h>0 THEN r.n_bigup_hist*1.0/r.n_hc_h END AS own_bigup_rate_hist,
        r.own_last_bigup_pct,
        CASE WHEN r.own_min_h>0 THEN r.价格/r.own_min_h END AS prev_vs_own_min,
        CASE WHEN r.pp2 IS NOT NULL THEN r.价格/r.pp2 - 1 END AS chg_2obs,
        r.up_cnt_3obs AS up_cnt_3obs,
        r.dn_cnt_3obs AS dn_cnt_3obs,
        r.p_std,
        r.own_max_h AS peak,
        CASE WHEN r.own_max_h>0 THEN r.价格/r.own_max_h END AS peak_ratio,
        -- days_since / recent
        date_diff('day', r.day_bd, r.观察日) AS days_since,
        date_diff('day', r.day_rise, r.观察日) AS days_since_rise,
        CASE WHEN r.day_bd IS NOT NULL AND date_diff('day', r.day_bd, r.观察日) BETWEEN 0 AND 3 THEN 1 ELSE 0 END AS recent_drop,
        CASE WHEN r.day_rise IS NOT NULL AND date_diff('day', r.day_rise, r.观察日) BETWEEN 0 AND 3 THEN 1 ELSE 0 END AS recent_rise,
        CASE WHEN r.day_rise IS NOT NULL AND r.价格 >= coalesce(r.pp, r.价格)*0.97 THEN 1 ELSE 0 END AS after_rise_elevated,
        CASE WHEN r.day_bd IS NOT NULL AND r.价格 > coalesce(
          (SELECT s2.价格 FROM seq s2 WHERE s2.航班号=r.航班号 AND s2.航线=r.航线 AND s2.起飞日=r.起飞日 AND s2.观察日=r.day_bd), r.价格)
          THEN 1 ELSE 0 END AS drop_then_rebound,
        CASE WHEN r.day_rise IS NOT NULL AND r.day_bd IS NOT NULL AND r.day_bd > r.day_rise
          THEN date_diff('day', r.day_rise, r.day_bd) END AS rise_then_drop_gap,
        -- trend: 简单线性（近窗价相对均值）
        CASE WHEN r.own_mean_h>0 THEN r.价格/r.own_mean_h - 1 END AS trend,
        CASE WHEN r.obs_idx=1 THEN 1 ELSE 0 END AS obs_is_first,
        CASE WHEN r.n_hc_h>0 THEN r.n_bigup_hist*1.0/r.n_hc_h END AS own_rise_rate,
        CASE WHEN r.pct IS NOT NULL AND r.pct>0 THEN r.pct ELSE 0 END AS prev_chg_pos,
        -- G4
        g.g4_days_since_bigdrop, g.g4_bigdrop_pct, g.g4_bigdrop_yuan, g.g4_rebound_since, g.g4_prev_vs_prebd,
        g.g4_last_is_bigdrop, g.g4_n_bigdrop_7d, g.g4_n_bigdrop_hist,
        CASE WHEN g.g4_bigdrop_pct IS NOT NULL THEN g.g4_bigdrop_pct * r.距起飞天数 END AS g4_bd_x_dtd,
        gb.g4_dn_days_since_bigdrop, gb.g4_dn_bigdrop_pct,
        CASE WHEN gb.g4_dn_bigdrop_pct IS NOT NULL THEN gb.g4_dn_bigdrop_pct * r.距起飞天数 END AS g4_dn_bd_x_dtd
      FROM roll r
      LEFT JOIN jump j USING (观察日, 航班号, 航线, 起飞日)
      LEFT JOIN g4 g USING (观察日, 航班号, 航线, 起飞日)
      LEFT JOIN g4b gb USING (观察日, 航班号, 航线, 起飞日)
    ) TO '{out2}' (FORMAT PARQUET)
    """)
    log(f"02_自身历史 done {time.time()-t0:.0f}s")
    mark_done("02_自身历史", {"secs": round(time.time()-t0)})
    c.close()

# ---------------- 3 竞争 / 横截面 / G2 G3 / A ----------------
def phase3():
    if phase_done("03_竞争横截面"):
        log("skip 03_竞争横截面"); return
    t0 = time.time()
    c = con_new()
    c.execute(f"""
    CREATE TABLE base AS
    SELECT CAST(观察日 AS DATE) 观察日, 航班号, 航线, CAST(起飞日 AS DATE) 起飞日, 航司,
           距起飞天数, CAST(价格 AS DOUBLE) 价格, 起飞时刻,
           CASE WHEN regexp_matches(起飞时刻,'^[0-9]{{1,2}}:[0-9]{{2}}$')
                THEN CAST(split_part(起飞时刻,':',1) AS INT)*60+CAST(split_part(起飞时刻,':',2) AS INT) END mm,
           CAST(起飞日 AS DATE) - DATE '2026-01-01' AS dd,
           出发机场, 到达机场
    FROM read_parquet('{MAIN}') WHERE 价格>0 AND length(航班号)<>7
    """)
    c.execute("ALTER TABLE base ADD COLUMN abs_min BIGINT")
    c.execute("UPDATE base SET abs_min = CASE WHEN mm IS NOT NULL THEN dd::BIGINT*1440+mm END")

    # 共享航班 R3 候选（同观察日同航线同起飞分钟）
    days = [r[0] for r in c.execute("SELECT DISTINCT 观察日 FROM base ORDER BY 1").fetchall()]
    log(f"obs days={len(days)}")
    c.execute("""
    CREATE TABLE smday AS
    SELECT c.观察日, c.航线, c.航班号 AS fc, p.航班号 AS fp, c.起飞日, (c.价格=p.价格)::INT AS eq
    FROM base c JOIN base p
      ON c.观察日=p.观察日 AND c.航线=p.航线 AND c.起飞日=p.起飞日
     AND c.abs_min=p.abs_min AND c.航班号<>p.航班号
    WHERE c.abs_min IS NOT NULL
    """)
    c.execute("""
    CREATE TABLE sm AS
    SELECT *, min(eq) OVER w AS 全同价, count(*) OVER w AS 共同天数
    FROM smday
    WINDOW w AS (PARTITION BY 航线, fc, fp, 起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    # 特征A 逐日（内存友好）
    c.execute("""CREATE TABLE fa(
      观察日 DATE, 航班号 VARCHAR, 航线 VARCHAR, 起飞日 DATE,
      n_ex BIGINT, med_ex DOUBLE, min_ex DOUBLE, lower_ex BIGINT
    )""")
    for d in days:
        c.execute(f"""
        INSERT INTO fa
        SELECT c.观察日, c.航班号, c.航线, c.起飞日,
          count(*) FILTER (WHERE coalesce(s.全同价,0)<>1) n_ex,
          median(p.价格) FILTER (WHERE coalesce(s.全同价,0)<>1) med_ex,
          min(p.价格) FILTER (WHERE coalesce(s.全同价,0)<>1) min_ex,
          count(*) FILTER (WHERE coalesce(s.全同价,0)<>1 AND p.价格<c.价格) lower_ex
        FROM (SELECT * FROM base WHERE 观察日=DATE '{d}' AND abs_min IS NOT NULL) c
        JOIN (SELECT * FROM base WHERE 观察日=DATE '{d}' AND abs_min IS NOT NULL AND 航司 NOT IN {廉航} AND 价格>0) p
          ON p.航线=c.航线 AND p.dd BETWEEN c.dd-1 AND c.dd+1
         AND abs(p.abs_min-c.abs_min)<={邻时段分钟}
         AND NOT (p.航班号=c.航班号 AND p.起飞日=c.起飞日)
        LEFT JOIN (SELECT * FROM sm WHERE 观察日=DATE '{d}') s
          ON s.航线=c.航线 AND s.fc=c.航班号 AND s.起飞日=c.起飞日 AND s.fp=p.航班号 AND p.起飞日=s.起飞日
        GROUP BY c.观察日, c.航班号, c.航线, c.起飞日
        """)
    log(f"fa done {time.time()-t0:.0f}s")

    # 同日同航线竞争（含当天快照；严格过去版另算 gap）
    c.execute("""
    CREATE TABLE same_day AS
    SELECT a.观察日, a.航班号, a.航线, a.起飞日, a.价格, a.航司, a.abs_min, a.mm,
      count(b.航班号) AS n_comp,
      count(DISTINCT b.航司) AS n_airlines,
      min(b.价格) AS comp_min,
      median(b.价格) AS comp_med,
      max(b.价格) AS comp_max,
      avg(CASE WHEN b.价格 < a.价格 THEN 1.0 WHEN b.价格=a.价格 THEN 0.5 ELSE 0.0 END) AS x_rank_pct
    FROM base a
    LEFT JOIN base b
      ON a.观察日=b.观察日 AND a.航线=b.航线 AND a.起飞日=b.起飞日 AND a.航班号<>b.航班号
    GROUP BY 1,2,3,4,5,6,7,8
    """)
    # ±2h 竞争
    c.execute("""
    CREATE TABLE pm2h AS
    SELECT a.观察日, a.航班号, a.航线, a.起飞日,
      count(b.航班号) AS n_flights_pm2h,
      count(DISTINCT b.航司) AS n_airlines_pm2h,
      min(b.价格) AS comp_min_pm2h,
      count(b.航班号) FILTER (WHERE b.价格 IS NOT NULL) AS n_priced_pm2h
    FROM base a
    LEFT JOIN base b
      ON a.观察日=b.观察日 AND a.航线=b.航线 AND a.起飞日=b.起飞日 AND a.航班号<>b.航班号
     AND a.abs_min IS NOT NULL AND b.abs_min IS NOT NULL AND abs(a.abs_min-b.abs_min)<=120
    GROUP BY 1,2,3,4
    """)
    # 同日航线全日
    c.execute("""
    CREATE TABLE day_route AS
    SELECT 观察日, 航线, 起飞日,
      count(*) AS n_flights_day,
      count(DISTINCT 航司) AS n_airlines_day,
      min(价格) AS min_day,
      count(*) AS n_priced_day,
      count(DISTINCT 航班号) AS g3_n_sale
    FROM base GROUP BY 1,2,3
    """)
    # 市场动价 G2：相对上一观察日
    c.execute("""
    CREATE TABLE seq_m AS
    SELECT *, lag(价格) OVER w pp, lag(观察日) OVER w po,
      date_diff('day', lag(观察日) OVER w, 观察日) gap_obs
    FROM base WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日)
    """)
    c.execute("""
    CREATE TABLE fl AS
    SELECT *,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 THEN 1 ELSE 0 END AS has_cmp,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp<=-0.15 OR 价格-pp<=-100) THEN 1 ELSE 0 END AS dr,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp>=0.10 OR 价格-pp>=80) THEN 1 ELSE 0 END AS ri
    FROM seq_m
    """)
    c.execute("""
    CREATE TABLE mk AS
    SELECT 航线, 起飞日, 观察日,
      count(*) AS n_sale, sum(has_cmp) AS n_cmp, sum(dr) AS n_dr, sum(ri) AS n_ri, min(价格) AS mn
    FROM fl GROUP BY 1,2,3
    """)
    c.execute("""
    CREATE TABLE mk2 AS
    SELECT *, lag(mn) OVER w mn_prev, lag(n_sale) OVER w ns_prev, lag(观察日) OVER w od_prev
    FROM mk WINDOW w AS (PARTITION BY 航线,起飞日 ORDER BY 观察日)
    """)
    c.execute("""
    CREATE TABLE mk3 AS
    SELECT a.*, b.n_sale AS ns_7
    FROM mk2 a LEFT JOIN mk b
      ON a.航线=b.航线 AND a.起飞日=b.起飞日 AND b.观察日 = a.观察日 - INTERVAL 7 DAY
    """)
    c.execute("""
    CREATE TABLE pm AS
    SELECT a.航线, a.起飞日, a.观察日,
      sum(b.n_cmp) AS n_cmp_pm1, sum(b.n_dr) AS n_dr_pm1, sum(b.n_ri) AS n_ri_pm1
    FROM mk a JOIN mk b
      ON a.航线=b.航线 AND b.观察日=a.观察日
     AND b.起飞日 BETWEEN a.起飞日 - INTERVAL 1 DAY AND a.起飞日 + INTERVAL 1 DAY
    GROUP BY 1,2,3
    """)
    c.execute("""CREATE TABLE first_seen AS SELECT 航班号,航线,起飞日,min(观察日) f0 FROM base GROUP BY 1,2,3""")
    c.execute("""
    CREATE TABLE ent AS
    SELECT m.航线, m.起飞日, m.观察日, count(f.航班号) AS n_enter
    FROM mk m LEFT JOIN first_seen f
      ON f.航线=m.航线 AND f.起飞日=m.起飞日 AND f.f0 > m.观察日 - INTERVAL 7 DAY AND f.f0 <= m.观察日
    GROUP BY 1,2,3
    """)
    # 兄弟航班（同航班号其他起飞日，同观察日）
    c.execute("""
    CREATE TABLE sib AS
    SELECT a.观察日, a.航班号, a.航线, a.起飞日,
      count(b.航班号) AS n_sib,
      median(b.价格) AS sib_median,
      avg(b.价格) AS sib_mean,
      avg(CASE WHEN b.价格 < a.价格 THEN 1.0 ELSE 0.0 END) AS sib_low_ratio,
      min(b.价格) AS sib_min
    FROM base a
    LEFT JOIN base b
      ON a.观察日=b.观察日 AND a.航班号=b.航班号 AND a.航线=b.航线 AND a.起飞日<>b.起飞日
    GROUP BY 1,2,3,4
    """)
    # 同航班号兄弟涨跌（用 fl 事件）
    c.execute("""
    CREATE TABLE sib_ev AS
    SELECT a.观察日, a.航班号, a.航线, a.起飞日,
      CASE WHEN sum(CASE WHEN b.航班号 IS NOT NULL AND b.has_cmp=1 THEN 1 ELSE 0 END)>0
        THEN sum(CASE WHEN b.航班号 IS NOT NULL THEN b.ri ELSE 0 END)*1.0
           / sum(CASE WHEN b.航班号 IS NOT NULL AND b.has_cmp=1 THEN 1 ELSE 0 END) END AS sib_bigup_share,
      avg(CASE WHEN b.航班号 IS NOT NULL THEN b.价格/nullif(b.pp,0)-1 END) AS sib_chg_mean
    FROM fl a
    LEFT JOIN fl b
      ON a.观察日=b.观察日 AND a.航班号=b.航班号 AND a.航线=b.航线 AND a.起飞日<>b.起飞日
    GROUP BY 1,2,3,4
    """)
    # 航线市场涨势
    c.execute("""
    CREATE TABLE route_ev AS
    SELECT 航线, 观察日,
      CASE WHEN sum(has_cmp)>0 THEN sum(ri)*1.0/sum(has_cmp) END AS route_bigup_share
    FROM fl GROUP BY 1,2
    """)
    c.execute("""
    CREATE TABLE route_ev7 AS
    SELECT 航线, 观察日,
      CASE WHEN sum(sum(has_cmp)) OVER w > 0
        THEN sum(sum(ri)) OVER w * 1.0 / sum(sum(has_cmp)) OVER w END AS route_bigup_share_7d
    FROM fl
    GROUP BY 1,2
    WINDOW w AS (PARTITION BY 航线 ORDER BY 观察日 RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)
    """)
    # 同航班多 OD
    c.execute("""
    CREATE TABLE odn AS
    SELECT 观察日, 航班号, 起飞日, count(DISTINCT 航线) AS n_od_same_flight
    FROM base GROUP BY 1,2,3
    """)
    # is_codeshare 近似：R3 全同价邻居数>0
    c.execute("""
    CREATE TABLE cs AS
    SELECT 观察日, fc AS 航班号, 航线, 起飞日, max(全同价) AS is_codeshare_raw
    FROM sm GROUP BY 1,2,3,4
    """)

    out3 = MID / "块03_竞争横截面.parquet"
    c.execute(f"""
    COPY (
      SELECT
        a.观察日, a.航班号, a.航线, a.起飞日,
        -- A
        CASE WHEN fa.n_ex>=1 THEN fa.n_ex END AS 特征A_邻时段对比航班数,
        CASE WHEN fa.n_ex>=1 THEN fa.med_ex END AS 特征A_邻时段中位价,
        CASE WHEN fa.n_ex>=1 THEN a.价格 - fa.med_ex END AS 特征A_差值_元,
        CASE WHEN fa.n_ex>=1 AND fa.med_ex>0 THEN a.价格/fa.med_ex END AS 特征A_比值,
        CASE WHEN fa.n_ex>=1 THEN fa.lower_ex::DOUBLE/fa.n_ex END AS 特征A_比自己便宜的邻居占比,
        CASE WHEN fa.n_ex>=1 THEN CASE WHEN a.价格<=fa.min_ex THEN 1 ELSE 0 END END AS 特征A_是否邻时段最低价,
        CASE WHEN fa.n_ex>=1 AND fa.med_ex>0 THEN CASE WHEN a.价格/fa.med_ex >= 1.15 THEN 1 ELSE 0 END END AS "特征A_贵于邻近中位≥15%",
        -- 竞争
        sd.n_comp, sd.n_airlines, sd.comp_min, sd.comp_med, sd.comp_max,
        CASE WHEN sd.comp_min IS NOT NULL THEN a.价格 - sd.comp_min END AS gap_to_min,
        CASE WHEN sd.comp_min>0 THEN a.价格 / sd.comp_min END AS prev_over_min,
        CASE WHEN sd.comp_max IS NOT NULL THEN a.价格 - sd.comp_max END AS gap_to_max,
        CASE WHEN sd.comp_max>0 THEN a.价格 / sd.comp_max END AS prev_over_max,
        sd.x_rank_pct,
        CASE WHEN sd.comp_med>0 THEN a.价格/sd.comp_med END AS x_over_median,
        CASE WHEN sd.comp_max>0 THEN a.价格/sd.comp_max END AS x_over_max,
        -- allair 同口径（本表已是全航司）
        sd.n_comp AS n_comp_all, sd.n_airlines AS n_air_all, sd.comp_min AS comp_min_all,
        CASE WHEN sd.comp_min IS NOT NULL THEN a.价格 - sd.comp_min END AS gap_to_min_all,
        CASE WHEN sd.comp_min>0 THEN a.价格 / sd.comp_min END AS prev_over_min_all,
        CASE WHEN sd.n_comp>0 THEN 1 ELSE 0 END AS has_af_comp,
        -- sameday 别名
        sd.comp_min AS comp_min_same,
        CASE WHEN sd.comp_min IS NOT NULL THEN a.价格 - sd.comp_min END AS gap_to_min_same,
        CASE WHEN sd.comp_min>0 THEN a.价格 / sd.comp_min END AS prev_over_min_same,
        -- pm2h / day
        dr.n_flights_day, dr.n_airlines_day, dr.n_priced_day, dr.g3_n_sale,
        dr.g3_n_sale AS sp_n_onsale,
        dr.n_airlines_day AS sp_n_airlines_onsale,
        pm.n_flights_pm2h, pm.n_airlines_pm2h, pm.n_priced_pm2h,
        CASE WHEN pm.n_flights_pm2h>0 THEN 1 ELSE 0 END AS has_comp_schedule,
        CASE WHEN dr.min_day IS NOT NULL THEN a.价格 - dr.min_day END AS gap_to_min_day,
        CASE WHEN dr.min_day>0 THEN a.价格 / dr.min_day END AS prev_over_min_day,
        CASE WHEN pm.comp_min_pm2h IS NOT NULL THEN a.价格 - pm.comp_min_pm2h END AS gap_to_min_pm2h,
        CASE WHEN pm.comp_min_pm2h>0 THEN a.价格 / pm.comp_min_pm2h END AS prev_over_min_pm2h,
        -- G2/G3
        mk.n_cmp AS g2_n_cmp,
        CASE WHEN mk.n_cmp>0 THEN mk.n_ri*1.0/mk.n_cmp END AS g2_mkt_rise_share,
        CASE WHEN mk.n_cmp>0 THEN mk.n_dr*1.0/mk.n_cmp END AS g2_mkt_drop_share,
        CASE WHEN pmm.n_cmp_pm1>0 THEN pmm.n_ri_pm1*1.0/pmm.n_cmp_pm1 END AS g2_mkt_rise_share_pm1,
        CASE WHEN pmm.n_cmp_pm1>0 THEN pmm.n_dr_pm1*1.0/pmm.n_cmp_pm1 END AS g2_mkt_drop_share_pm1,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN mk.mn/nullif(mk.mn_prev,0) - 1 END AS g2_mkt_min_chg,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN
          CASE WHEN mk.mn>mk.mn_prev THEN 1 WHEN mk.mn<mk.mn_prev THEN -1 ELSE 0 END END AS g2_mktmin_dir,
        -- oth = 市场减自身（近似：用市场占比）
        CASE WHEN mk.n_cmp>0 THEN mk.n_ri*1.0/mk.n_cmp END AS g2_oth_bigup_share,
        CASE WHEN mk.n_cmp>0 THEN mk.n_dr*1.0/mk.n_cmp END AS g2_oth_bigdn_share,
        CASE WHEN mk.n_cmp>0 THEN (mk.n_ri - mk.n_dr)*1.0/mk.n_cmp END AS g2_oth_net_move_share,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN mk.mn/nullif(mk.mn_prev,0) - 1 END AS g2_othmin_chg_pct,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN mk.n_sale - mk.ns_prev END AS g3_n_sale_chg1,
        mk.n_sale - mk.ns_7 AS g3_n_sale_chg7,
        coalesce(ent.n_enter,0) AS g3_enter_7d,
        0 AS g3_exit_7d,
        -- sib
        sib.n_sib, sib.sib_median,
        CASE WHEN sib.sib_median>0 THEN a.价格/sib.sib_median END AS sib_price_level,
        sib.sib_low_ratio,
        CASE WHEN sib.sib_mean>0 THEN a.价格/sib.sib_mean END AS sib_cur_vs_min_avg,
        sib.sib_mean AS sib_mean_same,
        sib.n_sib AS n_sib_same,
        se.sib_bigup_share, se.sib_chg_mean,
        se.sib_bigup_share AS sib_bigup_rate_hist,
        CASE WHEN sib.sib_min>0 THEN sib.sib_min/a.价格 END AS sib_drop_ratio,
        CASE WHEN se.sib_bigup_share IS NOT NULL THEN 1-se.sib_bigup_share END AS sib_recent_drop_rate,
        -- route market
        re.route_bigup_share, r7.route_bigup_share_7d,
        -- od / codeshare
        od.n_od_same_flight,
        coalesce(cs.is_codeshare_raw, 0) AS is_codeshare,
        -- argmin 简化占位
        CAST(NULL AS VARCHAR) AS argmin_cls,
        CAST(NULL AS DOUBLE) AS argmin_lag_days,
        CAST(NULL AS BIGINT) AS n_post,
        CAST(NULL AS BIGINT) AS n_same_in_ex,
        CAST(NULL AS BIGINT) AS n_stale
      FROM base a
      LEFT JOIN fa USING (观察日,航班号,航线,起飞日)
      LEFT JOIN same_day sd USING (观察日,航班号,航线,起飞日)
      LEFT JOIN pm2h pm USING (观察日,航班号,航线,起飞日)
      LEFT JOIN day_route dr USING (观察日,航线,起飞日)
      LEFT JOIN mk3 mk USING (观察日,航线,起飞日)
      LEFT JOIN pm pmm USING (观察日,航线,起飞日)
      LEFT JOIN ent USING (观察日,航线,起飞日)
      LEFT JOIN sib USING (观察日,航班号,航线,起飞日)
      LEFT JOIN sib_ev se USING (观察日,航班号,航线,起飞日)
      LEFT JOIN route_ev re ON re.航线=a.航线 AND re.观察日=a.观察日
      LEFT JOIN route_ev7 r7 ON r7.航线=a.航线 AND r7.观察日=a.观察日
      LEFT JOIN odn od ON od.观察日=a.观察日 AND od.航班号=a.航班号 AND od.起飞日=a.起飞日
      LEFT JOIN cs ON cs.观察日=a.观察日 AND cs.航班号=a.航班号 AND cs.航线=a.航线 AND cs.起飞日=a.起飞日
    ) TO '{out3}' (FORMAT PARQUET)
    """)
    log(f"03_竞争横截面 done {time.time()-t0:.0f}s")
    mark_done("03_竞争横截面", {"secs": round(time.time()-t0)})
    c.close()

# ---------------- 4 去年同期 / 淡旺季 R / train 分位 ----------------
def phase4():
    if phase_done("04_去年与训练分位"):
        log("skip 04_去年与训练分位"); return
    t0 = time.time()
    c = con_new()
    # 主表键
    c.execute(f"""
    CREATE TABLE S AS
    SELECT CAST(观察日 AS DATE) 观察日, 航班号, 航线, CAST(起飞日 AS DATE) 起飞日, 航司,
           距起飞天数 AS dtd, CAST(价格 AS DOUBLE) 价格, 出发机场, 到达机场
    FROM read_parquet('{MAIN}') WHERE 价格>0 AND length(航班号)<>7
    """)
    # 2025 扩样本（方圆+原库）做 ly / train
    c.execute(f"""
    CREATE TABLE L2025 AS
    SELECT CAST(观察日 AS DATE) 观察日, 航班号, 航线, CAST(起飞日 AS DATE) 起飞日, 航司,
           距起飞天数 AS dtd, CAST(价格 AS DOUBLE) 价格, 子来源
    FROM read_parquet('{EXP}')
    WHERE 价格>0 AND length(航班号)<>7
      AND CAST(观察日 AS DATE) >= DATE '2025-06-01' AND CAST(观察日 AS DATE) <= DATE '2025-12-30'
      AND CAST(起飞日 AS DATE) >= DATE '2025-06-01' AND CAST(起飞日 AS DATE) <= DATE '2025-12-31'
      AND 距起飞天数 BETWEEN 0 AND 30
    """)
    log(f"L2025 rows={c.execute('select count(*) from L2025').fetchone()[0]}")

    # ly_mean: 去年同航线同 dtd 均价（现网口径偏 9–12；这里用 6–12 全段，说明里披露）
    c.execute("""
    CREATE TABLE lym AS
    SELECT 航线, dtd, avg(价格) AS ly_mean, count(*) AS ly_n
    FROM L2025 GROUP BY 1,2
    """)
    # ly 涨跌率：去年同航线同 dtd 的相邻观察涨跌占比（简化：用同键序列）
    c.execute("""
    CREATE TABLE ly_seq AS
    SELECT 航线, 航班号, 起飞日, 观察日, dtd, 价格,
      lag(价格) OVER w pp,
      date_diff('day', lag(观察日) OVER w, 观察日) gap
    FROM L2025
    WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日)
    """)
    c.execute("""
    CREATE TABLE ly_rates AS
    SELECT 航线, dtd,
      avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 AND ((价格-pp)/pp>=0.10 OR 价格-pp>=80) THEN 1.0 ELSE 0.0 END) AS ly_rise_rate,
      avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 AND ((价格-pp)/pp<=-0.15 OR 价格-pp<=-100) THEN 1.0 ELSE 0.0 END) AS ly_drop_rate,
      avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 THEN 价格/pp - 1 END) AS ly_trend
    FROM ly_seq GROUP BY 1,2
    """)
    # R1/R4 去年同航线同起飞月
    c.execute("""
    CREATE TABLE rday AS
    SELECT 航线, 起飞日, sum(价格) s, count(*) n, count(DISTINCT 航班号) nf
    FROM L2025 GROUP BY 1,2
    """)
    c.execute("""
    CREATE TABLE rmon AS
    SELECT 航线, month(起飞日) m,
      sum(s)/sum(n) AS ly_route_depmonth_price,
      avg(nf) AS ly_route_depmonth_nflights
    FROM rday GROUP BY 1,2
    """)
    # 分母 as-of：对主表每个观察日，用 obs_day < 样本obs 的累计（2025 数据最晚 < 主表最早，整体断言）
    chk = c.execute("SELECT (SELECT max(观察日) FROM L2025), (SELECT min(观察日) FROM S)").fetchone()
    assert chk[0] < chk[1], chk
    log(f"leak assert ok: L2025 max obs {chk[0]} < S min obs {chk[1]}")
    c.execute("""
    CREATE TABLE rmean AS SELECT 航线, avg(价格) AS ly_route_mean FROM L2025 GROUP BY 1
    """)
    c.execute("""
    CREATE TABLE rcap AS SELECT 航线, avg(nf) AS ly_route_nflights_all FROM rday GROUP BY 1
    """)
    # R3 week
    c.execute("""
    CREATE TABLE rweek AS
    SELECT s.观察日, s.航班号, s.航线, s.起飞日,
      (SELECT sum(s2)/sum(n2) FROM (
         SELECT sum(价格) s2, count(*) n2 FROM L2025 w
         WHERE w.航线=s.航线 AND w.起飞日 BETWEEN s.起飞日 - INTERVAL 367 DAY AND s.起飞日 - INTERVAL 361 DAY
       )) AS week_price
    FROM S s
    """)
    # train 分位：按 航司×航线 在 2025 上
    c.execute("""
    CREATE TABLE train_route AS
    SELECT 航司, 航线,
      quantile_cont(价格,0.2) tr_p20, quantile_cont(价格,0.5) tr_p50,
      quantile_cont(价格,0.8) tr_p80, quantile_cont(价格,0.9) tr_p90,
      count(*) n
    FROM L2025 GROUP BY 1,2
    """)
    # city heat
    c.execute("""
    CREATE TABLE city AS
    WITH t AS (
      SELECT split_part(航线,'-',1) orig, split_part(航线,'-',2) dest, 观察日,
        count(*) vol,
        avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 AND ((价格-pp)/pp<=-0.15 OR 价格-pp<=-100) THEN 1.0 ELSE 0.0 END) drop_rate
      FROM ly_seq GROUP BY 1,2,3
    )
    SELECT orig, avg(vol) orig_avg_vol, avg(drop_rate) orig_drop_rate FROM t GROUP BY 1
    """)
    c.execute("""
    CREATE TABLE city_d AS
    WITH t AS (
      SELECT split_part(航线,'-',1) orig, split_part(航线,'-',2) dest, 观察日,
        count(*) vol,
        avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 AND ((价格-pp)/pp<=-0.15 OR 价格-pp<=-100) THEN 1.0 ELSE 0.0 END) drop_rate
      FROM ly_seq GROUP BY 1,2,3
    )
    SELECT dest, avg(vol) dest_avg_vol, avg(drop_rate) dest_drop_rate FROM t GROUP BY 1
    """)
    # route_freq：主表内 as-of 累计（观察日之前该航线出现次数的 log1p）——用整天频次
    c.execute("""
    CREATE TABLE route_day_cnt AS
    SELECT 航线, 观察日, count(*) AS cnt FROM S GROUP BY 1,2
    """)
    c.execute("""
    CREATE TABLE route_freq AS
    SELECT 航线, 观察日,
      ln(1 + sum(cnt) OVER (PARTITION BY 航线 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)) AS route_freq
    FROM route_day_cnt
    """)
    # train_pct_pos / train_bin_rank / train_dtd_rise_rate：用 2025
    c.execute("""
    CREATE TABLE train_dtd AS
    SELECT dtd,
      avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 AND ((价格-pp)/pp>=0.10 OR 价格-pp>=80) THEN 1.0 ELSE 0.0 END) AS train_dtd_rise_rate
    FROM ly_seq GROUP BY 1
    """)
    c.execute("""
    CREATE TABLE train_route_dtd AS
    SELECT 航线, dtd,
      avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 AND ((价格-pp)/pp>=0.10 OR 价格-pp>=80) THEN 1.0 ELSE 0.0 END) AS train_route_dtd_rise_rate
    FROM ly_seq GROUP BY 1,2
    """)
    # R6：已起飞航班价格滚动（用主表+扩样本2025的起飞日）
    c.execute("""
    CREATE TABLE dep_prices AS
    SELECT 航线, 起飞日, sum(价格) s, count(*) n FROM (
      SELECT 航线, 起飞日, 价格 FROM S
      UNION ALL
      SELECT 航线, 起飞日, 价格 FROM L2025
    ) GROUP BY 1,2
    """)

    # 航线近7日涨占比（预聚合，避免逐行相关子查询）
    c.execute("""
    CREATE TABLE fl_s AS
    WITH t AS (
      SELECT 航线,航班号,起飞日,观察日,价格,
        lag(价格) OVER w pp, date_diff('day', lag(观察日) OVER w, 观察日) gap
      FROM S WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日)
    )
    SELECT 航线,观察日,
      CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 THEN 1 ELSE 0 END has_cmp,
      CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3 AND ((价格-pp)/pp>=0.10 OR 价格-pp>=80) THEN 1 ELSE 0 END ri
    FROM t
    """)
    c.execute("""
    CREATE TABLE route_day_rise AS
    SELECT 航线,观察日, sum(ri) su, sum(has_cmp) nh FROM fl_s GROUP BY 1,2
    """)
    c.execute("""
    CREATE TABLE route_recent_rise AS
    SELECT 航线,观察日,
      CASE WHEN sum(nh) OVER w>0 THEN sum(su) OVER w *1.0/sum(nh) OVER w END AS route_recent_rise_rate
    FROM route_day_rise
    WINDOW w AS (PARTITION BY 航线 ORDER BY 观察日 RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)
    """)
    log("route_recent_rise preagg done")

    out4 = MID / "块04_去年与训练分位.parquet"
    c.execute(f"""
    COPY (
      SELECT
        s.观察日, s.航班号, s.航线, s.起飞日,
        l.ly_mean, l.ly_n,
        CASE WHEN l.ly_mean IS NOT NULL THEN 1 ELSE 0 END AS has_ly,
        CASE WHEN l.ly_mean>0 THEN s.价格/l.ly_mean END AS prev_vs_ly,
        lr.ly_rise_rate, lr.ly_drop_rate, lr.ly_trend,
        CASE WHEN l.ly_mean>0 THEN greatest(0, l.ly_mean/s.价格 - 1) END AS ly_room_to_rise,
        m.ly_route_depmonth_price,
        CASE WHEN rm.ly_route_mean>0 THEN m.ly_route_depmonth_price/rm.ly_route_mean END AS ly_route_depmonth_ratio,
        CASE WHEN rm.ly_route_mean>0 THEN rw.week_price/rm.ly_route_mean END AS ly_route_depweek_ratio,
        m.ly_route_depmonth_nflights,
        CASE WHEN rc.ly_route_nflights_all>0 THEN m.ly_route_depmonth_nflights/rc.ly_route_nflights_all END AS ly_route_depmonth_cap_ratio,
        CASE WHEN (
          SELECT count(*) FROM dep_prices d WHERE d.航线=s.航线 AND d.起飞日 BETWEEN s.观察日 - INTERVAL 28 DAY AND s.观察日 - INTERVAL 1 DAY
        ) >= 20 THEN
          (SELECT sum(s)/sum(n) FROM dep_prices d WHERE d.航线=s.航线 AND d.起飞日 BETWEEN s.观察日 - INTERVAL 7 DAY AND s.观察日 - INTERVAL 1 DAY)
          /
          nullif((SELECT sum(s)/sum(n) FROM dep_prices d WHERE d.航线=s.航线 AND d.起飞日 BETWEEN s.观察日 - INTERVAL 28 DAY AND s.观察日 - INTERVAL 1 DAY),0)
        END AS route_recent_dep_level,
        tr.tr_p20, tr.tr_p50, tr.tr_p80, tr.tr_p90,
        CASE WHEN tr.tr_p50>0 THEN s.价格/tr.tr_p50 END AS prev_vs_train_p50,
        CASE WHEN tr.tr_p50 IS NOT NULL THEN
          CASE WHEN s.价格 <= tr.tr_p20 THEN 0.1
               WHEN s.价格 <= tr.tr_p50 THEN 0.35
               WHEN s.价格 <= tr.tr_p80 THEN 0.65
               WHEN s.价格 <= tr.tr_p90 THEN 0.85
               ELSE 0.95 END END AS train_bin_rank,
        -- train_pct_pos: 价格在训练分位中的经验分位近似
        CASE WHEN tr.tr_p90>tr.tr_p20 THEN least(1, greatest(0, (s.价格 - tr.tr_p20)/nullif(tr.tr_p90-tr.tr_p20,0))) END AS train_pct_pos,
        rf.route_freq,
        co.orig_avg_vol, co.orig_drop_rate,
        cd.dest_avg_vol, cd.dest_drop_rate,
        td.train_dtd_rise_rate,
        trd.train_route_dtd_rise_rate,
        rrr.route_recent_rise_rate
      FROM S s
      LEFT JOIN lym l ON l.航线=s.航线 AND l.dtd=s.dtd
      LEFT JOIN ly_rates lr ON lr.航线=s.航线 AND lr.dtd=s.dtd
      LEFT JOIN rmon m ON m.航线=s.航线 AND m.m=month(s.起飞日)
      LEFT JOIN rmean rm ON rm.航线=s.航线
      LEFT JOIN rcap rc ON rc.航线=s.航线
      LEFT JOIN rweek rw ON rw.观察日=s.观察日 AND rw.航班号=s.航班号 AND rw.航线=s.航线 AND rw.起飞日=s.起飞日
      LEFT JOIN train_route tr ON tr.航司=s.航司 AND tr.航线=s.航线
      LEFT JOIN route_freq rf ON rf.航线=s.航线 AND rf.观察日=s.观察日
      LEFT JOIN city co ON co.orig=s.出发机场
      LEFT JOIN city_d cd ON cd.dest=s.到达机场
      LEFT JOIN train_dtd td ON td.dtd=s.dtd
      LEFT JOIN train_route_dtd trd ON trd.航线=s.航线 AND trd.dtd=s.dtd
      LEFT JOIN route_recent_rise rrr ON rrr.航线=s.航线 AND rrr.观察日=s.观察日
    ) TO '{out4}' (FORMAT PARQUET)
    """)
    log(f"04_去年与训练分位 done {time.time()-t0:.0f}s")
    mark_done("04_去年与训练分位", {"secs": round(time.time()-t0), "leak_assert": [str(chk[0]), str(chk[1])]})
    c.close()

# ---------------- 5 组装 + 组合线索 + 覆盖 + 去重 ----------------
def phase5():
    if phase_done("05_组装"):
        log("skip 05_组装"); return
    t0 = time.time()
    c = con_new()
    keep = pd.read_csv(CHK / "待构造特征清单_剔禁止与seats.csv")
    want = set(keep["特征名"].tolist())

    b1 = MID / "块01_基础日历.parquet"
    b2 = MID / "块02_自身历史.parquet"
    b3 = MID / "块03_竞争横截面.parquet"
    b4 = MID / "块04_去年与训练分位.parquet"
    assert b1.exists() and b2.exists() and b3.exists() and b4.exists()

    # 组合线索：≥15天+偏高价；距起飞×当前价分位交互；同航司同航线当日价分位
    out = DATA / "特征矩阵_全航司_dtd0_30.parquet"
    # 先拼大表到中间，再选列
    c.execute(f"""
    CREATE TABLE M AS
    SELECT
      a.*,
      b.* EXCLUDE (观察日, 航班号, 航线, 起飞日),
      c.* EXCLUDE (观察日, 航班号, 航线, 起飞日),
      d.* EXCLUDE (观察日, 航班号, 航线, 起飞日)
    FROM read_parquet('{b1}') a
    LEFT JOIN read_parquet('{b2}') b USING (观察日, 航班号, 航线, 起飞日)
    LEFT JOIN read_parquet('{b3}') c USING (观察日, 航班号, 航线, 起飞日)
    LEFT JOIN read_parquet('{b4}') d USING (观察日, 航班号, 航线, 起飞日)
    """)
    # 组合
    c.execute("""
    CREATE TABLE M2 AS
    SELECT *,
      CASE WHEN 距起飞天数>=15 AND "特征A_贵于邻近中位≥15%"=1 THEN 1 ELSE 0 END AS "≥15天+偏高价",
      距起飞天数 * coalesce(roll14_rank, train_pct_pos) AS "距起飞×当前价分位交互",
      percent_rank() OVER (PARTITION BY 观察日, 航司, 航线 ORDER BY 价格) AS 同航司同航线当日价分位
    FROM M
    """)
    # 列清单：键 + 目标特征（存在的）
    cols = [r[0] for r in c.execute("DESCRIBE M2").fetchall()]
    keys = ["观察日", "航班号", "航线", "起飞日", "航司", "距起飞天数", "出发机场", "到达机场"]
    # 别名映射：候选名 → 实际列
    alias = {
        # already matching mostly
    }
    present = []
    missing = []
    for f in sorted(want):
        if f in cols:
            present.append(f)
        elif f in alias and alias[f] in cols:
            present.append(f)
        else:
            missing.append(f)

    # 写出：键 + present（若有别名用原名）
    select_cols = keys + [f'"{c_}"' if not c_.isidentifier() and not all('\u4e00' <= ch <= '\u9fff' or ch.isalnum() or ch=='_' for ch in c_) else f'"{c_}"' for c_ in present]
    # simpler quoting
    def q(name):
        return '"' + name.replace('"','""') + '"'
    sel = ", ".join(q(k) for k in keys + present)
    c.execute(f"COPY (SELECT {sel} FROM M2) TO '{out}' (FORMAT PARQUET)")
    n = c.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    log(f"matrix rows={n} feats={len(present)} missing={len(missing)}")

    # 覆盖率
    cov_rows = []
    for f in present:
        r = c.execute(f"""
          SELECT count(*) n,
                 count({q(f)}) nn,
                 count({q(f)})*1.0/count(*) cov,
                 min({q(f)}) mn, max({q(f)}) mx
          FROM read_parquet('{out}')
        """).fetchone()
        cov_rows.append({"特征名": f, "非空数": r[1], "总行数": r[0], "覆盖率": round(r[2], 6),
                         "最小": r[3], "最大": r[4],
                         "是否需航管历史": "是" if f.startswith("ly_") or f.startswith("tr_") or f in (
                             "ly_mean","ly_n","has_ly","prev_vs_ly","ly_rise_rate","ly_drop_rate","ly_trend",
                             "ly_room_to_rise","orig_avg_vol","orig_drop_rate","dest_avg_vol","dest_drop_rate",
                             "train_pct_pos","train_bin_rank","prev_vs_train_p50","train_dtd_rise_rate",
                             "train_route_dtd_rise_rate","route_recent_dep_level") else "否"})
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(CHK / "特征覆盖率.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"特征名": missing, "原因": "本S1未实现或需额外源"}).to_csv(CHK / "未实现特征.csv", index=False, encoding="utf-8-sig")

    # 同义去重对照（基于候选池查重，不按测试集挑）
    pool = pd.read_csv(POOL)
    cands = pool[pool["泄露风险分类"] == "未涉及（非v2.1禁止入模列）"].copy()
    cands = cands[~cands["特征名"].astype(str).str.contains("seats|余座|余票", case=False, regex=True)]
    # 分组：查重结论含疑似冗余/高相关 的按说明归组；合并进来的同义特征
    groups = []
    gid = 0
    used = set()
    # 显式同义组
    syn_groups = [
        ["特征A_比值", "特征A_差值_元", "特征A_贵于邻近中位≥15%"],
        ["特征A_是否邻时段最低价", "特征A_比自己便宜的邻居占比"],
        ["特征B_比值", "特征B_差值_元", "prev_vs_own_mean"],
        ["上一观察价差_元", "上一观察价比值", "prev_chg_pos"],
        ["is_weekend_obs", "观察日星期"],
        ["is_weekend_dep", "起飞日星期"],
        ["log_price_prev", "价格"],
        ["has_ly", "ly_mean"],
        ["gap_to_min", "prev_over_min"],
        ["gap_to_max", "prev_over_max"],
        ["gap_to_min_all", "prev_over_min_all"],
        ["gap_to_min_same", "prev_over_min_same", "gap_to_min"],
        ["g2_oth_net_move_share", "g2_oth_bigup_share", "g2_oth_bigdn_share"],
        ["g2_mktmin_dir", "g2_mkt_min_chg"],
        ["g4_bd_x_dtd", "g4_bigdrop_pct", "距起飞天数"],
        ["g4_dn_bd_x_dtd", "g4_dn_bigdrop_pct", "距起飞天数"],
        ["g4_last_is_bigdrop", "days_since", "g4_days_since_bigdrop"],
        ["g4_prev_vs_prebd", "g4_rebound_since", "g4_bigdrop_pct"],
        ["peak", "peak_ratio", "roll14_max"],
        ["roll7_max", "roll7_drawdown", "prev_vs_roll7_p80", "roll7_p80"],
        ["roll14_max", "roll14_drawdown", "prev_vs_roll14_p20", "roll14_p20"],
        ["tr_p50", "prev_vs_train_p50"],
        ["is_weekend_or_hol_obs", "is_weekend_obs", "is_holiday_obs"],
        ["weekend_dep_x_near", "is_weekend_dep", "距起飞天数"],
        ["gw_x_near", "is_gw", "距起飞天数"],
        ["has_af_comp", "n_comp_all"],
        ["is_direct", "is_thru_leg", "is_thru_total", "leg_type"],
        ["gap_ge3", "上一观察间隔天数"],
        ["dep_is_chunyun", "g1_days_to_spring_festival"],
        ["dep_is_off_season", "dep_season_level"],
        ["g3_n_sale", "sp_n_onsale"],
        ["hd_dep_days_to_next_hol", "g1_days_to_next_hol"],
        ["距起飞天数", "dtd", "gap"],
        ["≥15天+偏高价", "距起飞天数", "特征A_贵于邻近中位≥15%"],
        ["距起飞×当前价分位交互", "距起飞天数", "roll14_rank"],
    ]
    rows = []
    for g in syn_groups:
        gid += 1
        members = [x for x in g if x in want or x in ("dtd",)]
        # 代表：优先已实现且查重结论为「保留」的
        rep = None
        for x in members:
            if x in present:
                rep = x; break
        if rep is None and members:
            rep = members[0]
        for x in members:
            if x == "dtd":
                continue
            rows.append({
                "同义组号": gid,
                "代表特征": rep,
                "成员特征": x,
                "是否已构造": "是" if x in present else ("键同义" if x in keys else "否"),
                "来源": "候选池查重/公式可推出",
                "计数规则": "同义组只计1次（有用性分析时）",
            })
            used.add(x)
    # 其余各自成组
    for f in keep["特征名"]:
        if f in used:
            continue
        gid += 1
        rows.append({
            "同义组号": gid,
            "代表特征": f,
            "成员特征": f,
            "是否已构造": "是" if f in present else "否",
            "来源": "独立候选",
            "计数规则": "计1次",
        })
    dedup = pd.DataFrame(rows)
    dedup.to_csv(CHK / "去重对照表.csv", index=False, encoding="utf-8-sig")
    n_groups = dedup["同义组号"].nunique()
    n_built_groups = dedup[dedup["是否已构造"]=="是"]["同义组号"].nunique()

    # 统计
    stats = {
        "主表行数": int(n),
        "候选216": 216,
        "禁止入模42": 42,
        "红线跳过seats": 2,
        "待构造214": 214,
        "已构造特征数": len(present),
        "未实现特征数": len(missing),
        "同义组数_全待构造": int(n_groups),
        "同义组数_已构造至少1个": int(n_built_groups),
        "去重后计数_已构造代表": int(dedup[(dedup["成员特征"]==dedup["代表特征"]) & (dedup["是否已构造"]=="是")].shape[0]),
        "特征矩阵": str(out),
        "特征矩阵md5": md5file(out),
        "用时秒": round(time.time()-t0),
        "未实现列表": missing,
    }
    (CHK / "组装统计.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"05_组装 done feats={len(present)} missing={len(missing)} groups={n_groups}")
    mark_done("05_组装", stats)
    c.close()
    return stats

# ---------------- 6 说明 + md5 ----------------
def phase6(stats: dict | None = None):
    if stats is None and (CHK / "组装统计.json").exists():
        stats = json.loads((CHK / "组装统计.json").read_text(encoding="utf-8"))
    keep = pd.read_csv(CHK / "待构造特征清单_剔禁止与seats.csv")
    cov = pd.read_csv(CHK / "特征覆盖率.csv") if (CHK / "特征覆盖率.csv").exists() else pd.DataFrame()
    missing = pd.read_csv(CHK / "未实现特征.csv") if (CHK / "未实现特征.csv").exists() else pd.DataFrame()
    dedup = pd.read_csv(CHK / "去重对照表.csv")

    need_hist = int((cov["是否需航管历史"]=="是").sum()) if len(cov) else 0
    low_cov = cov[cov["覆盖率"] < 0.5][["特征名","覆盖率"]].to_dict("records") if len(cov) else []

    doc = f"""# 11-02 新模型重训 · S1 特征构造说明

- 生成：2026-11-02（上海时间 / Asia/Shanghai）；**只做 S1 特征构造 + 同义去重记账，未做有用性分析、未训练**
- 授权：Yangyang 本人 t1350u「新模型现在可以开工了」；候选池 v3 md5 `b4072fc86324e76ffb2b39eb49b57b03`
- 红线落实：未碰线上推理服务 / [REDACTED_SERVICE]；未写服务器库；未读 ana_extract.py、pull8001.py；未打印 store.json；未动 AF；**seats 两列未构造**

## 1. 数量

| 项 | 数 |
|---|---|
| 候选池总行 | 258 |
| 候选特征 | **216** |
| 禁止入模列（质检 v2.1/v2.4 泄露分类） | **42**（未构造） |
| 红线跳过 seats | **2**（`seats当前仓位余票`、`seats变化`） |
| 实际待构造 | **214** |
| 本阶段已写入特征矩阵的列 | **{stats.get('已构造特征数') if stats else '?'}** |
| 未实现（见核对/未实现特征.csv） | **{stats.get('未实现特征数') if stats else '?'}** |
| 同义组数（去重后计数口径） | **{stats.get('同义组数_全待构造') if stats else '?'}** |
| 去重后已构造代表数 | **{stats.get('去重后计数_已构造代表') if stats else '?'}** |

主表键行数：{stats.get('主表行数') if stats else '6,422,519'}（与 S0 主表一致）。

## 2. 输入

- S0 主表：`../S0_底表/数据/主表_全航司日价_dtd0_30.parquet`（观察日 10-15..11-01，dtd 0..30）
- 扩样本（仅去年/训练分位/R1–R6 等需要时）：`../S0_底表/数据/扩样本表_航管2025与2026新_对齐_dtd0_30_含子来源.parquet` 中 **2025-06..12** 段
- 候选池：`特征候选池_11-02_20261028/淡旺季补充_20261102/特征候选池_去重后_v3_20261102_淡旺季补充.csv`
- 淡旺季日历：`淡旺季日历_2025_2027.csv`（v3）
- 公式复用：v2.1 `构建重训数据集_v2.1.py`、fb01/fb01c、v2_01_newfeats、s03_batch2、s03_route_season_check

## 3. 产出

| 文件 | 说明 |
|---|---|
| `数据/特征矩阵_全航司_dtd0_30.parquet` | 键 + 已构造特征；md5 见 md5清单 |
| `核对/去重对照表.csv` | 同义/可推出组合并组；有用性分析时同组只计 1 次 |
| `核对/特征覆盖率.csv` | 各特征非空率、是否依赖航管历史 |
| `核对/未实现特征.csv` | 本 S1 未能落地的候选 |
| `核对/待构造特征清单_剔禁止与seats.csv` | 214 行 |
| `脚本/构建特征_S1.py` | 本构造脚本（可断点续跑） |

## 4. 口径要点（防泄露）

1. 自身历史窗：`ROWS ... AND 1 PRECEDING`（特征B）或不含未来观察日；滚动分位含当天及过去观测。
2. 竞争横截面：同观察日同航线快照；特征A 邻居剔 R3 共享且剔廉航 9C/AQ。
3. `ly_*` / `tr_*` / 城市热度：只用扩样本 **2025**（观察日 ≤2025-12-30），整体断言「2025 最晚观察日 < 主表最早观察日」已通过。
4. R2/R5 分母：因 2025 整体早于样本，等价于固定区间 2025-06-01..12-31（与 v3 本地核验一致）。
5. **未在测试集上挑特征**；同义去重按候选池查重/公式关系，不是按标签相关。

## 5. 覆盖摘要

- 需航管历史的已构造列数：{need_hist}
- 覆盖率 <50% 的特征（冷启动/季节缺失等）：{json.dumps(low_cov, ensure_ascii=False)[:1500]}

## 6. 风险（只记录，S1 不修）

1. **样本无春运**：`dep_is_chunyun` 在主表起飞日上恒 0；春运信号无法在本窗学习。
2. **起飞月/暑运几乎全靠 2025 扩样本**：主表起飞日约 10-16..12-01，`dep_is_shuyun` 恒 0；R1–R5 / ly_* 的变异主要来自去年，有「年份当成季节」的跨期过拟合风险。
3. 主表仅 18 个观察日：自身滚动窗、近 N 天斜率在早期观察日大量缺失（冷启动）。
4. `ly_mean` 本版用 2025-06..12（比现网 9–12 更宽），与现网 serving 不完全一致，入模前需对齐或对照。
5. `g3_exit_7d`、部分 argmin_* / n_post / n_stale 等未完整复现旧脚本细节（见未实现表）。
6. seats 永禁。

## 7. 下一步

S2 有用性分析（净省钱尺子、同义组只计一次、盯春运缺失与年份-季节绑死风险）。**未授权前不开 S2。**
"""
    (ROOT / "特征说明.md").write_text(doc, encoding="utf-8")

    # md5 清单
    paths = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.name not in ("构建.log",) and "_DONE_" not in p.name and "duck_tmp" not in str(p):
            if p.suffix in (".parquet", ".csv", ".md", ".json", ".py", ".txt"):
                paths.append(p)
    lines = ["# S1 md5清单", f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai", ""]
    for p in paths:
        rel = p.relative_to(ROOT)
        lines.append(f"{md5file(p)}  {rel}")
    (ROOT / "md5清单.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("06 说明与md5 完成")
    mark_done("06_文档", {"md5清单": str(ROOT / 'md5清单.txt')})

def main():
    log("==== S1 start ====")
    phase0()
    phase1()
    phase2()
    # phase3 用高效脚本（窗口聚合，避免自连接卡死）
    if not phase_done("03_竞争横截面"):
        import subprocess
        script = ROOT / "脚本" / "构建特征_S1_phase3高效.py"
        log(f"run {script}")
        r = subprocess.run([str(Path(os.environ.get("VENV_PYTHON", "python3"))), "-u", str(script)], check=False)
        if r.returncode != 0 or not phase_done("03_竞争横截面"):
            raise SystemExit(f"phase3 failed rc={r.returncode}")
    else:
        log("skip 03_竞争横截面")
    phase4()
    stats = phase5()
    phase6(stats)
    log("==== S1 all done ====")

if __name__ == "__main__":
    main()
