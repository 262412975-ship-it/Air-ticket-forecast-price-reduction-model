#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仅离线：为观察日 2026-11-02 构造 A1 的 46 特征并打尖端分。
不改 S0/S1 主文件；不碰线上推理服务；不上线；不真收。
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import os

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "尖端预测_最新_20261102"
MID = OUT / "中间"
RES = OUT / "结果"
LOG = OUT / "日志"
S0 = ROOT / "S0_底表/数据/主表_全航司日价_dtd0_30.parquet"
EXP = ROOT / "S0_底表/数据/扩样本表_航管2025与2026新_对齐_dtd0_30_含子来源.parquet"
DAY = MID / "日价_观察日2026-11-02_dtd0_30.parquet"
MODEL = ROOT / "S3_训练/模型/model_A1_默认38加晋升ly_seed42.txt"
A_JSON = ROOT / "S3_训练/中间/A版定案.json"
THR = 0.749324691221704
OBS = "2026-11-02"
MODEL_MD5_EXPECT = "f4a120a953503ca5d1d82f41913aacd9"
DTD_MIN = 14
廉航 = ("9C", "AQ")
邻时段分钟 = 180
上一观察最大间隔天 = 3

for p in (MID, RES, LOG):
    p.mkdir(parents=True, exist_ok=True)


def md5(p: Path | str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG / "运行日志.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def make_X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = pd.Categorical(X[c].astype(str)).codes.astype(np.int32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(np.float32)


def con_new(mem="6GB", threads=2):
    tmp = MID / "duck_tmp"
    tmp.mkdir(exist_ok=True)
    c = duckdb.connect()
    c.execute(
        f"SET memory_limit='{mem}'; SET threads={threads}; "
        f"SET temp_directory='{tmp}'; SET preserve_insertion_order=false"
    )
    return c


def build_features() -> Path:
    out_feat = MID / "特征_观察日2026-11-02_A1所需.parquet"
    if out_feat.exists() and out_feat.stat().st_size > 1000:
        log("reuse features", out_feat)
        return out_feat

    t0 = time.time()
    c = con_new()
    log("load base = S0 history + day1102")
    # Combined base for feature windows / cross-section. Do NOT write back to S0.
    c.execute(f"""
    CREATE TABLE base AS
    SELECT CAST(观察日 AS DATE) 观察日, 航班号, 航线, CAST(起飞日 AS DATE) 起飞日, 航司,
           距起飞天数, CAST(价格 AS DOUBLE) 价格, 起飞时刻, 出发机场, 到达机场,
           CASE WHEN regexp_matches(起飞时刻, '^[0-9]{{1,2}}:[0-9]{{2}}$')
                THEN CAST(split_part(起飞时刻,':',1) AS INT)*60
                   + CAST(split_part(起飞时刻,':',2) AS INT) END AS mm,
           CAST(起飞日 AS DATE) - DATE '2026-01-01' AS dd
    FROM (
      SELECT 观察日,航班号,航线,起飞日,航司,距起飞天数,价格,起飞时刻,出发机场,到达机场
      FROM read_parquet('{S0}')
      WHERE 价格>0 AND length(航班号)<>7
      UNION ALL
      SELECT 观察日,航班号,航线,起飞日,航司,距起飞天数,价格,起飞时刻,出发机场,到达机场
      FROM read_parquet('{DAY}')
      WHERE 价格>0 AND length(航班号)<>7 AND 是否回补键='否'
    )
    """)
    c.execute("ALTER TABLE base ADD COLUMN abs_min BIGINT")
    c.execute("UPDATE base SET abs_min = CASE WHEN mm IS NOT NULL THEN dd::BIGINT*1440+mm END")
    n_all = c.execute("SELECT count(*) FROM base").fetchone()[0]
    n_day = c.execute(f"SELECT count(*) FROM base WHERE 观察日=DATE '{OBS}'").fetchone()[0]
    log(f"base rows={n_all} day={n_day} ({time.time()-t0:.0f}s)")

    # Restrict heavy work to routes present on OBS (keeps g2/market correct for those routes)
    c.execute(f"""
    CREATE TABLE day_routes AS
    SELECT DISTINCT 航线 FROM base WHERE 观察日=DATE '{OBS}'
    """)
    c.execute("""
    CREATE TABLE base_r AS
    SELECT b.* FROM base b JOIN day_routes r USING (航线)
    """)
    log(f"base_r rows={c.execute('select count(*) from base_r').fetchone()[0]} ({time.time()-t0:.0f}s)")

    # ---- 自身历史 (only emit OBS) ----
    log("phase own-history")
    c.execute("""
    CREATE TABLE seq AS
    WITH t AS (
      SELECT *, lag(价格) OVER w pp, lag(观察日) OVER w po,
             date_diff('day', lag(观察日) OVER w, 观察日) gap_obs,
             row_number() OVER w AS obs_idx
      FROM base_r
      WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日)
    )
    SELECT *,
      (pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3) AS hc,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp <= -0.15 OR 价格-pp <= -100) THEN 1 ELSE 0 END AS ev_dn,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp >= 0.10 OR 价格-pp >= 80) THEN 1 ELSE 0 END AS ev_up,
      CASE WHEN pp IS NOT NULL THEN 价格/pp - 1 END AS pct
    FROM t
    """)
    c.execute(f"""
    CREATE TABLE own AS
    SELECT 观察日,航班号,航线,起飞日,价格,距起飞天数,航司,出发机场,到达机场,起飞时刻,mm,abs_min,dd,
      CASE WHEN po IS NOT NULL AND gap_obs <= {上一观察最大间隔天} THEN 价格 - pp END AS 上一观察价差_元,
      CASE WHEN pp IS NOT NULL THEN ln(1 + pp) END AS log_price_prev,
      quantile_cont(价格, 0.2) OVER w7 AS roll7_p20,
      max(价格) OVER w7 AS roll7_max,
      quantile_cont(价格, 0.8) OVER w14 AS roll14_p80,
      CASE WHEN (max(价格) OVER w14) > (quantile_cont(价格, 0.2) OVER w14) THEN
        (价格 - quantile_cont(价格, 0.2) OVER w14)
        / nullif(max(价格) OVER w14 - quantile_cont(价格, 0.2) OVER w14, 0) END AS roll14_rank,
      last_value(CASE WHEN ev_up=1 THEN pct END IGNORE NULLS) OVER w_all AS own_last_bigup_pct,
      max(价格) OVER w_all AS peak,
      CASE WHEN (avg(价格) OVER w_all)>0 THEN 价格/(avg(价格) OVER w_all) - 1 END AS trend
    FROM seq
    WINDOW
      w7 AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
      w14 AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN 13 PRECEDING AND CURRENT ROW),
      w_all AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    c.execute(f"CREATE TABLE own_day AS SELECT * FROM own WHERE 观察日=DATE '{OBS}'")
    log(f"own_day={c.execute('select count(*) from own_day').fetchone()[0]} ({time.time()-t0:.0f}s)")

    # ---- 竞争横截面 for OBS only (+ history for g2/sm) ----
    log("phase cross-section / 特征A")
    # codeshare sm: only pairs that appear with same abs_min on OBS; check historical equality
    c.execute(f"""
    CREATE TABLE smday AS
    SELECT c.观察日, c.航线, c.航班号 fc, p.航班号 fp, c.起飞日, (c.价格=p.价格)::INT eq
    FROM base_r c JOIN base_r p
      ON c.观察日=p.观察日 AND c.航线=p.航线 AND c.起飞日=p.起飞日
     AND c.abs_min=p.abs_min AND c.航班号<>p.航班号
    WHERE c.abs_min IS NOT NULL
      AND c.观察日 <= DATE '{OBS}'
      AND EXISTS (
        SELECT 1 FROM own_day d
        WHERE d.航线=c.航线 AND d.起飞日=c.起飞日 AND d.abs_min IS NOT NULL
      )
    """)
    # The EXISTS above still scans heavy. Narrow: only obs days for keys on day.
    # Rebuild more tightly.
    c.execute("DROP TABLE smday")
    c.execute(f"""
    CREATE TABLE day_keys_time AS
    SELECT DISTINCT 航线, 起飞日 FROM own_day WHERE abs_min IS NOT NULL
    """)
    c.execute("""
    CREATE TABLE smday AS
    SELECT c.观察日, c.航线, c.航班号 fc, p.航班号 fp, c.起飞日, (c.价格=p.价格)::INT eq
    FROM base_r c
    JOIN day_keys_time k ON k.航线=c.航线 AND k.起飞日=c.起飞日
    JOIN base_r p
      ON c.观察日=p.观察日 AND c.航线=p.航线 AND c.起飞日=p.起飞日
     AND c.abs_min=p.abs_min AND c.航班号<>p.航班号
    WHERE c.abs_min IS NOT NULL
    """)
    c.execute("""
    CREATE TABLE sm AS
    SELECT *, min(eq) OVER w 全同价
    FROM smday
    WINDOW w AS (PARTITION BY 航线,fc,fp,起飞日 ORDER BY 观察日
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    log(f"sm rows={c.execute('select count(*) from sm').fetchone()[0]} ({time.time()-t0:.0f}s)")

    c.execute(f"""
    CREATE TABLE fa AS
    SELECT c.观察日,c.航班号,c.航线,c.起飞日,
      count(*) FILTER (WHERE coalesce(s.全同价,0)<>1) AS n_ex,
      median(p.价格) FILTER (WHERE coalesce(s.全同价,0)<>1) AS med_ex,
      min(p.价格) FILTER (WHERE coalesce(s.全同价,0)<>1) AS min_ex,
      count(*) FILTER (WHERE coalesce(s.全同价,0)<>1 AND p.价格<c.价格) AS lower_ex
    FROM (SELECT * FROM own_day WHERE abs_min IS NOT NULL) c
    JOIN (SELECT * FROM own_day WHERE abs_min IS NOT NULL AND 航司 NOT IN {廉航} AND 价格>0) p
      ON p.航线=c.航线 AND p.dd BETWEEN c.dd-1 AND c.dd+1
     AND abs(p.abs_min-c.abs_min)<={邻时段分钟}
     AND NOT (p.航班号=c.航班号 AND p.起飞日=c.起飞日)
    LEFT JOIN (SELECT * FROM sm WHERE 观察日=DATE '{OBS}') s
      ON s.航线=c.航线 AND s.fc=c.航班号 AND s.起飞日=c.起飞日 AND s.fp=p.航班号 AND p.起飞日=s.起飞日
    GROUP BY 1,2,3,4
    """)
    log(f"fa={c.execute('select count(*) from fa').fetchone()[0]} ({time.time()-t0:.0f}s)")

    # same-day market windows on own_day
    c.execute("""
    CREATE TABLE sd AS
    WITH w AS (
      SELECT 观察日,航班号,航线,起飞日,价格,航司,abs_min,
        count(*) OVER w AS n_sale,
        count(DISTINCT 航司) OVER w AS n_airlines,
        min(价格) OVER w AS min_all,
        max(价格) OVER w AS max_all,
        median(价格) OVER w AS med_all,
        min(价格) OVER (PARTITION BY 观察日,航线,起飞日 ORDER BY 价格
          ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS min_after,
        max(价格) OVER (PARTITION BY 观察日,航线,起飞日 ORDER BY 价格
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS max_before,
        percent_rank() OVER (PARTITION BY 观察日,航线,起飞日 ORDER BY 价格) AS x_rank_pct
      FROM own_day
      WINDOW w AS (PARTITION BY 观察日,航线,起飞日)
    )
    SELECT 观察日,航班号,航线,起飞日,价格,航司,abs_min,
      n_sale-1 AS n_comp, n_airlines,
      CASE WHEN 价格 > min_all THEN min_all ELSE coalesce(min_after, min_all) END AS comp_min,
      CASE WHEN 价格 < max_all THEN max_all ELSE coalesce(max_before, max_all) END AS comp_max,
      med_all AS comp_med, min_all AS min_day, n_sale AS g3_n_sale, x_rank_pct
    FROM w
    """)

    # ±2h hour-bin approx on day
    c.execute("""
    CREATE TABLE pm2h AS
    WITH b2 AS (
      SELECT *, CASE WHEN abs_min IS NOT NULL THEN abs_min // 60 END AS hour_bin
      FROM own_day WHERE abs_min IS NOT NULL
    ),
    w AS (
      SELECT 观察日,航班号,航线,起飞日,价格,hour_bin,
        count(*) OVER p AS n_bin,
        count(DISTINCT 航司) OVER p AS n_air_bin,
        min(价格) OVER p AS min_all,
        min(价格) OVER (PARTITION BY 观察日,航线,起飞日,hour_bin ORDER BY 价格
          ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS min_after
      FROM b2 WINDOW p AS (PARTITION BY 观察日,航线,起飞日,hour_bin)
    )
    SELECT 观察日,航班号,航线,起飞日,
      greatest(n_bin-1,0)::BIGINT AS n_flights_pm2h,
      n_air_bin::BIGINT AS n_airlines_pm2h,
      CASE WHEN 价格 > min_all THEN min_all ELSE coalesce(min_after, min_all) END AS comp_min_pm2h
    FROM w
    """)

    # g2 market shares: need fl over history for routes on day
    c.execute("""
    CREATE TABLE fl AS
    WITH t AS (
      SELECT *, lag(价格) OVER w pp, lag(观察日) OVER w po,
        date_diff('day', lag(观察日) OVER w, 观察日) gap_obs
      FROM base_r
      WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日)
    )
    SELECT 观察日,航班号,航线,起飞日,价格,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 THEN 1 ELSE 0 END has_cmp,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp<=-0.15 OR 价格-pp<=-100) THEN 1 ELSE 0 END dr,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp>=0.10 OR 价格-pp>=80) THEN 1 ELSE 0 END ri
    FROM t
    """)
    c.execute("""
    CREATE TABLE mk AS
    SELECT 航线,起飞日,观察日, count(*) n_sale, sum(has_cmp) n_cmp, sum(dr) n_dr, sum(ri) n_ri
    FROM fl GROUP BY 1,2,3
    """)
    c.execute(f"""
    CREATE TABLE pm AS
    SELECT a.航线,a.起飞日,a.观察日,
      sum(b.n_cmp) n_cmp_pm1, sum(b.n_dr) n_dr_pm1, sum(b.n_ri) n_ri_pm1
    FROM mk a
    JOIN mk b ON a.航线=b.航线 AND b.观察日=a.观察日
     AND b.起飞日 BETWEEN a.起飞日 - INTERVAL 1 DAY AND a.起飞日 + INTERVAL 1 DAY
    WHERE a.观察日=DATE '{OBS}'
    GROUP BY 1,2,3
    """)

    # siblings on day
    c.execute("""
    CREATE TABLE sib AS
    WITH w AS (
      SELECT 观察日,航班号,航线,起飞日,价格,
        count(*) OVER p AS n_all,
        sum(价格) OVER p AS sum_p,
        median(价格) OVER p AS med_p,
        min(价格) OVER p AS min_all,
        min(价格) OVER (PARTITION BY 观察日,航班号,航线 ORDER BY 价格
          ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS min_after
      FROM own_day WINDOW p AS (PARTITION BY 观察日,航班号,航线)
    )
    SELECT 观察日,航班号,航线,起飞日,
      n_all-1 AS n_sib,
      CASE WHEN n_all>1 THEN (sum_p-价格)/(n_all-1) END AS sib_mean,
      med_p AS sib_median,
      CASE WHEN 价格>min_all THEN min_all ELSE coalesce(min_after,min_all) END AS sib_min
    FROM w
    """)
    c.execute("""
    CREATE TABLE odn AS
    SELECT 观察日,航班号,起飞日, count(DISTINCT 航线) n_od_same_flight
    FROM own_day GROUP BY 1,2,3
    """)
    log(f"cross pieces done ({time.time()-t0:.0f}s)")

    # ---- ly / train / dest from 2025 expansion (static lookups) ----
    log("phase ly/train")
    c.execute(f"""
    CREATE TABLE L2025 AS
    SELECT CAST(观察日 AS DATE) 观察日, 航班号, 航线, CAST(起飞日 AS DATE) 起飞日, 航司,
           距起飞天数 AS dtd, CAST(价格 AS DOUBLE) 价格
    FROM read_parquet('{EXP}')
    WHERE 价格>0 AND length(航班号)<>7
      AND CAST(观察日 AS DATE) BETWEEN DATE '2025-06-01' AND DATE '2025-12-30'
      AND CAST(起飞日 AS DATE) BETWEEN DATE '2025-06-01' AND DATE '2025-12-31'
      AND 距起飞天数 BETWEEN 0 AND 30
    """)
    log(f"L2025={c.execute('select count(*) from L2025').fetchone()[0]}")
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
    c.execute("CREATE TABLE rmean AS SELECT 航线, avg(价格) AS ly_route_mean FROM L2025 GROUP BY 1")
    c.execute("CREATE TABLE rcap AS SELECT 航线, avg(nf) AS ly_route_nflights_all FROM rday GROUP BY 1")
    # week price for OBS rows only (efficient join)
    c.execute(f"""
    CREATE TABLE rweek AS
    SELECT s.航班号, s.航线, s.起飞日,
      (SELECT sum(s2)/nullif(sum(n2),0) FROM (
         SELECT sum(价格) s2, count(*) n2 FROM L2025 w
         WHERE w.航线=s.航线
           AND w.起飞日 BETWEEN s.起飞日 - INTERVAL 367 DAY AND s.起飞日 - INTERVAL 361 DAY
       )) AS week_price
    FROM own_day s
    """)
    c.execute("""
    CREATE TABLE train_route AS
    SELECT 航司, 航线,
      quantile_cont(价格,0.2) tr_p20, quantile_cont(价格,0.9) tr_p90
    FROM L2025 GROUP BY 1,2
    """)
    c.execute("""
    CREATE TABLE ly_seq AS
    SELECT 航线, 航班号, 起飞日, 观察日, 价格,
      lag(价格) OVER w pp,
      date_diff('day', lag(观察日) OVER w, 观察日) gap,
      split_part(航线,'-',2) dest
    FROM L2025
    WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日)
    """)
    c.execute("""
    CREATE TABLE city_d AS
    WITH t AS (
      SELECT dest, 观察日, count(*) vol,
        avg(CASE WHEN pp IS NOT NULL AND gap BETWEEN 1 AND 3
                  AND ((价格-pp)/pp<=-0.15 OR 价格-pp<=-100) THEN 1.0 ELSE 0.0 END) drop_rate
      FROM ly_seq GROUP BY 1,2
    )
    SELECT dest, avg(vol) dest_avg_vol, avg(drop_rate) dest_drop_rate FROM t GROUP BY 1
    """)
    log(f"ly lookups ready ({time.time()-t0:.0f}s)")

    # ---- assemble A1 feature frame for OBS ----
    log("assemble")
    c.execute(f"""
    COPY (
      SELECT
        a.观察日, a.航班号, a.航线, a.起飞日, a.航司, a.距起飞天数, a.价格,
        a.出发机场, a.到达机场,
        CASE WHEN a.mm IS NOT NULL THEN a.mm // 60 END AS 起飞小时,
        -- own
        a.上一观察价差_元, a.log_price_prev, a.roll7_p20, a.roll7_max, a.roll14_p80,
        a.own_last_bigup_pct, a.peak, a.trend,
        -- 特征A
        CASE WHEN fa.n_ex>=1 AND fa.med_ex>0 THEN a.价格/fa.med_ex END AS 特征A_比值,
        CASE WHEN fa.n_ex>=1 THEN fa.med_ex END AS 特征A_邻时段中位价,
        CASE WHEN fa.n_ex>=1 THEN CASE WHEN a.价格<=fa.min_ex THEN 1 ELSE 0 END END AS 特征A_是否邻时段最低价,
        -- market
        CASE WHEN sd.min_day>0 THEN a.价格/sd.min_day END AS prev_over_min_day,
        CASE WHEN pm2.comp_min_pm2h IS NOT NULL THEN a.价格-pm2.comp_min_pm2h END AS gap_to_min_pm2h,
        CASE WHEN sd.comp_max IS NOT NULL THEN a.价格-sd.comp_max END AS gap_to_max,
        CASE WHEN sd.comp_min IS NOT NULL THEN a.价格-sd.comp_min END AS gap_to_min_all,
        CASE WHEN pm2.comp_min_pm2h>0 THEN a.价格/pm2.comp_min_pm2h END AS prev_over_min_pm2h,
        CASE WHEN sd.comp_max>0 THEN a.价格/sd.comp_max END AS x_over_max,
        CASE WHEN sd.comp_med>0 THEN a.价格/sd.comp_med END AS x_over_median,
        percent_rank() OVER (PARTITION BY a.观察日, a.航司, a.航线 ORDER BY a.价格) AS 同航司同航线当日价分位,
        sd.g3_n_sale, sd.comp_med, sd.comp_min AS comp_min_all, sd.x_rank_pct, sd.comp_max,
        sd.n_airlines, pm2.n_airlines_pm2h, pm2.n_flights_pm2h,
        CASE WHEN pmm.n_cmp_pm1>0 THEN pmm.n_ri_pm1*1.0/pmm.n_cmp_pm1 END AS g2_mkt_rise_share_pm1,
        CASE WHEN pmm.n_cmp_pm1>0 THEN pmm.n_dr_pm1*1.0/pmm.n_cmp_pm1 END AS g2_mkt_drop_share_pm1,
        -- sib
        CASE WHEN sib.sib_min>0 THEN a.价格/sib.sib_min END AS sib_cur_vs_min_avg,
        CASE WHEN sib.sib_min>0 THEN sib.sib_min/a.价格 END AS sib_drop_ratio,
        CASE WHEN sib.sib_median>0 THEN a.价格/sib.sib_median END AS sib_price_level,
        CASE WHEN sib.sib_mean IS NOT NULL AND sib.n_sib>0 THEN
          CASE WHEN a.价格 > sib.sib_mean THEN 0.0 ELSE 1.0 END END AS sib_low_ratio,
        sib.n_sib, sib.n_sib AS n_sib_same,
        od.n_od_same_flight,
        -- ly
        m.ly_route_depmonth_price,
        CASE WHEN rm.ly_route_mean>0 THEN m.ly_route_depmonth_price/rm.ly_route_mean END AS ly_route_depmonth_ratio,
        CASE WHEN rm.ly_route_mean>0 THEN rw.week_price/rm.ly_route_mean END AS ly_route_depweek_ratio,
        m.ly_route_depmonth_nflights,
        CASE WHEN rc.ly_route_nflights_all>0 THEN m.ly_route_depmonth_nflights/rc.ly_route_nflights_all END AS ly_route_depmonth_cap_ratio,
        cd.dest_avg_vol, cd.dest_drop_rate,
        CASE WHEN tr.tr_p90>tr.tr_p20 THEN least(1, greatest(0, (a.价格 - tr.tr_p20)/nullif(tr.tr_p90-tr.tr_p20,0))) END AS train_pct_pos,
        -- interaction
        a.距起飞天数 * coalesce(a.roll14_rank,
          CASE WHEN tr.tr_p90>tr.tr_p20 THEN least(1, greatest(0, (a.价格 - tr.tr_p20)/nullif(tr.tr_p90-tr.tr_p20,0))) END
        ) AS "距起飞×当前价分位交互"
      FROM own_day a
      LEFT JOIN fa USING (观察日,航班号,航线,起飞日)
      LEFT JOIN sd USING (观察日,航班号,航线,起飞日)
      LEFT JOIN pm2h pm2 USING (观察日,航班号,航线,起飞日)
      LEFT JOIN pm pmm USING (观察日,航线,起飞日)
      LEFT JOIN sib USING (观察日,航班号,航线,起飞日)
      LEFT JOIN odn od ON od.观察日=a.观察日 AND od.航班号=a.航班号 AND od.起飞日=a.起飞日
      LEFT JOIN rmon m ON m.航线=a.航线 AND m.m=month(a.起飞日)
      LEFT JOIN rmean rm ON rm.航线=a.航线
      LEFT JOIN rcap rc ON rc.航线=a.航线
      LEFT JOIN rweek rw ON rw.航班号=a.航班号 AND rw.航线=a.航线 AND rw.起飞日=a.起飞日
      LEFT JOIN train_route tr ON tr.航司=a.航司 AND tr.航线=a.航线
      LEFT JOIN city_d cd ON cd.dest=a.到达机场
      WHERE a.距起飞天数 >= {DTD_MIN}
    ) TO '{out_feat}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = c.execute(f"SELECT count(*) FROM read_parquet('{out_feat}')").fetchone()[0]
    log(f"features written n={n} -> {out_feat} ({time.time()-t0:.0f}s)")
    c.close()
    return out_feat


def score_and_export(feat_path: Path):
    t0 = time.time()
    feats = json.loads(A_JSON.read_text(encoding="utf-8"))["最终A特征"]
    assert len(feats) == 46
    m_md5 = md5(MODEL)
    assert m_md5 == MODEL_MD5_EXPECT, (m_md5, MODEL_MD5_EXPECT)

    # null audit
    con = con_new(mem="3GB", threads=2)
    null_sql = ", ".join([f'avg(({c} IS NULL)::INT) AS n_{i}' for i, c in enumerate(feats)])
    null_row = con.execute(f"SELECT count(*) n, {null_sql} FROM read_parquet('{feat_path}')").fetchone()
    n_all = null_row[0]
    null_rates = {feats[i]: float(null_row[i+1]) for i in range(len(feats))}
    top_null = sorted(null_rates.items(), key=lambda x: -x[1])[:12]
    log("feature rows", n_all, "top null", top_null)
    # hard fail if ALL critical features empty
    critical = ["roll7_p20", "prev_over_min_day", "x_rank_pct", "g3_n_sale", "起飞小时"]
    if any(null_rates.get(c, 1) >= 0.999 for c in critical):
        raise SystemExit(f"critical features empty: {[(c, null_rates[c]) for c in critical]}")

    # score in chunks via duckdb+pandas
    booster = lgb.Booster(model_file=str(MODEL))
    cols_need = ["航班号", "航线", "起飞日", "观察日", "价格", "距起飞天数", "航司"] + feats
    df = con.execute(
        f"SELECT {', '.join('\"'+c+'\"' if any(x in c for x in '×≥') else c for c in cols_need)} "
        f"FROM read_parquet('{feat_path}')"
    ).df()
    con.close()
    log("loaded frame", len(df), list(df.columns)[:10])

    X = make_X(df, feats)
    df["预测分"] = booster.predict(X)
    df["阈值"] = THR
    tip = df.loc[df["预测分"] >= THR].copy()
    tip = tip.sort_values("预测分", ascending=False).reset_index(drop=True)
    log(f"scored {len(df)} tip={len(tip)} ({time.time()-t0:.0f}s)")

    # sanity
    assert (tip["观察日"].astype(str).str[:10] == OBS).all()
    assert (tip["距起飞天数"] >= DTD_MIN).all()
    assert (tip["航班号"].astype(str).str.len() != 7).all()
    assert len(tip.loc[tip["航班号"].astype(str).str.len() == 7]) == 0

    out_csv = OUT / "tip_scores_example.csv"
    export = tip.rename(columns={"价格": "现价"}).copy()
    export["观察日"] = export["观察日"].astype(str).str[:10]
    export["起飞日"] = export["起飞日"].astype(str).str[:10]
    cols = ["航班号", "航线", "起飞日", "观察日", "现价", "预测分", "阈值", "距起飞天数", "航司"]
    export[cols].to_csv(out_csv, index=False, encoding="utf-8-sig", float_format="%.10g")

    # full scored dump
    scored_path = MID / "全日打分_dtd14_观察日2026-11-02.parquet"
    df.to_parquet(scored_path, index=False)

    summary = {
        "观察日": OBS,
        "选用批": "202611021110",
        "当日批次": ["202611020454", "202611021110"],
        "阈值": THR,
        "模型": str(MODEL),
        "模型md5": m_md5,
        "特征数": 46,
        "可打分行_dtd14": int(len(df)),
        "尖端条数": int(len(tip)),
        "通程条数": 0,
        "已剔通程": True,
        "读法B_入场dtd": DTD_MIN,
        "仅预测名单": True,
        "不等于真收": True,
        "不等于上线": True,
        "未碰线上推理服务": True,
        "top_null_rates": top_null,
        "预测分_min_tip": float(tip["预测分"].min()) if len(tip) else None,
        "预测分_max_tip": float(tip["预测分"].max()) if len(tip) else None,
        "tip_scores_example": str(out_csv),
    }
    (MID / "打分摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("summary", summary)
    return summary, out_csv, tip


def write_docs(summary: dict, tip: pd.DataFrame):
    md = f"""# 尖端预测（最新）大白话一页纸

- **观察日**：{summary['观察日']}（禁止用旧日冒充）
- **价格批次**：当日末批 **{summary['选用批']}**（同日还有 {', '.join(summary['当日批次'])}；按 S0「每航司末批、不足70%改最大批」选批，本日全部航司末批够大）
- **模型**：S3 A1（46 特征）`model_A1_默认38加晋升ly_seed42.txt`
- **模型 md5**：`{summary['模型md5']}`（与训练报告一致）
- **尖端阈值**：`{summary['阈值']}`（验证 tip≈前 0.5%）
- **入场口径**：读法B，**dtd≥{DTD_MIN}**；**已剔通程**（`len(航班号)==7` → 本名单通程=0）
- **可打分行**：{summary['可打分行_dtd14']}；**尖端条数**：{summary['尖端条数']}
- **特征**：为 11-02 单独构造并写入本目录 `中间/`，**未改** S0/S1 主文件历史 md5
- **红线**：仅离线预测名单；**≠真收 ≠上线 ≠改阈 ≠开启线上推理服务**

> 金额/赚亏未结算；本文件只给「今天按 A1 尖端阈值会圈到谁」。
"""
    (OUT / "大白话一页纸.md").write_text(md, encoding="utf-8")
    note = f"""# 说明

## 数据
- 日价：`中间/日价_观察日2026-11-02_dtd0_30.parquet`（incremental `batch=202611020454` + `batch=202611021110`）
- 特征：`中间/特征_观察日2026-11-02_A1所需.parquet`（A1 的 46 列；历史窗用既有 S0≤11-01 + 本日日价）
- 打分：`中间/全日打分_dtd14_观察日2026-11-02.parquet`

## 未改动
- 未改 `S0_底表/数据/主表_全航司日价_dtd0_30.parquet`
- 未改 `S1_特征/数据/特征矩阵_全航司_dtd0_30.parquet`
- 未碰线上推理服务 / 不上线 / 不真收 / 不写服务器库

## 空值
见 `中间/打分摘要.json` 的 top_null_rates（与 11-01 S1 量级同阶：own_last_bigup / 特征A / ly_* 允许空）。
"""
    (OUT / "说明.md").write_text(note, encoding="utf-8")


def write_md5():
    paths = [
        OUT / "tip_scores_example.csv",
        OUT / "大白话一页纸.md",
        OUT / "说明.md",
        MID / "日价_观察日2026-11-02_dtd0_30.parquet",
        MID / "特征_观察日2026-11-02_A1所需.parquet",
        MID / "全日打分_dtd14_观察日2026-11-02.parquet",
        MID / "打分摘要.json",
        MODEL,
    ]
    lines = []
    for p in paths:
        if p.exists():
            lines.append(f"{md5(p)}  {p}")
    (OUT / "md5清单.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("md5清单 written")


def main():
    log("START tip predict", OBS)
    assert DAY.exists(), DAY
    assert md5(MODEL) == MODEL_MD5_EXPECT
    feat = build_features()
    summary, csv_path, tip = score_and_export(feat)
    write_docs(summary, tip)
    write_md5()
    log("DONE", "tip_n=", summary["尖端条数"], "csv=", csv_path)


if __name__ == "__main__":
    main()
