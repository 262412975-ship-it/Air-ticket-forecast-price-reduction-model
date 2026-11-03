#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 A1 双影子记账（尖端 0.7493 + 宽档 0.62），主报扣100后 k1。

授权：Yangyang t1402（经经理 Grok 转达）。仅离线影子；不碰线上推理服务 / 不上线 / 不真收 /
不写服务器库 / 不改现网阈值 / 不重训。通过≠上线。

读法B：入场 dtd≥14；窗=7；卖出须在起飞前7天前完成（入场 dtd≥14 ⇒ 窗内结算时 dtd≥7）。
PRICE_BASIS=last（S0 当日选批规则=末批）。通程 len(航班号)==7 剔除；回补键=否。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "影子记账_扣100_k1_尖端与0.62_20261102"
S0 = ROOT / "S0_底表/数据/主表_全航司日价_dtd0_30.parquet"
S1 = ROOT / "S1_特征/数据/特征矩阵_全航司_dtd0_30.parquet"
S2_LAB = ROOT / "S2_有用性/数据/标签_全航司_满窗.parquet"
MODEL = ROOT / "S3_训练/模型/model_A1_默认38加晋升ly_seed42.txt"
A_JSON = ROOT / "S3_训练/中间/A版定案.json"

DTD_MIN = 14
W = 7
DROP_YUAN, DROP_RATIO = 100.0, 0.15
FEE = 100.0  # 每单先扣 100 再算主报 k1
THR_TIP = 0.7493
THR_WIDE = 0.62
MODEL_MD5_EXPECT = "f4a120a953503ca5d1d82f41913aacd9"

ACCOUNTS = [
    {"账本名": "尖端_阈值0.7493", "阈值": THR_TIP, "文件名": "尖端_阈值0.7493_日记账.csv",
     "来源说明": "对齐 S3 A1 验证 tip≈0.5%（报告阈值 0.749324…，本包写死 0.7493）；同为单日 VAL tip，有过拟合风险"},
    {"账本名": "宽档_阈值0.62", "阈值": THR_WIDE, "文件名": "宽档_阈值0.62_日记账.csv",
     "来源说明": "每单扣100后、单日 VAL 上合计 k1 峰附近 ~4.6% 那一档；写死 0.62；过拟合风险"},
]


def md5(p: Path | str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    OUT.joinpath("结果").mkdir(parents=True, exist_ok=True)
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "结果/运行日志.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def make_X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = pd.Categorical(X[c].astype(str)).codes.astype(np.int32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(np.float32)


def list_obs_days(con) -> list[str]:
    df = con.execute(f"""
        SELECT DISTINCT CAST(观察日 AS VARCHAR) d
        FROM read_parquet('{S1}')
        ORDER BY 1
    """).df()
    return df["d"].tolist()


def max_price_day(con) -> str:
    return con.execute(f"""
        SELECT CAST(max(观察日) AS VARCHAR) FROM read_parquet('{S0}')
    """).fetchone()[0]


def load_day_candidates(con, day: str, feats: list[str]) -> pd.DataFrame:
    feat_sql = ", ".join(f'f."{c}"' for c in feats)
    df = con.execute(f"""
        SELECT CAST(f.观察日 AS DATE) AS 观察日,
               f.航班号, f.航线, CAST(f.起飞日 AS DATE) AS 起飞日,
               f.航司, f.距起飞天数,
               CAST(f.价格 AS DOUBLE) AS 基准价,
               coalesce(s.是否回补键, '否') AS 是否回补键,
               coalesce(s.是否满7天窗口, '否') AS 是否满7天窗口,
               {feat_sql}
        FROM read_parquet('{S1}') f
        LEFT JOIN (
            SELECT 观察日, 航班号, 航线, 起飞日, 是否回补键, 是否满7天窗口
            FROM read_parquet('{S0}')
            WHERE 观察日 = DATE '{day}'
        ) s
          ON f.观察日=s.观察日 AND f.航班号=s.航班号 AND f.航线=s.航线 AND f.起飞日=s.起飞日
        WHERE f.观察日 = DATE '{day}'
          AND f.距起飞天数 >= {DTD_MIN}
          AND length(CAST(f.航班号 AS VARCHAR)) <> 7
    """).df()
    if len(df) == 0:
        return df
    df = df.loc[df["是否回补键"].astype(str) != "是"].copy()
    return df.reset_index(drop=True)


def attach_s2_labels(con, day: str, df: pd.DataFrame) -> pd.DataFrame:
    lab = con.execute(f"""
        SELECT CAST(obs_day AS DATE) AS 观察日,
               flight_no AS 航班号, route AS 航线, CAST(dep_date AS DATE) AS 起飞日,
               CAST(降价收益_若收单 AS DOUBLE) AS 降价收益_若收单,
               CAST(涨价亏损_若收单 AS DOUBLE) AS 涨价亏损_若收单,
               CAST(每单净省钱_k1 AS DOUBLE) AS 原k1_标签,
               n_win, 标签状态
        FROM read_parquet('{S2_LAB}')
        WHERE obs_day = DATE '{day}'
    """).df()
    if len(lab) == 0:
        df["降价收益_若收单"] = np.nan
        df["涨价亏损_若收单"] = np.nan
        df["结算来源"] = ""
        return df
    out = df.merge(lab, on=["观察日", "航班号", "航线", "起飞日"], how="left")
    out["结算来源"] = np.where(out["降价收益_若收单"].notna(), "S2标签", "")
    return out


def self_settle_missing(con, day: str, df: pd.DataFrame) -> pd.DataFrame:
    """对尚无 S2 标签、但日历窗已满的行，按读法B自算降价收益/涨价亏损。"""
    need = df["降价收益_若收单"].isna()
    if not need.any():
        return df
    # 日历窗是否满：观察日+W ≤ 底表最大观察日
    mx = max_price_day(con)
    day_ts = pd.Timestamp(day)
    mx_ts = pd.Timestamp(mx)
    window_full_calendar = (day_ts + pd.Timedelta(days=W)) <= mx_ts
    if not window_full_calendar:
        return df

    sub = df.loc[need, ["航班号", "航线", "起飞日", "基准价"]].copy()
    sub = sub.drop_duplicates()
    if len(sub) == 0:
        return df
    tmp = OUT / "中间" / f"_tmp_need_settle_{day}.parquet"
    sub.to_parquet(tmp, index=False)
    settled = con.execute(f"""
        WITH base AS (
          SELECT 航班号 AS flight_no, 航线 AS route, CAST(起飞日 AS DATE) AS dep_date,
                 CAST(基准价 AS DOUBLE) AS p0
          FROM read_parquet('{tmp}')
        ),
        j AS (
          SELECT b.flight_no, b.route, b.dep_date, b.p0,
                 p.观察日 AS bd, CAST(p.价格 AS DOUBLE) AS bp,
                 date_diff('day', DATE '{day}', CAST(p.观察日 AS DATE)) AS h
          FROM base b
          JOIN read_parquet('{S0}') p
            ON p.航班号=b.flight_no AND p.航线=b.route AND CAST(p.起飞日 AS DATE)=b.dep_date
           AND CAST(p.观察日 AS DATE) > DATE '{day}'
           AND CAST(p.观察日 AS DATE) <= DATE '{day}' + INTERVAL {W} DAY
           AND p.价格 > 0
        ),
        g AS (
          SELECT flight_no, route, dep_date, p0,
                 count(*) AS n_win,
                 arg_min(bp, bd) FILTER (
                   WHERE bp <= p0 - {DROP_YUAN} OR bp <= p0 * {1.0 - DROP_RATIO}
                 ) AS p_drop,
                 arg_max(bp, bd) AS p_end
          FROM j
          GROUP BY 1,2,3,4
        )
        SELECT flight_no AS 航班号, route AS 航线, dep_date AS 起飞日,
               n_win,
               CASE WHEN p_drop IS NOT NULL THEN p0 - p_drop ELSE 0 END AS 降价收益_若收单,
               CASE WHEN p_drop IS NULL THEN greatest(p_end - p0, 0) ELSE 0 END AS 涨价亏损_若收单
        FROM g
        WHERE n_win > 0
    """).df()
    try:
        tmp.unlink()
    except OSError:
        pass
    if len(settled) == 0:
        return df
    # 填回
    df = df.copy()
    m = df.merge(
        settled.rename(columns={
            "降价收益_若收单": "_g", "涨价亏损_若收单": "_l", "n_win": "_n",
        }),
        on=["航班号", "航线", "起飞日"], how="left",
    )
    fill = m["降价收益_若收单"].isna() & m["_g"].notna()
    m.loc[fill, "降价收益_若收单"] = m.loc[fill, "_g"]
    m.loc[fill, "涨价亏损_若收单"] = m.loc[fill, "_l"]
    m.loc[fill, "结算来源"] = "自算读法B"
    if "n_win" in m.columns:
        m.loc[fill, "n_win"] = m.loc[fill, "_n"]
    else:
        m["n_win"] = m["_n"]
    return m.drop(columns=[c for c in m.columns if c.startswith("_")])


def summarize_day(day: str, acc_name: str, thr: float, sel: pd.DataFrame) -> dict:
    n = len(sel)
    settled = sel["降价收益_若收单"].notna()
    n_set = int(settled.sum())
    n_unset = int((~settled).sum())
    if n_set > 0:
        g = sel.loc[settled, "降价收益_若收单"].to_numpy(float)
        l = sel.loc[settled, "涨价亏损_若收单"].to_numpy(float)
        k1_raw = g - l
        k1_fee = k1_raw - FEE
        win = int((k1_fee > 0).sum())
        lose = int((k1_fee < 0).sum())
        flat = int((k1_fee == 0).sum())
        row = {
            "观察日": day,
            "账本名": acc_name,
            "阈值": thr,
            "入选条数": n,
            "已结算条数": n_set,
            "未结算条数": n_unset,
            "扣后k1合计": float(k1_fee.sum()),
            "扣后k1每单": float(k1_fee.mean()),
            "原k1合计": float(k1_raw.sum()),
            "原k1每单": float(k1_raw.mean()),
            "赚钱条数": win,
            "亏钱条数": lose,
            "打平条数": flat,
            "窗是否已满可结算": "是" if n_unset == 0 else ("部分" if n_set > 0 else "否"),
            "备注": "",
        }
    else:
        row = {
            "观察日": day,
            "账本名": acc_name,
            "阈值": thr,
            "入选条数": n,
            "已结算条数": 0,
            "未结算条数": n,
            "扣后k1合计": np.nan,
            "扣后k1每单": np.nan,
            "原k1合计": np.nan,
            "原k1每单": np.nan,
            "赚钱条数": 0,
            "亏钱条数": 0,
            "打平条数": 0,
            "窗是否已满可结算": "否",
            "备注": "未满窗或窗内无后续价，未计入主累计",
        }
    if n_set > 0 and n_unset > 0:
        row["备注"] = f"已结算{n_set}/入选{n}；未结算不入主累计"
    return row


def process_day(con, model, feats, day: str) -> list[dict]:
    log(f"处理观察日 {day}")
    df = load_day_candidates(con, day, feats)
    if len(df) == 0:
        log(f"  无候选")
        rows = []
        for acc in ACCOUNTS:
            rows.append({
                "观察日": day, "账本名": acc["账本名"], "阈值": acc["阈值"],
                "入选条数": 0, "已结算条数": 0, "未结算条数": 0,
                "扣后k1合计": np.nan, "扣后k1每单": np.nan,
                "原k1合计": np.nan, "原k1每单": np.nan,
                "赚钱条数": 0, "亏钱条数": 0, "打平条数": 0,
                "窗是否已满可结算": "否", "备注": "无候选",
            })
        return rows

    X = make_X(df, feats)
    df["分数"] = model.predict(X)
    df = attach_s2_labels(con, day, df)
    df = self_settle_missing(con, day, df)

    # 结算时距起飞：入场 dtd - 结算偏移；满窗末最差为 dtd-W；要求 ≥7，读法B入场≥14 已保证
    df["原k1"] = df["降价收益_若收单"] - df["涨价亏损_若收单"]
    df["扣后k1"] = df["原k1"] - FEE

    day_dir = OUT / "中间" / f"观察日_{day}"
    day_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for acc in ACCOUNTS:
        thr = acc["阈值"]
        sel = df.loc[df["分数"] >= thr].copy()
        # 入选明细
        cols = ["观察日", "航班号", "航线", "起飞日", "航司", "距起飞天数", "基准价",
                "分数", "降价收益_若收单", "涨价亏损_若收单", "原k1", "扣后k1",
                "结算来源", "是否满7天窗口", "是否回补键"]
        for c in cols:
            if c not in sel.columns:
                sel[c] = np.nan
        detail_path = day_dir / f"{acc['账本名']}_入选明细.parquet"
        sel[cols].to_parquet(detail_path, index=False)
        # 也写一份 csv 便于抽查（仅前若太大则只 parquet；这里每日 tip 约千级、宽档约万级可 csv）
        sel[cols].to_csv(day_dir / f"{acc['账本名']}_入选明细.csv", index=False, encoding="utf-8-sig")
        rows.append(summarize_day(day, acc["账本名"], thr, sel))
        log(f"  {acc['账本名']}: 入选={len(sel)} 已结算={rows[-1]['已结算条数']} "
            f"扣后k1合计={rows[-1]['扣后k1合计']}")
    # 全日打分摘要（不按阈值）便于复核
    df[["观察日", "航班号", "航线", "起飞日", "距起飞天数", "基准价", "分数",
        "降价收益_若收单", "涨价亏损_若收单", "结算来源"]].to_parquet(
        day_dir / "全日打分_dtd14.parquet", index=False
    )
    return rows


def write_ledgers(all_rows: list[dict]):
    OUT.joinpath("账本").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["账本名", "观察日"]).reset_index(drop=True)
    for acc in ACCOUNTS:
        sub = df.loc[df["账本名"] == acc["账本名"]].copy()
        # 主累计只加已结算
        sub["累计入选条数"] = sub["入选条数"].cumsum()
        set_k1 = sub["扣后k1合计"].fillna(0)
        # 仅已结算日累加：用已结算条数>0 的扣后合计
        mask_set = sub["已结算条数"] > 0
        cum = []
        s = 0.0
        for ok, v in zip(mask_set, sub["扣后k1合计"]):
            if ok and pd.notna(v):
                s += float(v)
            cum.append(s if s != 0 or (ok and pd.notna(v) and float(v) == 0) else (s if any(cum) else np.nan))
        # cleaner cum
        s = 0.0
        cum = []
        has = False
        for ok, v in zip(mask_set, sub["扣后k1合计"]):
            if ok and pd.notna(v):
                s += float(v)
                has = True
            cum.append(s if has else np.nan)
        sub["累计扣后k1_仅已结算"] = cum
        s2 = 0.0
        cum2 = []
        has2 = False
        for ok, v in zip(mask_set, sub["原k1合计"]):
            if ok and pd.notna(v):
                s2 += float(v)
                has2 = True
            cum2.append(s2 if has2 else np.nan)
        sub["累计原k1_仅已结算"] = cum2
        path = OUT / "账本" / acc["文件名"]
        sub.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"写日记账 {path.name} 行={len(sub)}")

    # 累计摘要：两账并列
    summary_rows = []
    for acc in ACCOUNTS:
        sub = df.loc[df["账本名"] == acc["账本名"]]
        mask_set = sub["已结算条数"] > 0
        summary_rows.append({
            "账本名": acc["账本名"],
            "阈值": acc["阈值"],
            "来源说明": acc["来源说明"],
            "覆盖观察日数": int(sub["观察日"].nunique()),
            "首个观察日": str(sub["观察日"].min()) if len(sub) else "",
            "末个观察日": str(sub["观察日"].max()) if len(sub) else "",
            "入选条数合计": int(sub["入选条数"].sum()),
            "已结算条数合计": int(sub["已结算条数"].sum()),
            "未结算条数合计": int(sub["未结算条数"].sum()),
            "已结算观察日数": int(mask_set.sum()),
            "未结算观察日数": int((~mask_set).sum()),
            "扣后k1累计_仅已结算": float(sub.loc[mask_set, "扣后k1合计"].sum()) if mask_set.any() else np.nan,
            "扣后k1每单_已结算均": float(
                sub.loc[mask_set, "扣后k1合计"].sum() / sub.loc[mask_set, "已结算条数"].sum()
            ) if mask_set.any() and sub.loc[mask_set, "已结算条数"].sum() > 0 else np.nan,
            "原k1累计_仅已结算": float(sub.loc[mask_set, "原k1合计"].sum()) if mask_set.any() else np.nan,
            "原k1每单_已结算均": float(
                sub.loc[mask_set, "原k1合计"].sum() / sub.loc[mask_set, "已结算条数"].sum()
            ) if mask_set.any() and sub.loc[mask_set, "已结算条数"].sum() > 0 else np.nan,
            "赚钱条数_已结算": int(sub.loc[mask_set, "赚钱条数"].sum()) if mask_set.any() else 0,
            "亏钱条数_已结算": int(sub.loc[mask_set, "亏钱条数"].sum()) if mask_set.any() else 0,
            "打平条数_已结算": int(sub.loc[mask_set, "打平条数"].sum()) if mask_set.any() else 0,
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(OUT / "账本/累计摘要.csv", index=False, encoding="utf-8-sig")
    log("写累计摘要.csv")
    return df


def write_md5_list():
    lines = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "md5清单.txt" and "__pycache__" not in str(p):
            rel = p.relative_to(OUT)
            lines.append(f"{md5(p)}  {rel}")
    text = "\n".join(lines) + "\n"
    (OUT / "md5清单.txt").write_text(text, encoding="utf-8")
    log(f"md5清单 {len(lines)} 项")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="", help="只跑单个观察日 YYYY-MM-DD；默认回填全部可打分日")
    ap.add_argument("--from-day", default="", help="回填起始日（含）")
    ap.add_argument("--to-day", default="", help="回填截止日（含）")
    args = ap.parse_args()

    (OUT / "结果").mkdir(parents=True, exist_ok=True)
    (OUT / "中间").mkdir(parents=True, exist_ok=True)
    (OUT / "账本").mkdir(parents=True, exist_ok=True)

    # 清空本轮日志头
    with open(OUT / "结果/运行日志.txt", "a", encoding="utf-8") as f:
        f.write("\n===== 新跑 " + time.strftime("%Y-%m-%d %H:%M:%S") + " =====\n")

    m = md5(MODEL)
    if m != MODEL_MD5_EXPECT:
        raise SystemExit(f"模型 md5 不符: got {m} expect {MODEL_MD5_EXPECT}")
    feats = json.load(open(A_JSON, encoding="utf-8"))["最终A特征"]
    assert len(feats) == 46, len(feats)
    model = lgb.Booster(model_file=str(MODEL))
    log(f"模型 md5={m} 特征={len(feats)} trees={model.num_trees()}")

    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='10GB'")
    days = list_obs_days(con)
    mx = max_price_day(con)
    log(f"S1 观察日 {days[0]}..{days[-1]} 共{len(days)}；S0 最大观察日={mx}")

    if args.day:
        days = [args.day]
    else:
        if args.from_day:
            days = [d for d in days if d >= args.from_day]
        if args.to_day:
            days = [d for d in days if d <= args.to_day]

    # 若非单日增量：重跑范围内日记账
    meta = {
        "授权": "Yangyang t1402",
        "模型": str(MODEL),
        "模型md5": m,
        "特征数": len(feats),
        "阈值_尖端": THR_TIP,
        "阈值_宽档": THR_WIDE,
        "每单扣费": FEE,
        "读法B_dtd下限": DTD_MIN,
        "窗": W,
        "PRICE_BASIS": "last/末批",
        "S1最大观察日": days[-1] if days else "",
        "S0最大观察日": mx,
        "起始观察日_本包": days[0] if days else "",
        "说明": "主累计只含已结算；未满窗标未结算；11-02及之后等特征增量后再追加",
    }
    json.dump(meta, open(OUT / "中间/运行元信息.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    all_rows: list[dict] = []
    # 若全量回填，合并已有范围外的日记账？本脚本默认重写指定范围；全量则覆盖
    ledger_cache = {}
    if args.day or args.from_day or args.to_day:
        for acc in ACCOUNTS:
            p = OUT / "账本" / acc["文件名"]
            if p.exists():
                old = pd.read_csv(p)
                for _, r in old.iterrows():
                    d = str(r["观察日"])[:10]
                    if args.day and d == args.day:
                        continue
                    if args.from_day and args.to_day and args.from_day <= d <= args.to_day:
                        continue
                    if args.from_day and not args.to_day and d >= args.from_day:
                        continue
                    if args.to_day and not args.from_day and d <= args.to_day:
                        continue
                    ledger_cache[(r["账本名"], d)] = r.to_dict()

    for day in days:
        rows = process_day(con, model, feats, day)
        for r in rows:
            ledger_cache[(r["账本名"], r["观察日"])] = r
        all_rows.extend(rows)

    # 合成全部 cache
    merged = list(ledger_cache.values())
    write_ledgers(merged)

    # 对齐核对：测试日尖端应≈1205（S2 可结算宇宙）
    tip_1024 = OUT / "中间/观察日_2026-10-24/尖端_阈值0.7493_入选明细.parquet"
    if tip_1024.exists():
        t = pd.read_parquet(tip_1024)
        n_all = len(t)
        n_s2 = int(t["结算来源"].eq("S2标签").sum()) if "结算来源" in t.columns else -1
        chk = {
            "观察日": "2026-10-24",
            "尖端入选_全部候选": n_all,
            "尖端入选_S2可结算": n_s2,
            "S3A1测试_阈值出手参考": 1205,
            "对齐说明": "与 S3 A1 测试_阈值出手逐字节对齐的是「联结/标签宇宙」；"
                       "本影子对 S1 全日 dtd≥14 打分，无后续价的行会入选但未结算",
        }
        json.dump(chk, open(OUT / "中间/对齐核对_测试日尖端.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        log(f"对齐核对 10-24 尖端入选={n_all} 其中S2结算={n_s2} 参考1205")

    write_md5_list()
    log("完成")
    con.close()


if __name__ == "__main__":
    main()
