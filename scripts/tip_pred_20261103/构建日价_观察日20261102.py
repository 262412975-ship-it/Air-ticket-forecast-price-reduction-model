#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仅离线：为观察日 2026-11-02 用当日全部 4 批重建 dtd0..30 日价。
对齐 S0 选批/回补口径；不写回 S0 主表。
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

import duckdb
import os

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "尖端预测_最新_20261103"
MID = OUT / "中间"
LOG = OUT / "日志"
INC = Path(os.environ["AF_INCREMENTAL_DIR"])
OBS = "2026-11-02"
BATCHES = ["202611020454", "202611021110", "202611021503", "202611022116"]

for p in (MID, LOG, OUT / "结果", OUT / "脚本"):
    p.mkdir(parents=True, exist_ok=True)


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG / "日价构建日志.txt", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def main():
    t0 = time.time()
    files = []
    for b in BATCHES:
        d = INC / f"batch={b}"
        assert (d / "_SUCCESS").exists(), d
        fp = d / "batch.parquet"
        assert fp.exists(), fp
        files.append(str(fp))
    log("batches", BATCHES)

    tmp = MID / "_duck_day"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"SET memory_limit='2GB'; SET threads=2; "
        f"SET temp_directory='{tmp}'; SET preserve_insertion_order=false"
    )

    con.execute(f"""
    CREATE TABLE raw AS
    SELECT flight_no, route, CAST(date AS VARCHAR) dep_date,
           CAST(batch AS VARCHAR) batch,
           TRY_CAST(updated AS TIMESTAMP) obs_ts,
           TRY_CAST(price AS DOUBLE) price,
           time AS dep_time,
           'incremental' AS src
    FROM read_parquet({files!r}, union_by_name=true, hive_partitioning=false)
    """)
    n_raw = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    n_通 = con.execute("SELECT count(*) FROM raw WHERE length(flight_no)=7").fetchone()[0]
    n_bad = con.execute(
        "SELECT count(*) FROM raw WHERE length(flight_no)<>7 AND NOT coalesce(price>0,false)"
    ).fetchone()[0]
    log(f"raw={n_raw} 通程={n_通} 无效价={n_bad}")

    con.execute("""
    CREATE TABLE r AS
    SELECT * FROM raw WHERE length(flight_no)<>7 AND price>0
    """)
    con.execute("DROP TABLE raw")
    con.execute("""
    CREATE TABLE pb AS
    SELECT flight_no, route, dep_date, batch,
           any_value(src) src, min(price) price, count(*) 批内重复,
           arg_min(obs_ts, price) obs_ts, arg_min(dep_time, price) dep_time
    FROM r GROUP BY flight_no, route, dep_date, batch
    """)
    con.execute("DROP TABLE r")
    con.execute("""
    ALTER TABLE pb ADD COLUMN dy VARCHAR;
    UPDATE pb SET dy=substr(batch,1,4)||'-'||substr(batch,5,2)||'-'||substr(batch,7,2);
    ALTER TABLE pb ADD COLUMN al VARCHAR;
    UPDATE pb SET al=substr(flight_no,1,2);
    ALTER TABLE pb ADD COLUMN dtd INT;
    UPDATE pb SET dtd=date_diff('day', CAST(dy AS DATE), CAST(dep_date AS DATE))
    """)
    # filter to OBS day only for selection context (all batches already OBS)
    n_pb = con.execute(f"SELECT count(*) FROM pb WHERE dy='{OBS}'").fetchone()[0]
    log(f"pb OBS rows={n_pb}")

    con.execute("""
    CREATE TABLE bs AS
    SELECT al, dy, batch, count(*) n FROM pb
    WHERE dtd BETWEEN 0 AND 30 AND dy = '""" + OBS + """'
    GROUP BY ALL
    """)
    con.execute("""
    CREATE TABLE sel AS
    WITH m AS (SELECT al, dy, max(n) mx, max(batch) lastb FROM bs GROUP BY al, dy),
         big AS (SELECT al, dy, arg_max(batch, n) bigb FROM bs GROUP BY ALL)
    SELECT m.al, m.dy, m.lastb, m.mx, b.n nl, big.bigb,
           CASE WHEN b.n >= 0.7*m.mx THEN m.lastb ELSE big.bigb END selb,
           b.n >= 0.7*m.mx AS 末批够大
    FROM m
    JOIN bs b ON b.al=m.al AND b.dy=m.dy AND b.batch=m.lastb
    JOIN big ON big.al=m.al AND big.dy=m.dy
    """)
    sel_df = con.execute(
        "SELECT al, lastb, selb, nl, mx, 末批够大 FROM sel ORDER BY al"
    ).fetchdf()
    log("sel rules:\n" + sel_df.to_string(index=False))

    con.execute(f"""
    CREATE TABLE pp AS
    WITH a AS (
      SELECT pb.*, (pb.batch=sel.selb) is_sel, sel.selb, sel.末批够大
      FROM pb
      JOIN sel ON sel.al=pb.al AND sel.dy=pb.dy
      WHERE pb.dy='{OBS}'
    )
    SELECT flight_no, route, dep_date, dy, any_value(al) al, any_value(dtd) dtd,
      coalesce(max(price) FILTER (WHERE is_sel), arg_max(price, batch)) price,
      coalesce(max(batch) FILTER (WHERE is_sel), max(batch)) 来源批次,
      coalesce(max(obs_ts) FILTER (WHERE is_sel), arg_max(obs_ts, batch)) 取价时点,
      coalesce(max(src) FILTER (WHERE is_sel), arg_max(src, batch)) 数据源,
      (max(price) FILTER (WHERE is_sel)) IS NULL AS 回补,
      any_value(selb) 当日选用批,
      bool_and(末批够大) 末批够大,
      arg_max(dep_time, batch) 起飞时刻,
      count(*) 当天出现批数
    FROM a
    GROUP BY flight_no, route, dep_date, dy
    """)

    out_path = MID / "日价_观察日2026-11-02_dtd0_30.parquet"
    con.execute(f"""
    COPY (
      SELECT CAST(dy AS DATE) 观察日, flight_no 航班号, route 航线,
             split_part(route,'-',1) 出发机场, split_part(route,'-',2) 到达机场,
             CAST(dep_date AS DATE) 起飞日, 起飞时刻, al 航司, dtd 距起飞天数, price 价格,
             来源批次, CAST(取价时点 AS VARCHAR) 取价时点, 数据源,
             当日选用批, 当天出现批数,
             CASE WHEN 回补 THEN '是' ELSE '否' END 是否回补键,
             CASE WHEN 末批够大 THEN '末批' ELSE '末批不足70%改用最大批' END 当日选批规则
      FROM pp
      WHERE dtd BETWEEN 0 AND 30
      ORDER BY 航班号, 航线, 起飞日
    ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
    dtd14 = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}') WHERE 是否回补键='否' AND 距起飞天数>=14"
    ).fetchone()[0]
    bf = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}') WHERE 是否回补键='是'"
    ).fetchone()[0]
    dist = con.execute(
        f"SELECT 当日选用批 b, count(*) n FROM read_parquet('{out_path}') GROUP BY 1 ORDER BY 1"
    ).fetchdf()
    # dominant selected batch (mode)
    top_sel = dist.sort_values("n", ascending=False).iloc[0]["b"]
    n_not_末批够大 = int(con.execute("SELECT count(*) FROM sel WHERE NOT 末批够大").fetchone()[0])
    meta = {
        "观察日": OBS,
        "batches": [f"batch={b}" for b in BATCHES],
        "rows": int(rows),
        "dtd14_non_backfill": int(dtd14),
        "backfill": int(bf),
        "选用批分布": dist.to_dict(orient="records"),
        "主导选用批": str(top_sel),
        "末批不足70pct航司数": n_not_末批够大,
        "构建时间": time.strftime("%F %T"),
        "时区": "Asia/Shanghai",
        "耗时秒": round(time.time() - t0, 1),
    }
    (MID / "日价构建元信息.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("meta", meta)
    log("DONE", out_path, f"rows={rows}")


if __name__ == "__main__":
    main()
