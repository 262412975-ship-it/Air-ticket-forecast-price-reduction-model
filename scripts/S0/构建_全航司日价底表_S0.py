#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S0 全航司日价底表（只读本地 AF 拷贝；本地与远端对照库（占位，非真实路径） 已逐批核对行数，见 核对/本地与远端_逐批行数核对.csv）
口径沿用 重训准备_20261102/v3f 构建脚本（末批只数 dtd<=30 键、70% 规则、回补定义），改动：
  - 观察日 2026-10-15..2026-11-02（11-02 全日四批已齐：0454/1110/1503/2116，正式增量进表）
  - 主表 dtd 0..30 且非回补；31..50 另存离线对照表；回补键另存并计数
  - 加采集窗标记（切换日 2026-10-29）、1117 空日标记；不带 seats 列
"""
import duckdb, glob, json, os, re, time, hashlib, sys
R = "${ROUND4_ROOT}/allflights/data"
OUT = sys.argv[1]
D = os.path.join(OUT, "数据"); K = os.path.join(OUT, "核对"); os.makedirs(D, exist_ok=True); os.makedirs(K, exist_ok=True)
SNAP批次序 = ["202610151328", "202610152000", "202610160556", "202610161404", "202610162000",
              "202610170556", "202610171401", "202610172000", "202610180556"]
OBS_FROM, OBS_TO, SWITCH = "2026-10-15", "2026-11-02", "2026-10-29"
事故批次 = {"202610262059": ["2026-10-27", "2026-10-28", "2026-10-29", "2026-10-31", "2026-11-01"]}
t0 = time.time()
tmpd = os.path.join(OUT, "_tmp_duckdb"); os.makedirs(tmpd, exist_ok=True)
con = duckdb.connect()
con.execute(f"SET memory_limit='2GB'; SET threads=2; SET temp_directory='{tmpd}'; SET preserve_insertion_order=false")
snap = os.path.join(R, "af_snap.parquet")
incs = sorted(glob.glob(os.path.join(R, "incremental", "batch=*", "batch.parquet")))
for f in incs: assert os.path.exists(os.path.join(os.path.dirname(f), "_SUCCESS")), f
con.execute("CREATE TABLE snapmap(b BIGINT, bid VARCHAR)")
con.executemany("INSERT INTO snapmap VALUES (?,?)", [(i + 1, b) for i, b in enumerate(SNAP批次序)])
con.execute(f"""CREATE TABLE raw AS
  SELECT s.flight_no, s.route, s.date dep_date, m.bid batch, s.ts obs_ts, CAST(s.price AS DOUBLE) price, s.time dep_time, 'af_snap' src
  FROM read_parquet('{snap}') s JOIN snapmap m ON m.b=s.batch
  UNION ALL
  SELECT flight_no, route, CAST(date AS VARCHAR), CAST(batch AS VARCHAR), TRY_CAST(updated AS TIMESTAMP), TRY_CAST(price AS DOUBLE), time, 'incremental'
  FROM read_parquet({incs!r}, union_by_name=true, hive_partitioning=false)""")
步 = []
def 记(名, sql, 备注=""):
    n = con.execute(sql).fetchone()[0]; 步.append({"步骤": 名, "行数": n, "备注": 备注}); print(名, n, flush=True); return n
记("1_原始行（af_snap+增量，键×批次）", "SELECT count(*) FROM raw", f"af_snap 9 批 + 增量 {len(incs)} 批")
记("2_其中通程(len航班号=7)", "SELECT count(*) FROM raw WHERE length(flight_no)=7", "剔除")
记("3_其中无效价(空或<=0)", "SELECT count(*) FROM raw WHERE length(flight_no)<>7 AND NOT coalesce(price>0,false)", "剔除")
con.execute("""CREATE TABLE 批统计 AS SELECT batch 批次号, src 来源, count(*) 原始行数, count(*) FILTER (WHERE length(flight_no)=7) 通程行数,
  count(*) FILTER (WHERE length(flight_no)<>7 AND NOT coalesce(price>0,false)) 无效价行数, min(obs_ts) 最早取价, max(obs_ts) 最晚取价,
  min(dep_date) 起飞日最小, max(dep_date) 起飞日最大 FROM raw GROUP BY 1,2 ORDER BY 1""")
con.execute("CREATE TABLE r AS SELECT * FROM raw WHERE length(flight_no)<>7 AND price>0"); con.execute("DROP TABLE raw")
记("4_剔通程与无效价后", "SELECT count(*) FROM r")
con.execute("""CREATE TABLE pb AS SELECT flight_no, route, dep_date, batch, any_value(src) src, min(price) price, count(*) 批内重复,
  arg_min(obs_ts, price) obs_ts, arg_min(dep_time, price) dep_time FROM r GROUP BY flight_no, route, dep_date, batch""")
con.execute("DROP TABLE r")
记("5_批内同键去重后(同批同键取最低价)", "SELECT count(*) FROM pb")
con.execute("""ALTER TABLE pb ADD COLUMN dy VARCHAR; UPDATE pb SET dy=substr(batch,1,4)||'-'||substr(batch,5,2)||'-'||substr(batch,7,2);
  ALTER TABLE pb ADD COLUMN al VARCHAR; UPDATE pb SET al=substr(flight_no,1,2);
  ALTER TABLE pb ADD COLUMN dtd INT; UPDATE pb SET dtd=date_diff('day', CAST(dy AS DATE), CAST(dep_date AS DATE))""")
# 选批：每航司每天末批；末批(只数 dtd<=30 键)不足当天最大批 70% 改用最大批（v3e/v3f 既有口径）
con.execute("CREATE TABLE bs AS SELECT al, dy, batch, count(*) n FROM pb WHERE dtd BETWEEN 0 AND 30 GROUP BY ALL")
con.execute("""CREATE TABLE sel AS
  WITH m AS (SELECT al, dy, max(n) mx, max(batch) lastb FROM bs GROUP BY al, dy),
       big AS (SELECT al, dy, arg_max(batch, n) bigb FROM bs GROUP BY ALL)
  SELECT m.al, m.dy, m.lastb, m.mx, b.n nl, big.bigb, CASE WHEN b.n >= 0.7*m.mx THEN m.lastb ELSE big.bigb END selb, b.n >= 0.7*m.mx 末批够大
  FROM m JOIN bs b ON b.al=m.al AND b.dy=m.dy AND b.batch=m.lastb JOIN big ON big.al=m.al AND big.dy=m.dy""")
con.execute("""CREATE TABLE pp AS
  WITH a AS (SELECT pb.*, (pb.batch=sel.selb) is_sel, sel.selb, sel.末批够大 FROM pb JOIN sel ON sel.al=pb.al AND sel.dy=pb.dy)
  SELECT flight_no, route, dep_date, dy, any_value(al) al, any_value(dtd) dtd,
    coalesce(max(price) FILTER (WHERE is_sel), arg_max(price, batch)) price,
    coalesce(max(batch) FILTER (WHERE is_sel), max(batch)) 来源批次,
    coalesce(max(obs_ts) FILTER (WHERE is_sel), arg_max(obs_ts, batch)) 取价时点,
    coalesce(max(src) FILTER (WHERE is_sel), arg_max(src, batch)) 数据源,
    (max(price) FILTER (WHERE is_sel)) IS NULL AS 回补,
    any_value(selb) 当日选用批, bool_and(末批够大) 末批够大, arg_max(dep_time, batch) 起飞时刻, count(*) 当天出现批数
  FROM a GROUP BY flight_no, route, dep_date, dy""")
记("6_每键每观察日一价(全部观察日、全部dtd)", "SELECT count(*) FROM pp")
con.execute("CREATE TABLE days AS SELECT dy, count(DISTINCT batch) 批次数, max(batch) 末批, string_agg(DISTINCT batch, ' ' ORDER BY batch) 批次清单 FROM pb GROUP BY dy ORDER BY dy")
截止 = con.execute("SELECT max(dy) FROM days WHERE CAST(substr(末批,9,2) AS INT)>=19").fetchone()[0]
记("7_观察日<10-15(剔)", f"SELECT count(*) FROM pp WHERE dy<'{OBS_FROM}'")
记("8_观察日>OBS_TO(剔)", f"SELECT count(*) FROM pp WHERE dy>'{OBS_TO}'", "OBS_TO=11-02；11-03 及以后剔")
事故 = ",".join(f"('{b}','{d}')" for b, ds in 事故批次.items() for d in ds)
con.execute(f"CREATE TABLE acc(batch VARCHAR, dep_date VARCHAR); INSERT INTO acc VALUES {事故}")
con.execute(f"""CREATE TABLE out AS
  SELECT p.dy 观察日, p.flight_no 航班号, p.route 航线, split_part(p.route,'-',1) 出发机场, split_part(p.route,'-',2) 到达机场,
    p.dep_date 起飞日, p.起飞时刻, p.al 航司, p.dtd 距起飞天数, p.price 价格, p.来源批次, CAST(p.取价时点 AS VARCHAR) 取价时点, p.数据源,
    p.当日选用批, p.当天出现批数,
    CASE WHEN p.回补 THEN '是' ELSE '否' END 是否回补键,
    CASE WHEN p.末批够大 THEN '末批' ELSE '末批不足70%改用最大批' END 当日选批规则,
    CASE WHEN p.dy<'{SWITCH}' THEN '30天窗' WHEN p.dy='{SWITCH}' THEN '切换日(30/50混)' ELSE '50天窗' END 采集窗,
    CASE WHEN p.dy='2026-10-21' THEN '混批日' WHEN NOT p.末批够大 THEN '末批不足70%' ELSE '' END 混批标记,
    CASE WHEN a.batch IS NOT NULL AND p.回补 THEN '事故批起飞日_本键不在该批_取当天前批价'
         WHEN a.batch IS NOT NULL THEN '事故批起飞日_该批已补拉齐' ELSE '' END 批次事故标记,
    CASE WHEN p.dy='2026-10-30' AND p.dep_date BETWEEN '2026-11-30' AND '2026-12-19'
         THEN '1117批该起飞日为放行空日(不可用)；本行价取自'||p.来源批次 ELSE '' END 批1117空日标记,
    CASE WHEN CAST(p.dy AS DATE)+INTERVAL 7 DAY <= CAST('{截止}' AS DATE) THEN '是' ELSE '否' END 是否满7天窗口
  FROM pp p LEFT JOIN acc a ON a.batch=p.当日选用批 AND a.dep_date=p.dep_date
  WHERE p.dy BETWEEN '{OBS_FROM}' AND '{OBS_TO}'""")
记("9_观察日10-15..OBS_TO", "SELECT count(*) FROM out")
记("10_其中回补键(主表剔，另存)", "SELECT count(*) FROM out WHERE 是否回补键='是'")
记("11_其中dtd<0或>50(异常)", "SELECT count(*) FROM out WHERE 距起飞天数<0 OR 距起飞天数>50")
n主 = 记("12_主表：非回补且dtd0..30", "SELECT count(*) FROM out WHERE 是否回补键='否' AND 距起飞天数 BETWEEN 0 AND 30")
n对 = 记("13_对照表：非回补且dtd31..50", "SELECT count(*) FROM out WHERE 是否回补键='否' AND 距起飞天数 BETWEEN 31 AND 50")
记("14_主表中来源批次=1117的行", "SELECT count(*) FROM out WHERE 是否回补键='否' AND 来源批次='202610301117'")
记("15_10-30观察日起飞11-30..12-19的行(1117空日区间)", "SELECT count(*) FROM out WHERE 批1117空日标记<>''", "价格均来自非1117批，见核对")
dup = con.execute("SELECT count(*) FROM (SELECT 观察日,航班号,航线,起飞日 FROM out GROUP BY ALL HAVING count(*)>1)").fetchone()[0]
assert dup == 0
cols_ban = [c for c in con.execute("DESCRIBE out").df()["column_name"] if "seat" in c.lower() or "余座" in c]
assert not cols_ban, cols_ban
def cp(sql, name): con.execute(f"COPY ({sql}) TO '{D}/{name}' (FORMAT PARQUET, COMPRESSION ZSTD)")
cp("SELECT * FROM out WHERE 是否回补键='否' AND 距起飞天数 BETWEEN 0 AND 30 ORDER BY 观察日, 航班号, 航线, 起飞日", "主表_全航司日价_dtd0_30.parquet")
cp("SELECT * FROM out WHERE 是否回补键='否' AND 距起飞天数 BETWEEN 31 AND 50 ORDER BY 观察日, 航班号, 航线, 起飞日", "对照表_全航司日价_dtd31_50.parquet")
cp("SELECT *, CASE WHEN 距起飞天数<=30 THEN 'dtd0..30' ELSE 'dtd31..50' END dtd分段 FROM out WHERE 是否回补键='是' ORDER BY 观察日, 航班号, 航线, 起飞日", "回补表_全航司日价_另存.parquet")
def csv(sql, name): con.execute(sql).df().to_csv(os.path.join(K, name), index=False, encoding="utf-8-sig")
import pandas as pd
pd.DataFrame(步).to_csv(os.path.join(K, "全航司_各步行数.csv"), index=False, encoding="utf-8-sig")
csv("SELECT * FROM 批统计", "全航司_逐批原始统计.csv")
csv("SELECT d.dy 观察日, d.批次数 当天批数, d.末批, d.批次清单 FROM days d ORDER BY 1", "全航司_每观察日批次.csv")
csv("SELECT dy 观察日, count(*) 航司数, sum((NOT 末批够大)::INT) 末批不足70pct航司数, string_agg(CASE WHEN NOT 末批够大 THEN al||':'||selb END, ' ') 改用最大批明细 FROM sel GROUP BY 1 ORDER BY 1", "全航司_选批规则统计.csv")
csv("""SELECT 观察日, 采集窗, count(*) 总行, sum((是否回补键='否' AND 距起飞天数<=30)::INT) 主表行, sum((是否回补键='否' AND 距起飞天数>30)::INT) 对照表行,
   sum((是否回补键='是')::INT) 回补行, count(DISTINCT 航司) 航司数, min(距起飞天数) dtd最小, max(距起飞天数) dtd最大 FROM out GROUP BY 1,2 ORDER BY 1""", "全航司_每观察日行数.csv")
csv("SELECT 航司, count(*) 主表行, count(DISTINCT 航线) 航线数, count(DISTINCT 航班号) 航班数 FROM out WHERE 是否回补键='否' AND 距起飞天数<=30 GROUP BY 1 ORDER BY 2 DESC", "全航司_主表按航司.csv")
csv("SELECT 来源批次, 起飞日, count(*) 行数 FROM out WHERE 批1117空日标记<>'' GROUP BY 1,2 ORDER BY 2,1", "全航司_1117空日区间_实际取价批.csv")
csv("SELECT 距起飞天数, count(*) FILTER (WHERE 是否回补键='否') 非回补行, count(*) FILTER (WHERE 是否回补键='是') 回补行 FROM out GROUP BY 1 ORDER BY 1", "全航司_按dtd行数.csv")
meta = {"构建时间": time.strftime("%F %T"), "时区": "Asia/Shanghai", "观察日": [OBS_FROM, OBS_TO], "采集窗切换日": SWITCH, "数据截止日(有晚批)": 截止,
        "主表行": n主, "对照表行": n对, "增量批数": len(incs), "耗时秒": round(time.time() - t0)}
json.dump(meta, open(os.path.join(K, "全航司_构建元信息.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(meta)
