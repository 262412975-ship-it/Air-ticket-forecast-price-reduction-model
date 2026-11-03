#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S0 时间线底表 + 口径对齐
输入：_原始拉取/tl2025/*.tsv.gz、_原始拉取/tl2026新/*.tsv.gz（服务器只读分块拉取），本地 ${TIMELINE_DB_URI} 拷贝，06:16 已与服务器逐日计数核对一致）
      数据/主表_全航司日价_dtd0_30.parquet（本 S0 前一步产物）
输出：扩样本对齐表（航管 2025+2026新）、timeline_2026 五家对照表、口径对齐表与各项核对 CSV
"""
import duckdb, glob, json, os, sys, time
import pandas as pd
OUT = sys.argv[1]; D = os.path.join(OUT, "数据"); K = os.path.join(OUT, "核对"); RAW = os.path.join(OUT, "_原始拉取")
TL26 = str(Path(os.environ["TL2026_LOCAL_DIR"]) / "obs=*/*.parquet")
t0 = time.time()
tmpd = os.path.join(OUT, "_tmp_duckdb2"); os.makedirs(tmpd, exist_ok=True)
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=2; SET temp_directory='{tmpd}'; SET preserve_insertion_order=false")
def csv(sql, name):
    df = c.execute(sql).df(); df.to_csv(os.path.join(K, name), index=False, encoding="utf-8-sig"); return df
cols = "{'flight_no':'VARCHAR','route':'VARCHAR','dep_date':'VARCHAR','obs_day':'VARCHAR','price':'DOUBLE'}"
步 = []
def 记(源, 名, sql, 备注=""):
    n = c.execute(sql).fetchone()[0]; 步.append({"来源": 源, "步骤": 名, "行数": n, "备注": 备注}); print(源, 名, n, flush=True); return n
for 源, sub in [("timeline_2025", "tl2025"), ("timeline_2026新", "tl2026新")]:
    fs = sorted(glob.glob(os.path.join(RAW, sub, "*.tsv.gz")))
    exp = sum(int(open(f + ".ok").read()) for f in fs)
    tb = "h25" if sub == "tl2025" else "h26"
    c.execute(f"CREATE TABLE {tb} AS SELECT *, '{源}' src FROM read_csv({fs!r}, delim='\t', header=false, columns={cols}, quote='', escape='', nullstr='')")
    n = 记(源, "1_服务器只读拉取原始行", f"SELECT count(*) FROM {tb}", f"{len(fs)} 个分块；分块 ROWS 合计 {exp}")
    assert n == exp, (源, n, exp)
c.execute(f"CREATE TABLE t26 AS SELECT flight_no, route, dep_date, obs_day, price, 'timeline_2026' src FROM read_parquet('{TL26}')")
记("timeline_2026", "1_本地拷贝原始行", "SELECT count(*) FROM t26", "tl2026_local_copy 06:16 拉取，逐日计数与服务器一致")
c.execute("CREATE TABLE allh AS SELECT * FROM h25 UNION ALL SELECT * FROM h26 UNION ALL SELECT * FROM t26")
c.execute("DROP TABLE h25; DROP TABLE h26; DROP TABLE t26")
c.execute("ALTER TABLE allh ADD COLUMN dtd INT; UPDATE allh SET dtd=date_diff('day', CAST(obs_day AS DATE), CAST(dep_date AS DATE))")
for 源 in ["timeline_2025", "timeline_2026新", "timeline_2026"]:
    w = f"src='{源}'"
    记(源, "2_其中通程(len航班号=7)", f"SELECT count(*) FROM allh WHERE {w} AND length(flight_no)=7", "剔除")
    记(源, "3_其中无效价(空或<=0)", f"SELECT count(*) FROM allh WHERE {w} AND length(flight_no)<>7 AND NOT coalesce(price>0,false)", "剔除")
    记(源, "4_其中dtd<0", f"SELECT count(*) FROM allh WHERE {w} AND length(flight_no)<>7 AND price>0 AND dtd<0", "剔除")
    记(源, "5_其中dtd>30", f"SELECT count(*) FROM allh WHERE {w} AND length(flight_no)<>7 AND price>0 AND dtd>30", "扩样本表不收；timeline_2026 对照表保留并标注")
    记(源, "6_保留(dtd0..30)", f"SELECT count(*) FROM allh WHERE {w} AND length(flight_no)<>7 AND price>0 AND dtd BETWEEN 0 AND 30")
    记(源, "7_其中价格>=5000(疑似录入错误，标注不删)", f"SELECT count(*) FROM allh WHERE {w} AND length(flight_no)<>7 AND price>=5000 AND dtd BETWEEN 0 AND 30")
c.execute("CREATE TABLE h AS SELECT * FROM allh WHERE length(flight_no)<>7 AND price>0 AND dtd>=0"); c.execute("DROP TABLE allh")
# 同键跨观察日中位价，用于异常价标注（>5 倍或 <1/5）
c.execute("CREATE TABLE kmed AS SELECT src, flight_no, route, dep_date, median(price) km FROM h GROUP BY ALL")
# ---- 扩样本对齐表（航管）
c.execute("""CREATE TABLE ext AS SELECT h.obs_day 观察日, h.flight_no 航班号, h.route 航线, split_part(h.route,'-',1) 出发机场, split_part(h.route,'-',2) 到达机场,
   h.dep_date 起飞日, substr(h.flight_no,1,2) 航司, h.dtd 距起飞天数, h.price 价格, h.src 数据源, '航管' 价格来源族,
   substr(h.obs_day,1,4) 观察年, substr(h.dep_date,6,2) 起飞月,
   '源库一天一价(同日多批取最晚抓取)' 当日选批规则, '否(源无回补概念)' 是否回补键,
   CASE WHEN h.price>=5000 THEN '价格>=5000疑似录入错误' WHEN h.price>5*k.km OR h.price<k.km/5 THEN '偏离同键中位价5倍以上' ELSE '' END 价格异常标记
   FROM h JOIN kmed k USING (src, flight_no, route, dep_date) WHERE h.src IN ('timeline_2025','timeline_2026新') AND h.dtd BETWEEN 0 AND 30""")
c.execute(f"COPY (SELECT * FROM ext ORDER BY 数据源, 观察日, 航班号, 航线, 起飞日) TO '{D}/扩样本表_航管2025与2026新_对齐_dtd0_30.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")
c.execute("""CREATE TABLE t26o AS SELECT h.obs_day 观察日, h.flight_no 航班号, h.route 航线, split_part(h.route,'-',1) 出发机场, split_part(h.route,'-',2) 到达机场,
   h.dep_date 起飞日, substr(h.flight_no,1,2) 航司, h.dtd 距起飞天数, h.price 价格, 'timeline_2026' 数据源, 'timeline_2026(五家)' 价格来源族,
   CASE WHEN h.dtd>30 THEN 'dtd>30_主口径外' ELSE '' END dtd标记,
   CASE WHEN h.price>=5000 THEN '价格>=5000疑似录入错误' WHEN h.price>5*k.km OR h.price<k.km/5 THEN '偏离同键中位价5倍以上' ELSE '' END 价格异常标记
   FROM h JOIN kmed k USING (src, flight_no, route, dep_date) WHERE h.src='timeline_2026'""")
c.execute(f"COPY (SELECT * FROM t26o ORDER BY 观察日, 航班号, 航线, 起飞日) TO '{D}/对照表_timeline_2026五家.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")
pd.DataFrame(步).to_csv(os.path.join(K, "时间线_各步行数.csv"), index=False, encoding="utf-8-sig")
# ---- 统一视图（四源 dtd0..30）
c.execute("""CREATE VIEW u AS
  SELECT '全航司' 源, 观察日 o, 航班号 f, 航线 r, 起飞日 d, 距起飞天数 dtd, 价格 p FROM read_parquet('""" + D + """/主表_全航司日价_dtd0_30.parquet')
  UNION ALL SELECT 数据源, 观察日, 航班号, 航线, 起飞日, 距起飞天数, 价格 FROM ext
  UNION ALL SELECT 'timeline_2026', 观察日, 航班号, 航线, 起飞日, 距起飞天数, 价格 FROM t26o WHERE 距起飞天数<=30""")
# 基本画像
csv("""SELECT 源, count(*) 行数, min(o) 观察日最小, max(o) 观察日最大, count(DISTINCT o) 观察天数, min(d) 起飞日最小, max(d) 起飞日最大,
   count(DISTINCT substr(f,1,2)) 航司数, count(DISTINCT r) 航线数, count(DISTINCT f) 航班号数, min(dtd) dtd最小, max(dtd) dtd最大, median(dtd) dtd中位,
   median(p) 价格中位, avg((p=round(p))::INT) 整数价占比, avg((p%10=0)::INT) 十元整数倍占比
   FROM u GROUP BY 1 ORDER BY 1""", "各源画像.csv")
csv("SELECT 源, substr(o,1,7) 观察月, count(*) 行数, count(DISTINCT o) 观察天数, count(DISTINCT substr(f,1,2)) 航司数 FROM u GROUP BY 1,2 ORDER BY 1,2", "各源按观察月.csv")
csv("SELECT 源, substr(d,1,7) 起飞月, count(*) 行数, min(dtd) dtd最小, max(dtd) dtd最大, count(DISTINCT d) 起飞天数 FROM u GROUP BY 1,2 ORDER BY 1,2", "各源按起飞月_季节覆盖.csv")
csv("SELECT 源, dtd, count(*) 行数 FROM u GROUP BY 1,2 ORDER BY 1,2", "各源按dtd行数.csv")
csv("""WITH a AS (SELECT DISTINCT 源, substr(f,1,2) al FROM u)
   SELECT al 航司, max((源='全航司')::INT) 全航司, max((源='timeline_2025')::INT) 航管2025, max((源='timeline_2026新')::INT) 航管2026新, max((源='timeline_2026')::INT) tl2026五家 FROM a GROUP BY 1 ORDER BY 1""", "各源航司覆盖.csv")
csv("""WITH a AS (SELECT DISTINCT 源, r FROM u), af AS (SELECT r FROM a WHERE 源='全航司')
   SELECT 源, count(*) 航线数, count(*) FILTER (WHERE r IN (SELECT r FROM af)) 与全航司共有航线数, (SELECT count(*) FROM af) 全航司航线数 FROM a GROUP BY 1 ORDER BY 1""", "各源航线覆盖.csv")
csv("""WITH a AS (SELECT DISTINCT 源, o FROM u) SELECT x.源 源A, y.源 源B, count(*) 重叠观察天数, min(x.o) 起, max(x.o) 止
   FROM a x JOIN a y ON x.o=y.o AND x.源<y.源 GROUP BY 1,2 ORDER BY 1,2""", "各源重叠观察日.csv")
# 动态画像：相邻观察日（间隔1天）同键价格变动
c.execute("""CREATE TABLE dyn AS SELECT 源, f, r, d, o, dtd, p, lag(p) OVER w pp, date_diff('day', CAST(lag(o) OVER w AS DATE), CAST(o AS DATE)) gap
   FROM u WINDOW w AS (PARTITION BY 源, f, r, d ORDER BY o)""")
csv("""SELECT 源, CASE WHEN dtd<7 THEN '1_dtd0..6' WHEN dtd<14 THEN '2_dtd7..13' WHEN dtd<21 THEN '3_dtd14..20' ELSE '4_dtd21..30' END dtd段,
   count(*) 相邻一天对数, avg((p<>pp)::INT) 变动占比, avg((p<=pp*0.9 OR p<=pp-80)::INT) 显著降占比, avg((p>=pp*1.1 OR p>=pp+80)::INT) 显著涨占比, median(abs(p-pp)) 绝对变动中位
   FROM dyn WHERE gap=1 GROUP BY 1,2 ORDER BY 1,2""", "各源相邻日价格动态.csv")
# ---- 重叠键价差
def pair(A, B, name):
    c.execute(f"""CREATE OR REPLACE TABLE pr AS SELECT a.f, a.r, a.d, a.o, a.dtd, a.p pa, b.p pb FROM u a JOIN u b ON a.f=b.f AND a.r=b.r AND a.d=b.d AND a.o=b.o
       WHERE a.源='{A}' AND b.源='{B}'""")
    q = lambda g: f"""SELECT '{A}' 源A, '{B}' 源B, {g} count(*) 重叠键数, avg((pa=pb)::INT) 完全相等占比, avg((abs(pa-pb)<=10)::INT) 差在10元内占比,
       median(pa-pb) 价差中位_A减B, avg(pa-pb) 价差均值_A减B, median(pa/pb) 比值中位_A除B, avg(pa/pb) 比值均值,
       quantile_cont(ln(pa/pb),0.05) 对数比P5, quantile_cont(ln(pa/pb),0.95) 对数比P95, avg((pa>pb)::INT) A高占比, avg((pa<pb)::INT) A低占比 FROM pr"""
    r0 = c.execute(q("")).df()
    r1 = c.execute(q("substr(f,1,2) 航司,") + " GROUP BY 航司 ORDER BY 航司").df()
    r2 = c.execute(q("CASE WHEN dtd<7 THEN '1_dtd0..6' WHEN dtd<14 THEN '2_dtd7..13' WHEN dtd<21 THEN '3_dtd14..20' ELSE '4_dtd21..30' END dtd段,") + " GROUP BY dtd段 ORDER BY dtd段").df()
    r3 = c.execute(q("o 观察日,") + " GROUP BY 观察日 ORDER BY 观察日").df()
    # 方向一致性：同键相邻观察日两源各自的涨跌方向
    r4 = c.execute(f"""WITH x AS (SELECT f,r,d,o,pa,pb, lag(pa) OVER w la, lag(pb) OVER w lb, date_diff('day', CAST(lag(o) OVER w AS DATE), CAST(o AS DATE)) g FROM pr WINDOW w AS (PARTITION BY f,r,d ORDER BY o))
       SELECT '{A}' 源A, '{B}' 源B, count(*) 双源相邻一天对数, avg((sign(pa-la)=sign(pb-lb))::INT) 涨跌方向一致占比,
         avg(((pa<=la*0.9 OR pa<=la-80)=(pb<=lb*0.9 OR pb<=lb-80))::INT) 显著降判定一致占比,
         avg((pa<=la*0.9 OR pa<=la-80)::INT) A显著降占比, avg((pb<=lb*0.9 OR pb<=lb-80)::INT) B显著降占比 FROM x WHERE g=1""").df()
    for df, suf in [(r1, "按航司"), (r2, "按dtd段"), (r3, "按观察日")]:
        df.to_csv(os.path.join(K, f"重叠键价差_{name}_{suf}.csv"), index=False, encoding="utf-8-sig")
    return r0, r4
res = [pair("全航司", "timeline_2026", "全航司vs时间线2026"), pair("timeline_2026新", "timeline_2026", "航管2026新vs时间线2026"),
       pair("timeline_2026新", "全航司", "航管2026新vs全航司"), pair("timeline_2025", "全航司", "航管2025vs全航司")]
pd.concat([r[0] for r in res]).to_csv(os.path.join(K, "重叠键价差_汇总.csv"), index=False, encoding="utf-8-sig")
pd.concat([r[1] for r in res]).to_csv(os.path.join(K, "重叠键涨跌方向一致性.csv"), index=False, encoding="utf-8-sig")
# 同航线同 dtd 价格水平（不同观察期，仅作水平参考，季节混杂）
csv("""WITH m AS (SELECT 源, r, dtd, median(p) mp FROM u GROUP BY ALL)
   SELECT a.源 源, count(*) 共有航线dtd格数, median(a.mp/b.mp) 中位价比_对全航司 FROM m a JOIN m b ON a.r=b.r AND a.dtd=b.dtd AND b.源='全航司' WHERE a.源<>'全航司' GROUP BY 1""", "同航线同dtd中位价比_对全航司_季节混杂仅参考.csv")
json.dump({"构建时间": time.strftime("%F %T"), "耗时秒": round(time.time() - t0)}, open(os.path.join(K, "时间线_构建元信息.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("done", round(time.time() - t0))
