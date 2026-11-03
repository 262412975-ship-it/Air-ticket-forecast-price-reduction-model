# -*- coding: utf-8 -*-
"""补充：timeline_2025 内含两段不同来源（2025-06..08 方圆写入 39 家；2025-09..12 原库 11 家），给扩样本表加「子来源」列并出分子来源画像"""
import duckdb, os, sys
OUT = sys.argv[1]; D = os.path.join(OUT, "数据"); K = os.path.join(OUT, "核对")
tmpd = os.path.join(OUT, "_tmp_duckdb3"); os.makedirs(tmpd, exist_ok=True)
c = duckdb.connect(); c.execute(f"SET memory_limit='2GB'; SET threads=2; SET temp_directory='{tmpd}'; SET preserve_insertion_order=false")
src = f"{D}/扩样本表_航管2025与2026新_对齐_dtd0_30.parquet"; dst = f"{D}/扩样本表_航管2025与2026新_对齐_dtd0_30_含子来源.parquet"
if not os.path.exists(dst):
    c.execute(f"""COPY (SELECT *, CASE WHEN 数据源='timeline_2026新' THEN '航管2026_03..09(39家)'
        WHEN 观察日<'2025-09-01' THEN '方圆2025_06..08(39家,10-30写入)' ELSE '原2025库_09..12(11家)' END 子来源
        FROM read_parquet('{src}') ORDER BY 数据源, 观察日, 航班号, 航线, 起飞日) TO '{dst}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    os.rename(dst + ".tmp", dst)
c.execute(f"CREATE VIEW e AS SELECT * FROM read_parquet('{dst}')")
def csv(sql, name): c.execute(sql).df().to_csv(os.path.join(K, name), index=False, encoding="utf-8-sig")
csv("""SELECT 子来源, count(*) 行数, min(观察日) 观察日最小, max(观察日) 观察日最大, min(起飞日) 起飞日最小, max(起飞日) 起飞日最大, count(DISTINCT 航司) 航司数,
  count(DISTINCT 航线) 航线数, max(距起飞天数) dtd最大, median(价格) 价格中位, avg((价格%10=0)::INT) 十元整数倍占比, sum((价格异常标记<>'')::INT) 异常价标记行
  FROM e GROUP BY 1 ORDER BY 1""", "扩样本_分子来源画像.csv")
csv("""WITH d AS (SELECT 子来源, 距起飞天数 dtd, 价格 p, lag(价格) OVER w pp, date_diff('day', CAST(lag(观察日) OVER w AS DATE), CAST(观察日 AS DATE)) g
   FROM e WINDOW w AS (PARTITION BY 数据源, 航班号, 航线, 起飞日 ORDER BY 观察日))
  SELECT 子来源, CASE WHEN dtd<7 THEN '1_dtd0..6' WHEN dtd<14 THEN '2_dtd7..13' WHEN dtd<21 THEN '3_dtd14..20' ELSE '4_dtd21..30' END dtd段, count(*) 相邻一天对数,
   avg((p<>pp)::INT) 变动占比, avg((p<=pp*0.9 OR p<=pp-80)::INT) 显著降占比, avg((p>=pp*1.1 OR p>=pp+80)::INT) 显著涨占比
  FROM d WHERE g=1 GROUP BY 1,2 ORDER BY 1,2""", "扩样本_分子来源相邻日动态.csv")
csv("SELECT 子来源, 航司, count(*) 行数 FROM e GROUP BY 1,2 ORDER BY 1,2", "扩样本_分子来源按航司.csv")
print("done")
