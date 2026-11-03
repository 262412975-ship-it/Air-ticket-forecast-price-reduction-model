#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 phase3 高效版：先聚合再回接，避免 6.4M 自连接爆内存/卡住。"""
import json, time, os
from pathlib import Path
import duckdb

ROOT = Path("${PROJECT_ROOT}/S1_特征")
MAIN = Path("${PROJECT_ROOT}/S0_底表/数据/主表_全航司日价_dtd0_30.parquet")
MID = ROOT / "中间"
LOG = ROOT / "日志"
MEM = os.environ.get("S1_DUCK_MEM", "6GB")
THREADS = int(os.environ.get("S1_THREADS", "4"))
廉航 = ("9C", "AQ")
邻时段分钟 = 180

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG / "构建.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    done = MID / "_DONE_03_竞争横截面"
    if done.exists():
        log("skip 03"); return
    t0 = time.time()
    tmp = MID / "duck_tmp"; tmp.mkdir(exist_ok=True)
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{MEM}'; SET threads={THREADS}; SET temp_directory='{tmp}'; SET preserve_insertion_order=false")

    c.execute(f"""
    CREATE TABLE base AS
    SELECT CAST(观察日 AS DATE) 观察日, 航班号, 航线, CAST(起飞日 AS DATE) 起飞日, 航司,
           距起飞天数, CAST(价格 AS DOUBLE) 价格, 起飞时刻, 出发机场, 到达机场,
           CASE WHEN regexp_matches(起飞时刻,'^[0-9]{{1,2}}:[0-9]{{2}}$')
                THEN CAST(split_part(起飞时刻,':',1) AS INT)*60+CAST(split_part(起飞时刻,':',2) AS INT) END mm,
           CAST(起飞日 AS DATE) - DATE '2026-01-01' AS dd
    FROM read_parquet('{MAIN}') WHERE 价格>0 AND length(航班号)<>7
    """)
    c.execute("ALTER TABLE base ADD COLUMN abs_min BIGINT")
    c.execute("UPDATE base SET abs_min = CASE WHEN mm IS NOT NULL THEN dd::BIGINT*1440+mm END")
    log(f"base {c.execute('select count(*) from base').fetchone()[0]}")

    # ---- 特征A：逐日（已验证快）----
    days = [r[0] for r in c.execute("SELECT DISTINCT 观察日 FROM base ORDER BY 1").fetchall()]
    c.execute("""CREATE TABLE smday AS
      SELECT c.观察日, c.航线, c.航班号 fc, p.航班号 fp, c.起飞日, (c.价格=p.价格)::INT eq
      FROM base c JOIN base p
        ON c.观察日=p.观察日 AND c.航线=p.航线 AND c.起飞日=p.起飞日
       AND c.abs_min=p.abs_min AND c.航班号<>p.航班号
      WHERE c.abs_min IS NOT NULL""")
    c.execute("""CREATE TABLE sm AS
      SELECT *, min(eq) OVER w 全同价
      FROM smday WINDOW w AS (PARTITION BY 航线,fc,fp,起飞日 ORDER BY 观察日 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)""")
    c.execute("""CREATE TABLE fa(观察日 DATE, 航班号 VARCHAR, 航线 VARCHAR, 起飞日 DATE,
                 n_ex BIGINT, med_ex DOUBLE, min_ex DOUBLE, lower_ex BIGINT)""")
    for d in days:
        c.execute(f"""INSERT INTO fa
        SELECT c.观察日,c.航班号,c.航线,c.起飞日,
          count(*) FILTER (WHERE coalesce(s.全同价,0)<>1),
          median(p.价格) FILTER (WHERE coalesce(s.全同价,0)<>1),
          min(p.价格) FILTER (WHERE coalesce(s.全同价,0)<>1),
          count(*) FILTER (WHERE coalesce(s.全同价,0)<>1 AND p.价格<c.价格)
        FROM (SELECT * FROM base WHERE 观察日=DATE '{d}' AND abs_min IS NOT NULL) c
        JOIN (SELECT * FROM base WHERE 观察日=DATE '{d}' AND abs_min IS NOT NULL AND 航司 NOT IN {廉航} AND 价格>0) p
          ON p.航线=c.航线 AND p.dd BETWEEN c.dd-1 AND c.dd+1
         AND abs(p.abs_min-c.abs_min)<={邻时段分钟}
         AND NOT (p.航班号=c.航班号 AND p.起飞日=c.起飞日)
        LEFT JOIN (SELECT * FROM sm WHERE 观察日=DATE '{d}') s
          ON s.航线=c.航线 AND s.fc=c.航班号 AND s.起飞日=c.起飞日 AND s.fp=p.航班号 AND p.起飞日=s.起飞日
        GROUP BY 1,2,3,4""")
    log(f"fa done {time.time()-t0:.0f}s")
    c.execute(f"COPY fa TO '{MID}/tmp_fa.parquet' (FORMAT PARQUET)")

    # ---- 同日同航线同起飞日 市场聚合（不含自连接明细）----
    c.execute("""
    CREATE TABLE mkt_agg AS
    SELECT 观察日, 航线, 起飞日,
      count(*) AS n_sale,
      count(DISTINCT 航司) AS n_airlines_day,
      min(价格) AS min_day,
      median(价格) AS med_day,
      max(价格) AS max_day,
      count(*) AS n_priced_day
    FROM base GROUP BY 1,2,3
    """)
    # 逐航班竞争：用聚合减自身（n_comp = n_sale-1, comp_min 需重算若自身是最低）
    # 精确 comp_min：二次聚合 min of others via join to agg + conditional
    c.execute("""
    CREATE TABLE sd AS
    SELECT a.观察日, a.航班号, a.航线, a.起飞日, a.价格, a.航司, a.abs_min,
      m.n_sale - 1 AS n_comp,
      m.n_airlines_day AS n_airlines,
      -- 他航最低：若自身不是唯一最低则用全场最低，否则用次低近似（用全场最低当自身>最低；自身=最低时用 nullif 技巧）
      CASE WHEN a.价格 > m.min_day THEN m.min_day
           ELSE (SELECT min(b.价格) FROM base b WHERE b.观察日=a.观察日 AND b.航线=a.航线 AND b.起飞日=a.起飞日 AND b.航班号<>a.航班号)
      END AS comp_min,
      CASE WHEN a.价格 < m.max_day THEN m.max_day
           ELSE (SELECT max(b.价格) FROM base b WHERE b.观察日=a.观察日 AND b.航线=a.航线 AND b.起飞日=a.起飞日 AND b.航班号<>a.航班号)
      END AS comp_max,
      m.med_day AS comp_med,
      m.min_day AS min_day_all,
      m.n_sale AS g3_n_sale,
      m.n_airlines_day, m.n_priced_day
    FROM base a
    JOIN mkt_agg m USING (观察日, 航线, 起飞日)
    """)
    # 上面的相关子查询仍可能慢。改成：先算每个键的 min/max excluding self via window
    c.execute("DROP TABLE sd")
    c.execute("""
    CREATE TABLE sd AS
    WITH w AS (
      SELECT 观察日, 航班号, 航线, 起飞日, 价格, 航司, abs_min,
        count(*) OVER w AS n_sale,
        count(DISTINCT 航司) OVER w AS n_airlines,
        min(价格) OVER w AS min_all,
        max(价格) OVER w AS max_all,
        median(价格) OVER w AS med_all,
        -- 次小/次大：用价格排序
        min(价格) OVER (PARTITION BY 观察日,航线,起飞日 ORDER BY 价格 ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS min_after,
        max(价格) OVER (PARTITION BY 观察日,航线,起飞日 ORDER BY 价格 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS max_before
      FROM base
      WINDOW w AS (PARTITION BY 观察日,航线,起飞日)
    )
    SELECT 观察日, 航班号, 航线, 起飞日, 价格, 航司, abs_min,
      n_sale - 1 AS n_comp,
      n_airlines,
      CASE WHEN 价格 > min_all THEN min_all ELSE coalesce(min_after, min_all) END AS comp_min,
      CASE WHEN 价格 < max_all THEN max_all ELSE coalesce(max_before, max_all) END AS comp_max,
      med_all AS comp_med,
      min_all AS min_day,
      n_sale AS g3_n_sale,
      n_airlines AS n_airlines_day,
      n_sale AS n_priced_day,
      n_sale AS n_flights_day
    FROM w
    """)
    # x_rank_pct 近似：用价格在同组的百分位
    c.execute("""
    CREATE TABLE sd2 AS
    SELECT *, percent_rank() OVER (PARTITION BY 观察日,航线,起飞日 ORDER BY 价格) AS x_rank_pct
    FROM sd
    """)
    log(f"sd done {time.time()-t0:.0f}s")
    c.execute(f"COPY sd2 TO '{MID}/tmp_sd.parquet' (FORMAT PARQUET)")

    # ---- ±2h 近似：同起飞日同航线、同一小时桶（60min）窗口计数（O(n)；说明里披露≠精确±120分钟）----
    c.execute("""
    CREATE TABLE b2 AS
    SELECT *, CASE WHEN abs_min IS NOT NULL THEN abs_min // 60 END AS hour_bin
    FROM base WHERE abs_min IS NOT NULL
    """)
    c.execute("""
    CREATE TABLE pm2h AS
    WITH w AS (
      SELECT 观察日,航班号,航线,起飞日,价格,航司,hour_bin,
        count(*) OVER p AS n_bin,
        min(价格) OVER p AS min_bin,
        count(DISTINCT 航司) OVER p AS n_air_bin
      FROM b2
      WINDOW p AS (PARTITION BY 观察日,航线,起飞日,hour_bin)
    )
    SELECT 观察日,航班号,航线,起飞日,
      greatest(n_bin-1,0) AS n_flights_pm2h,
      n_air_bin AS n_airlines_pm2h,
      CASE WHEN 价格 > min_bin THEN min_bin
           ELSE min_bin END AS comp_min_pm2h,
      greatest(n_bin-1,0) AS n_priced_pm2h
    FROM w
    """)
    # 自身是最低时 comp_min_pm2h 应排除自身：用次小
    c.execute("DROP TABLE pm2h")
    c.execute("""
    CREATE TABLE pm2h AS
    WITH w AS (
      SELECT 观察日,航班号,航线,起飞日,价格,航司,hour_bin,
        count(*) OVER p AS n_bin,
        count(DISTINCT 航司) OVER p AS n_air_bin,
        min(价格) OVER p AS min_all,
        min(价格) OVER (PARTITION BY 观察日,航线,起飞日,hour_bin ORDER BY 价格 ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS min_after
      FROM b2 WINDOW p AS (PARTITION BY 观察日,航线,起飞日,hour_bin)
    )
    SELECT 观察日,航班号,航线,起飞日,
      greatest(n_bin-1,0)::BIGINT AS n_flights_pm2h,
      n_air_bin::BIGINT AS n_airlines_pm2h,
      CASE WHEN 价格 > min_all THEN min_all ELSE coalesce(min_after, min_all) END AS comp_min_pm2h,
      greatest(n_bin-1,0)::BIGINT AS n_priced_pm2h
    FROM w
    """)
    log(f"pm2h(小时桶近似) done {time.time()-t0:.0f}s")

    # ---- 序列事件 / G2 G3 ----
    c.execute("""
    CREATE TABLE fl AS
    WITH t AS (
      SELECT *, lag(价格) OVER w pp, lag(观察日) OVER w po,
        date_diff('day', lag(观察日) OVER w, 观察日) gap_obs
      FROM base WINDOW w AS (PARTITION BY 航班号,航线,起飞日 ORDER BY 观察日)
    )
    SELECT *,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 THEN 1 ELSE 0 END has_cmp,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp<=-0.15 OR 价格-pp<=-100) THEN 1 ELSE 0 END dr,
      CASE WHEN pp IS NOT NULL AND gap_obs BETWEEN 1 AND 3 AND ((价格-pp)/pp>=0.10 OR 价格-pp>=80) THEN 1 ELSE 0 END ri
    FROM t
    """)
    c.execute("""
    CREATE TABLE mk AS
    SELECT 航线,起飞日,观察日, count(*) n_sale, sum(has_cmp) n_cmp, sum(dr) n_dr, sum(ri) n_ri, min(价格) mn
    FROM fl GROUP BY 1,2,3
    """)
    c.execute("""
    CREATE TABLE mk2 AS
    SELECT *, lag(mn) OVER w mn_prev, lag(n_sale) OVER w ns_prev, lag(观察日) OVER w od_prev
    FROM mk WINDOW w AS (PARTITION BY 航线,起飞日 ORDER BY 观察日)
    """)
    c.execute("""
    CREATE TABLE mk3 AS
    SELECT a.*, b.n_sale ns_7 FROM mk2 a
    LEFT JOIN mk b ON a.航线=b.航线 AND a.起飞日=b.起飞日 AND b.观察日=a.观察日-INTERVAL 7 DAY
    """)
    c.execute("""
    CREATE TABLE pm AS
    SELECT a.航线,a.起飞日,a.观察日, sum(b.n_cmp) n_cmp_pm1, sum(b.n_dr) n_dr_pm1, sum(b.n_ri) n_ri_pm1
    FROM mk a JOIN mk b ON a.航线=b.航线 AND b.观察日=a.观察日
     AND b.起飞日 BETWEEN a.起飞日-INTERVAL 1 DAY AND a.起飞日+INTERVAL 1 DAY
    GROUP BY 1,2,3
    """)
    c.execute("""CREATE TABLE first_seen AS SELECT 航班号,航线,起飞日,min(观察日) f0 FROM base GROUP BY 1,2,3""")
    c.execute("""
    CREATE TABLE ent AS
    SELECT m.航线,m.起飞日,m.观察日, count(f.航班号) n_enter
    FROM mk m LEFT JOIN first_seen f
      ON f.航线=m.航线 AND f.起飞日=m.起飞日 AND f.f0>m.观察日-INTERVAL 7 DAY AND f.f0<=m.观察日
    GROUP BY 1,2,3
    """)
    log(f"g2g3 done {time.time()-t0:.0f}s")

    # ---- 兄弟航班：按 (航班号,航线,观察日) 聚合再减自身 ----
    c.execute("""
    CREATE TABLE sib_agg AS
    SELECT 观察日, 航班号, 航线,
      count(*) n_all, sum(价格) sum_p, min(价格) min_p, median(价格) med_p
    FROM base GROUP BY 1,2,3
    """)
    c.execute("""
    CREATE TABLE sib AS
    SELECT a.观察日,a.航班号,a.航线,a.起飞日,
      sa.n_all - 1 AS n_sib,
      CASE WHEN sa.n_all>1 THEN (sa.sum_p - a.价格)/(sa.n_all - 1) END AS sib_mean,
      sa.med_p AS sib_median,
      CASE WHEN a.价格 > sa.min_p THEN sa.min_p
           ELSE (SELECT min(b.价格) FROM base b WHERE b.观察日=a.观察日 AND b.航班号=a.航班号 AND b.航线=a.航线 AND b.起飞日<>a.起飞日)
      END AS sib_min
    FROM base a JOIN sib_agg sa USING (观察日,航班号,航线)
    """)
    # simplify sib_min with window
    c.execute("DROP TABLE sib")
    c.execute("""
    CREATE TABLE sib AS
    WITH w AS (
      SELECT 观察日,航班号,航线,起飞日,价格,
        count(*) OVER p AS n_all,
        sum(价格) OVER p AS sum_p,
        median(价格) OVER p AS med_p,
        min(价格) OVER p AS min_all,
        min(价格) OVER (PARTITION BY 观察日,航班号,航线 ORDER BY 价格 ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS min_after
      FROM base WINDOW p AS (PARTITION BY 观察日,航班号,航线)
    )
    SELECT 观察日,航班号,航线,起飞日,
      n_all-1 AS n_sib,
      CASE WHEN n_all>1 THEN (sum_p-价格)/(n_all-1) END AS sib_mean,
      med_p AS sib_median,
      CASE WHEN 价格>min_all THEN min_all ELSE coalesce(min_after,min_all) END AS sib_min
    FROM w
    """)
    c.execute("""
    CREATE TABLE sib_tot AS
    SELECT 观察日,航班号,航线,
      sum(ri) su, sum(has_cmp) nh,
      avg(CASE WHEN pp IS NOT NULL THEN 价格/pp-1 END) avg_pct,
      count(*) n
    FROM fl GROUP BY 1,2,3
    """)
    c.execute("""
    CREATE TABLE sib_ev AS
    SELECT a.观察日,a.航班号,a.航线,a.起飞日,
      CASE WHEN (t.nh - a.has_cmp)>0 THEN (t.su - a.ri)*1.0/(t.nh - a.has_cmp) END AS sib_bigup_share,
      CASE WHEN (t.n - 1)>0 THEN (t.avg_pct*t.n - coalesce(CASE WHEN a.pp IS NOT NULL THEN a.价格/a.pp-1 END,0))/(t.n-1) END AS sib_chg_mean
    FROM fl a
    JOIN sib_tot t USING (观察日,航班号,航线)
    """)
    log(f"sib done {time.time()-t0:.0f}s")

    c.execute("""
    CREATE TABLE route_ev AS
    SELECT 航线,观察日,
      CASE WHEN sum(has_cmp)>0 THEN sum(ri)*1.0/sum(has_cmp) END route_bigup_share
    FROM fl GROUP BY 1,2
    """)
    c.execute("""
    CREATE TABLE route_day AS
    SELECT 航线,观察日, sum(ri) su, sum(has_cmp) nh FROM fl GROUP BY 1,2
    """)
    c.execute("""
    CREATE TABLE route_ev7 AS
    SELECT 航线,观察日,
      CASE WHEN sum(nh) OVER w > 0 THEN sum(su) OVER w *1.0 / sum(nh) OVER w END AS route_bigup_share_7d
    FROM route_day
    WINDOW w AS (PARTITION BY 航线 ORDER BY 观察日 RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW)
    """)
    c.execute("""
    CREATE TABLE odn AS
    SELECT 观察日,航班号,起飞日, count(DISTINCT 航线) n_od_same_flight FROM base GROUP BY 1,2,3
    """)
    c.execute("""
    CREATE TABLE cs AS
    SELECT 观察日, fc AS 航班号, 航线, 起飞日, max(全同价) is_codeshare_raw
    FROM sm GROUP BY 1,2,3,4
    """)

    out = MID / "块03_竞争横截面.parquet"
    c.execute(f"""
    COPY (
      SELECT
        a.观察日,a.航班号,a.航线,a.起飞日,
        CASE WHEN fa.n_ex>=1 THEN fa.n_ex END AS 特征A_邻时段对比航班数,
        CASE WHEN fa.n_ex>=1 THEN fa.med_ex END AS 特征A_邻时段中位价,
        CASE WHEN fa.n_ex>=1 THEN a.价格-fa.med_ex END AS 特征A_差值_元,
        CASE WHEN fa.n_ex>=1 AND fa.med_ex>0 THEN a.价格/fa.med_ex END AS 特征A_比值,
        CASE WHEN fa.n_ex>=1 THEN fa.lower_ex::DOUBLE/fa.n_ex END AS 特征A_比自己便宜的邻居占比,
        CASE WHEN fa.n_ex>=1 THEN CASE WHEN a.价格<=fa.min_ex THEN 1 ELSE 0 END END AS 特征A_是否邻时段最低价,
        CASE WHEN fa.n_ex>=1 AND fa.med_ex>0 AND a.价格/fa.med_ex>=1.15 THEN 1 WHEN fa.n_ex>=1 THEN 0 END AS "特征A_贵于邻近中位≥15%",
        sd.n_comp, sd.n_airlines, sd.comp_min, sd.comp_med, sd.comp_max, sd.x_rank_pct,
        CASE WHEN sd.comp_min IS NOT NULL THEN a.价格-sd.comp_min END AS gap_to_min,
        CASE WHEN sd.comp_min>0 THEN a.价格/sd.comp_min END AS prev_over_min,
        CASE WHEN sd.comp_max IS NOT NULL THEN a.价格-sd.comp_max END AS gap_to_max,
        CASE WHEN sd.comp_max>0 THEN a.价格/sd.comp_max END AS prev_over_max,
        CASE WHEN sd.comp_med>0 THEN a.价格/sd.comp_med END AS x_over_median,
        CASE WHEN sd.comp_max>0 THEN a.价格/sd.comp_max END AS x_over_max,
        sd.n_comp AS n_comp_all, sd.n_airlines AS n_air_all, sd.comp_min AS comp_min_all,
        CASE WHEN sd.comp_min IS NOT NULL THEN a.价格-sd.comp_min END AS gap_to_min_all,
        CASE WHEN sd.comp_min>0 THEN a.价格/sd.comp_min END AS prev_over_min_all,
        CASE WHEN sd.n_comp>0 THEN 1 ELSE 0 END AS has_af_comp,
        sd.comp_min AS comp_min_same,
        CASE WHEN sd.comp_min IS NOT NULL THEN a.价格-sd.comp_min END AS gap_to_min_same,
        CASE WHEN sd.comp_min>0 THEN a.价格/sd.comp_min END AS prev_over_min_same,
        sd.n_flights_day, sd.n_airlines_day, sd.n_priced_day, sd.g3_n_sale,
        sd.g3_n_sale AS sp_n_onsale, sd.n_airlines_day AS sp_n_airlines_onsale,
        pm.n_flights_pm2h, pm.n_airlines_pm2h, pm.n_priced_pm2h,
        CASE WHEN pm.n_flights_pm2h>0 THEN 1 ELSE 0 END AS has_comp_schedule,
        CASE WHEN sd.min_day IS NOT NULL THEN a.价格-sd.min_day END AS gap_to_min_day,
        CASE WHEN sd.min_day>0 THEN a.价格/sd.min_day END AS prev_over_min_day,
        CASE WHEN pm.comp_min_pm2h IS NOT NULL THEN a.价格-pm.comp_min_pm2h END AS gap_to_min_pm2h,
        CASE WHEN pm.comp_min_pm2h>0 THEN a.价格/pm.comp_min_pm2h END AS prev_over_min_pm2h,
        mk.n_cmp AS g2_n_cmp,
        CASE WHEN mk.n_cmp>0 THEN mk.n_ri*1.0/mk.n_cmp END AS g2_mkt_rise_share,
        CASE WHEN mk.n_cmp>0 THEN mk.n_dr*1.0/mk.n_cmp END AS g2_mkt_drop_share,
        CASE WHEN pmm.n_cmp_pm1>0 THEN pmm.n_ri_pm1*1.0/pmm.n_cmp_pm1 END AS g2_mkt_rise_share_pm1,
        CASE WHEN pmm.n_cmp_pm1>0 THEN pmm.n_dr_pm1*1.0/pmm.n_cmp_pm1 END AS g2_mkt_drop_share_pm1,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN mk.mn/nullif(mk.mn_prev,0)-1 END AS g2_mkt_min_chg,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN
          CASE WHEN mk.mn>mk.mn_prev THEN 1 WHEN mk.mn<mk.mn_prev THEN -1 ELSE 0 END END AS g2_mktmin_dir,
        CASE WHEN mk.n_cmp>0 THEN mk.n_ri*1.0/mk.n_cmp END AS g2_oth_bigup_share,
        CASE WHEN mk.n_cmp>0 THEN mk.n_dr*1.0/mk.n_cmp END AS g2_oth_bigdn_share,
        CASE WHEN mk.n_cmp>0 THEN (mk.n_ri-mk.n_dr)*1.0/mk.n_cmp END AS g2_oth_net_move_share,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN mk.mn/nullif(mk.mn_prev,0)-1 END AS g2_othmin_chg_pct,
        CASE WHEN date_diff('day', mk.od_prev, mk.观察日)<=3 THEN mk.n_sale-mk.ns_prev END AS g3_n_sale_chg1,
        mk.n_sale - mk.ns_7 AS g3_n_sale_chg7,
        coalesce(ent.n_enter,0) AS g3_enter_7d,
        0 AS g3_exit_7d,
        sib.n_sib, sib.sib_median, sib.sib_mean AS sib_mean_same, sib.n_sib AS n_sib_same,
        CASE WHEN sib.sib_median>0 THEN a.价格/sib.sib_median END AS sib_price_level,
        CASE WHEN sib.sib_mean IS NOT NULL AND sib.n_sib>0 THEN
          CASE WHEN a.价格 > sib.sib_mean THEN 0.0 ELSE 1.0 END END AS sib_low_ratio,
        CASE WHEN sib.sib_min>0 THEN a.价格/sib.sib_min END AS sib_cur_vs_min_avg,
        CASE WHEN sib.sib_min>0 THEN sib.sib_min/a.价格 END AS sib_drop_ratio,
        se.sib_bigup_share, se.sib_chg_mean,
        se.sib_bigup_share AS sib_bigup_rate_hist,
        CASE WHEN se.sib_bigup_share IS NOT NULL THEN 1-se.sib_bigup_share END AS sib_recent_drop_rate,
        re.route_bigup_share, r7.route_bigup_share_7d,
        od.n_od_same_flight,
        coalesce(cs.is_codeshare_raw,0) AS is_codeshare,
        CAST(NULL AS VARCHAR) AS argmin_cls,
        CAST(NULL AS DOUBLE) AS argmin_lag_days,
        CAST(NULL AS BIGINT) AS n_post,
        CAST(NULL AS BIGINT) AS n_same_in_ex,
        CAST(NULL AS BIGINT) AS n_stale
      FROM base a
      LEFT JOIN fa USING (观察日,航班号,航线,起飞日)
      LEFT JOIN sd2 sd USING (观察日,航班号,航线,起飞日)
      LEFT JOIN pm2h pm USING (观察日,航班号,航线,起飞日)
      LEFT JOIN mk3 mk USING (观察日,航线,起飞日)
      LEFT JOIN pm pmm USING (观察日,航线,起飞日)
      LEFT JOIN ent USING (观察日,航线,起飞日)
      LEFT JOIN sib USING (观察日,航班号,航线,起飞日)
      LEFT JOIN sib_ev se USING (观察日,航班号,航线,起飞日)
      LEFT JOIN route_ev re ON re.航线=a.航线 AND re.观察日=a.观察日
      LEFT JOIN route_ev7 r7 ON r7.航线=a.航线 AND r7.观察日=a.观察日
      LEFT JOIN odn od ON od.观察日=a.观察日 AND od.航班号=a.航班号 AND od.起飞日=a.起飞日
      LEFT JOIN cs ON cs.观察日=a.观察日 AND cs.航班号=a.航班号 AND cs.航线=a.航线 AND cs.起飞日=a.起飞日
    ) TO '{out}' (FORMAT PARQUET)
    """)
    log(f"03_竞争横截面 done {time.time()-t0:.0f}s -> {out}")
    done.write_text(json.dumps({"secs": round(time.time()-t0)}, ensure_ascii=False), encoding="utf-8")
    c.close()

if __name__ == "__main__":
    main()
