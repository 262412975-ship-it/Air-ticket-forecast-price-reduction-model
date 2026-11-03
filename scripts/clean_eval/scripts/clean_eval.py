#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
干净口径评估脚本（只读、离线）。2026-10-26 依据《流程体检_20261026》结论。

一句话：给它「打分表」和「价格路径」，它按固定阈值、只看已走完7天的日子、价格只用该次快照、
同航班重复要标明，算出：精度、召回、每单净省钱（涨价亏损按 k=2 和 k=1 两档并列）。

不连网、不连库、不碰线上推理服务、不跑 ana_extract；只读输入文件，只写到 --输出目录。
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
import argparse, glob, hashlib, json, re, shutil, tempfile, datetime as dt
import numpy as np, pandas as pd, duckdb

# ---------------------------------------------------------------- 参数
def 解析参数():
    p = argparse.ArgumentParser(description="干净口径评估（固定阈值、剔除未满窗日、末批价、重复标注、每单净省钱 k=2/k=1）")
    p.add_argument("--打分表", required=True, help="parquet 或 csv：每行=某航班某出发日在某观察日的一次打分")
    p.add_argument("--价格路径", required=True, help="时间线目录(含 obs=YYYY-MM-DD/part.parquet) 或 增量批次目录(含 batch=*/batch.parquet)")
    p.add_argument("--价格路径类型", required=True, choices=["时间线", "批次"], help="时间线=每键每天已是末批价；批次=逐批抓取，脚本按「每天只看最后一批」自己挑")
    p.add_argument("--阈值", required=True, type=float, help="固定阈值（分数>=阈值算告警）。禁止在评估集上挑！")
    p.add_argument("--阈值来源", required=True, choices=["现网固定阈", "验证集选定", "其他(须在--阈值说明写清)"], help="阈值怎么来的（必填，会写进每张表）")
    p.add_argument("--阈值说明", default="", help="一句话说明阈值出处（文件/日期）")
    p.add_argument("--窗口天数", type=int, default=7)
    p.add_argument("--k档", default="2,1", help="涨价亏损权重，逗号分隔，默认 2,1（并列报告）")
    p.add_argument("--降价元数", type=float, default=100.0); p.add_argument("--降价比例", type=float, default=0.15)
    p.add_argument("--数据截止日", default="auto", help="价格路径里「已走完整天」的最后一天；auto=自动判断")
    p.add_argument("--近端线", type=int, default=8, help="距起飞天数<此值的行，7天窗口会被起飞截短；单独分档报告（默认8，同收单评估口径）")
    p.add_argument("--坏价日", default="", help="价格不可信的日子(逗号分隔)；观察日或窗口含这些日的行剔除")
    p.add_argument("--剔除标记", default="", help="例：'8001无源相关=1;回补分组!=不涉及回补'；分号分隔多条，值可逗号分隔多个")
    p.add_argument("--旧标签列", default="", help="打分表里旧口径的 0/1 真降标签列（给出则输出「旧口径→干净口径」阶梯对照）")
    p.add_argument("--观察日列", default=""); p.add_argument("--航班号列", default=""); p.add_argument("--航线列", default="")
    p.add_argument("--回补键列", default="", help="(可选)打分表里标记「是否回补键」的列(是/1/True 表示是)；只做标记，写进告警明细，不影响计算")
    p.add_argument("--出发日列", default=""); p.add_argument("--分数列", default=""); p.add_argument("--价格列", default=""); p.add_argument("--批次列", default="")
    p.add_argument("--抽样次数", type=int, default=300, help="按航班分组的自助抽样次数（算区间）；0=不算")
    p.add_argument("--输出目录", required=True); p.add_argument("--名称", default="示例")
    return p.parse_args()

候选列 = {
    "观察日": ["obs_day", "观察日S", "观察日", "day"], "航班号": ["flight_no", "航班号"], "航线": ["route", "航线"],
    "出发日": ["dep_date", "出发日"], "分数": ["score", "w7_new", "W7首次大降分", "分数"],
    "价格": ["price_prev", "price_t", "现价", "价格"], "批次": ["来源批次", "price_source", "价来源", "批次"],
}
def 找列(df, 指定, 键, 必须=True):
    if 指定:
        assert 指定 in df.columns, f"打分表里没有列 {指定}"; return 指定
    for c in 候选列[键]:
        if c in df.columns: return c
    if 必须: raise SystemExit(f"找不到「{键}」列，请用对应的 --{键}列 指定；表内列：{list(df.columns)}")
    return None

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def wilson(h, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = h / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)

# ---------------------------------------------------------------- 价格路径
def 建价格路径(con, 路径, 类型, 降价元数, 降价比例):
    """建 pp(flight_no,route,dep_date,day,price)=每键每天一个价（末批价）；批次类型另建 pb(键,batch,price)。返回(自动截止日, 说明dict)"""
    说明 = {}
    if 类型 == "时间线":
        files = sorted(glob.glob(os.path.join(路径, "obs=*", "part.parquet")))
        assert files, f"时间线目录里没找到 obs=*/part.parquet：{路径}"
        con.execute(f"""CREATE TABLE pp AS SELECT flight_no, route, CAST(dep_date AS VARCHAR) dep_date, CAST(obs_day AS VARCHAR) dy, CAST(price AS DOUBLE) price
            FROM read_parquet({[f for f in files]!r}, hive_partitioning=false) WHERE price>0""")
        days = sorted(re.search(r"obs=([\d-]+)", f).group(1) for f in files)
        说明.update(类型="时间线", 分区数=len(days), 最早日=days[0], 最晚日=days[-1])
        auto = days[-1]
    else:
        files = sorted(glob.glob(os.path.join(路径, "batch=*", "batch.parquet")))
        assert files, f"批次目录里没找到 batch=*/batch.parquet：{路径}"
        con.execute(f"""CREATE TABLE raw AS SELECT flight_no, route, CAST(date AS VARCHAR) dep_date, CAST(batch AS VARCHAR) batch, TRY_CAST(price AS DOUBLE) price
            FROM read_parquet({[f for f in files]!r}, union_by_name=true, hive_partitioning=false) WHERE TRY_CAST(price AS DOUBLE)>0""")
        # 同键同批取最低价（与回访口径一致）
        con.execute("CREATE TABLE pb AS SELECT flight_no, route, dep_date, batch, min(price) price FROM raw GROUP BY ALL")
        con.execute("ALTER TABLE pb ADD COLUMN dy VARCHAR"); con.execute("UPDATE pb SET dy=substr(batch,1,4)||'-'||substr(batch,5,2)||'-'||substr(batch,7,2)")
        con.execute("ALTER TABLE pb ADD COLUMN al VARCHAR"); con.execute("UPDATE pb SET al=substr(flight_no,1,2)")
        # 每(航司,天)：末批；若末批键数不足当天最大批的70%，改用最大批（与时间线口径一致）
        con.execute("""CREATE TABLE bs AS SELECT al, dy, batch, count(*) n FROM pb GROUP BY ALL""")
        con.execute("""CREATE TABLE sel AS
          WITH m AS (SELECT al, dy, max(n) mx, max(batch) lastb FROM bs GROUP BY al, dy),
               ln AS (SELECT m.al, m.dy, m.lastb, m.mx, (SELECT n FROM bs WHERE bs.al=m.al AND bs.dy=m.dy AND bs.batch=m.lastb) nl FROM m),
               big AS (SELECT bs.al, bs.dy, arg_max(bs.batch, bs.n) bigb FROM bs GROUP BY ALL)
          SELECT ln.al, ln.dy, CASE WHEN ln.nl >= 0.7*ln.mx THEN ln.lastb ELSE big.bigb END selb, ln.lastb, ln.nl>=0.7*ln.mx 末批够大
          FROM ln JOIN big USING (al, dy)""")
        con.execute("""CREATE TABLE pp AS
          WITH a AS (SELECT pb.flight_no, pb.route, pb.dep_date, pb.dy, pb.batch, pb.price, (pb.batch=sel.selb) is_sel
                     FROM pb JOIN sel ON sel.al=pb.al AND sel.dy=pb.dy)
          SELECT flight_no, route, dep_date, dy,
                 coalesce(max(price) FILTER (WHERE is_sel), arg_max(price, batch)) price,     -- 选中批没有该键→当天最后一次观测价
                 (max(price) FILTER (WHERE is_sel)) IS NULL AS fb
          FROM a GROUP BY flight_no, route, dep_date, dy""")
        fin = con.execute("SELECT dy, max(batch) FROM bs GROUP BY dy ORDER BY dy").fetchall()
        完整 = [d for d, b in fin if int(b[8:12]) >= 1900]  # 当天出现过晚间(>=19:00)批次，才算当天走完
        说明.update(类型="批次", 批次数=len(files), 最早日=fin[0][0], 最晚日=fin[-1][0], 最晚日末批=fin[-1][1], 走完整天的最后一天=完整[-1] if 完整 else None)
        说明["键不在当天选中批、回退到当天最后一次观测价的(航班,出发日,天)数"] = int(con.execute("SELECT count(*) FROM pp WHERE fb").fetchone()[0])
        说明["价格路径键天总数"] = int(con.execute("SELECT count(*) FROM pp").fetchone()[0])
        说明["末批不足70%改用最大批的(航司,天)数"] = int(con.execute("SELECT count(*) FROM sel WHERE NOT 末批够大").fetchone()[0])
        auto = 完整[-1] if 完整 else fin[-1][0]
    return auto, 说明

# ---------------------------------------------------------------- 主流程
def main():
    A = 解析参数(); W = A.窗口天数; KS = [float(x) for x in A.k档.split(",")]
    out = A.输出目录; os.makedirs(out, exist_ok=True)
    tmpd = tempfile.mkdtemp(prefix="cleaneval_")
    t = pd.read_parquet(A.打分表) if A.打分表.endswith(".parquet") else pd.read_csv(A.打分表, dtype=str)
    列 = {k: 找列(t, getattr(A, k + "列"), k, 必须=(k in ["观察日", "航班号", "航线", "出发日", "分数"])) for k in 候选列}
    n_表内 = len(t)
    d = pd.DataFrame({"obs_day": t[列["观察日"]].astype(str).str[:10], "flight_no": t[列["航班号"]].astype(str), "route": t[列["航线"]].astype(str),
                      "dep_date": t[列["出发日"]].astype(str).str[:10], "score": pd.to_numeric(t[列["分数"]], errors="coerce")})
    d["表内行号"] = np.arange(len(t))
    d["list_price"] = pd.to_numeric(t[列["价格"]], errors="coerce") if 列["价格"] else np.nan
    d["src_batch"] = t[列["批次"]].astype(str).str.extract(r"(\d{12})")[0] if 列["批次"] else None
    if A.回补键列:
        assert A.回补键列 in t.columns, f"打分表里没有列 {A.回补键列}"
        d["回补键"] = t[A.回补键列].astype(str).str.strip().isin(["是", "1", "True", "true", "1.0"]).map({True: "是", False: "否"})
    if A.旧标签列: d["old_y"] = pd.to_numeric(t[A.旧标签列], errors="coerce")
    # 剔除标记
    d["异常标记"] = ""
    if A.剔除标记:
        for rule in [r for r in A.剔除标记.split(";") if r.strip()]:
            m = re.match(r"^(.+?)(!=|=)(.+)$", rule.strip()); assert m, f"剔除标记格式不对：{rule}"
            c, op, vals = m.group(1), m.group(2), [v.strip() for v in m.group(3).split(",")]
            assert c in t.columns, f"剔除标记里的列 {c} 不在打分表"
            hit = t[c].astype(str).isin(vals) if op == "=" else ~t[c].astype(str).isin(vals)
            d.loc[hit.values, "异常标记"] += f"{rule.strip()} "
    未打分 = d.score.isna()
    n_未打分 = int(未打分.sum())
    d = d[~未打分].copy()
    d["dtd"] = (pd.to_datetime(d.dep_date) - pd.to_datetime(d.obs_day)).dt.days
    d["告警"] = (d.score >= A.阈值).astype(int)
    坏价日 = [x.strip() for x in A.坏价日.split(",") if x.strip()]

    con = duckdb.connect(); con.execute(f"SET memory_limit='3GB'; SET threads=2; SET temp_directory='{tmpd}'; SET preserve_insertion_order=false")
    auto, 路径说明 = 建价格路径(con, A.价格路径, A.价格路径类型, A.降价元数, A.降价比例)
    截止日 = auto if A.数据截止日 == "auto" else A.数据截止日
    con.execute(f"DELETE FROM pp WHERE dy>'{截止日}'")   # 截止日之后的价不用（可能是未走完的半天）
    # 打分行 → 窗口结果
    con.register("sc", d[["表内行号", "flight_no", "route", "dep_date", "obs_day", "list_price", "src_batch"]])
    con.execute("CREATE TABLE s AS SELECT * FROM sc")
    drop_cond = f"(b.price <= p0 - {A.降价元数} OR b.price <= p0 * {1 - A.降价比例})"
    窗 = con.execute(f"""
      WITH base AS (
        SELECT s.*, o.price path_p0, CASE WHEN s.list_price IS NOT NULL AND NOT isnan(s.list_price) THEN s.list_price ELSE o.price END p0
        FROM s LEFT JOIN pp o ON o.flight_no=s.flight_no AND o.route=s.route AND o.dep_date=s.dep_date AND o.dy=s.obs_day),
      j AS (SELECT base.表内行号, base.p0, b.dy bd, b.price bp, date_diff('day', CAST(base.obs_day AS DATE), CAST(b.dy AS DATE)) h
            FROM base JOIN pp b ON b.flight_no=base.flight_no AND b.route=base.route AND b.dep_date=base.dep_date
              AND b.dy>base.obs_day AND CAST(b.dy AS DATE) <= CAST(base.obs_day AS DATE) + INTERVAL {W} DAY WHERE base.p0 IS NOT NULL),
      g AS (SELECT 表内行号, count(*) n_win,
              arg_min(bp, bd) FILTER (WHERE bp <= p0-{A.降价元数} OR bp <= p0*{1-A.降价比例}) p_drop,
              min(h) FILTER (WHERE bp <= p0-{A.降价元数} OR bp <= p0*{1-A.降价比例}) h_drop,
              arg_max(bp, bd) p_end, max(bd) d_end FROM j GROUP BY 表内行号)
      SELECT base.表内行号, base.path_p0, base.p0, g.n_win, g.p_drop, g.h_drop, g.p_end, g.d_end FROM base LEFT JOIN g USING (表内行号)""").df()
    d = d.merge(窗, on="表内行号", how="left")
    # 批次类型：名单价是否等于「来源批次」里该键的价
    d["批次核对"] = ""
    if A.价格路径类型 == "批次" and d.src_batch.notna().any():
        con.register("sc2", d.loc[d.src_batch.notna(), ["表内行号", "flight_no", "route", "dep_date", "src_batch"]])
        r = con.execute("""SELECT x.表内行号, pb.price bprice FROM sc2 x LEFT JOIN pb ON pb.flight_no=x.flight_no AND pb.route=x.route AND pb.dep_date=x.dep_date AND pb.batch=x.src_batch""").df()
        d = d.merge(r, on="表内行号", how="left")
        has = d.src_batch.notna()
        d.loc[has & d.bprice.isna(), "批次核对"] = "来源批次里没有该键"
        d.loc[has & d.bprice.notna() & d.list_price.notna() & ((d.bprice - d.list_price).abs() > 0.5), "批次核对"] = "名单价与来源批次价对不上"
    # ---- 剔除原因（按顺序，取第一条命中的）
    d["满窗最晚数据日"] = (pd.to_datetime(d.obs_day) + pd.Timedelta(days=W)).dt.strftime("%Y-%m-%d")
    d["未满窗"] = d.满窗最晚数据日 > 截止日
    d["原因"] = ""
    def 设(mask, txt):
        m = mask & (d.原因 == ""); d.loc[m, "原因"] = txt
    设(d.未满窗, f"未满窗（观察日+{W}天 超过数据截止日{截止日}）")
    if 坏价日:
        bad = pd.Series(False, index=d.index)
        for b in 坏价日:
            bad |= (d.obs_day <= b) & (b <= d.满窗最晚数据日)
        设(bad, "观察日或窗口含坏价日(" + ",".join(坏价日) + ")")
    if d.src_batch.notna().any():
        设(d.src_batch.notna() & (d.src_batch.str[:4] + "-" + d.src_batch.str[4:6] + "-" + d.src_batch.str[6:8] != d.obs_day), "旧价结转（来源批次日期不是观察日）")
    设(d.批次核对 == "来源批次里没有该键", "名单价对不上来源批次（该批没有此航班）")
    设(d.批次核对 == "名单价与来源批次价对不上", "名单价与来源批次价对不上")
    设(d.p0.isna(), "当天无快照价（禁止拿旧观察日价结转）")
    设(d.异常标记 != "", "数据异常标记")
    设(d.n_win.isna() | (d.n_win == 0), "窗口内无后续价（无法判定）")
    d["可评估"] = d.原因 == ""
    d["窗口被起飞截短"] = d.dtd < A.近端线
    d["名单价是否当天末批价"] = np.where(d.list_price.notna() & d.path_p0.notna(), np.where((d.list_price - d.path_p0).abs() <= 0.5, "是", "否"), "无法核对")
    # ---- 结果
    d["真降"] = d.p_drop.notna().astype(int)
    d["省的钱"] = np.where(d.真降 == 1, d.p0 - d.p_drop, 0.0)
    d["亏的钱"] = np.where(d.真降 == 1, 0.0, np.maximum(d.p_end - d.p0, 0.0))
    for k in KS: d[f"净省钱_k{k:g}"] = d.省的钱 - k * d.亏的钱
    key = ["flight_no", "route", "dep_date"]
    d["航班键"] = d.flight_no + "|" + d.route + "|" + d.dep_date
    # ---- 重复航班标注（只在「可评估的告警行」里数，按观察日先后）
    a = d[d.可评估 & (d.告警 == 1)].sort_values(["航班键", "obs_day"]).copy()
    a["该航班第几次告警"] = a.groupby("航班键").cumcount() + 1
    a["该航班告警总天数"] = a.groupby("航班键")["航班键"].transform("size")
    a["是否首次告警"] = (a.该航班第几次告警 == 1).astype(int)
    a["是否末次告警"] = (a.该航班第几次告警 == a.该航班告警总天数).astype(int)
    d = d.merge(a[["表内行号", "该航班第几次告警", "该航班告警总天数", "是否首次告警", "是否末次告警"]], on="表内行号", how="left")
    全部告警天数 = d[d.告警 == 1].groupby("航班键").obs_day.nunique().rename("该航班全部告警天数含被剔除日")
    d = d.merge(全部告警天数, left_on="航班键", right_index=True, how="left")

    # ================= 输出1：每日明细
    def 指标(x, 告警x, 含召回=True):
        n = len(告警x); h = int(告警x.真降.sum()) if n else 0
        r = {"告警行数": n, "命中行数": h, "精度": h / n if n else np.nan}
        if 含召回:
            pos = int(x.真降.sum()); r["真降行数"] = pos; r["召回"] = h / pos if pos else np.nan
            r["基础降价率"] = pos / len(x) if len(x) else np.nan
            r["精度是基础降价率的几倍"] = r["精度"] / r["基础降价率"] if r["基础降价率"] else np.nan
        r["降价收益合计"] = float(告警x.省的钱.sum()); r["涨价亏损合计"] = float(告警x.亏的钱.sum())
        for k in KS:
            r[f"每单净省钱_k{k:g}"] = float(告警x[f"净省钱_k{k:g}"].mean()) if n else np.nan
            r[f"净省钱合计_k{k:g}"] = float(告警x[f"净省钱_k{k:g}"].sum())
        return r
    有全量 = (d.告警 == 0).any()
    日表 = []
    for day, g in d.groupby("obs_day"):
        e = g[g.可评估]; ea = e[e.告警 == 1]
        行 = {"观察日": day, "表内已打分行数": len(g), "满窗状态": "未满窗" if g.未满窗.iloc[0] else "已满窗",
              "满窗要等到的数据日": g.满窗最晚数据日.iloc[0], "窗口已走天数": int(max(0, min(W, (pd.Timestamp(截止日) - pd.Timestamp(day)).days))), "是否计入验收": "是" if len(e) else "否", "可评估行数": len(e),
              "整日被剔除的主要原因": "" if len(e) else g.原因.value_counts().index[0], "告警行数(含被剔除)": int((g.告警 == 1).sum())}
        m = 指标(e, ea, 含召回=有全量) if len(e) else {}
        if A.旧标签列:
            ga = g[(g.告警 == 1) & g.old_y.notna()]
            行["旧口径精度(表内旧标签,不剔除任何行)"] = ga.old_y.mean() if len(ga) else np.nan
        行.update(m); 行["告警里同航班此前已告警过的行数"] = int((ea.该航班第几次告警 > 1).sum()) if len(ea) else 0
        日表.append(行)
    日表 = pd.DataFrame(日表)
    日表.insert(1, "阈值", A.阈值)
    日表.to_csv(f"{out}/{A.名称}_每日明细.csv", index=False, encoding="utf-8-sig")

    # ================= 输出2：剔除清单
    剔 = d[~d.可评估].groupby(["obs_day", "原因"]).size().rename("被剔除行数").reset_index()
    tot = d.groupby("obs_day").size().rename("当天已打分行数"); 剔 = 剔.merge(tot, left_on="obs_day", right_index=True)
    剔["占当天比例"] = 剔.被剔除行数 / 剔.当天已打分行数
    剔["是否整日剔除"] = np.where(剔.被剔除行数 == 剔.当天已打分行数, "是", "否")
    剔.columns = ["观察日", "剔除原因", "被剔除行数", "当天已打分行数", "占当天比例", "是否整日剔除"]
    剔.to_csv(f"{out}/{A.名称}_剔除清单.csv", index=False, encoding="utf-8-sig")
    # 剔除后「被剔除的告警行」另算
    # ================= 输出3：告警明细（含重复标注）
    cols = {"obs_day": "观察日", "flight_no": "航班号", "route": "航线", "dep_date": "出发日", "dtd": "距起飞天数", "score": "分数", "p0": "基准价",
            "src_batch": "价格来源批次", "名单价是否当天末批价": "名单价是否当天末批价", "窗口被起飞截短": "窗口被起飞截短", "真降": "是否真降", "p_drop": "首次降到线的价",
            "h_drop": "第几天降到线", "p_end": "窗口最后一次价", "省的钱": "降价收益", "亏的钱": "涨价亏损"}
    for k in KS: cols[f"净省钱_k{k:g}"] = f"每单净省钱_k{k:g}"
    cols.update({"该航班第几次告警": "该航班第几次告警", "该航班告警总天数": "该航班告警总天数(可评估日内)", "该航班全部告警天数含被剔除日": "该航班全部告警天数(含被剔除日)", "是否首次告警": "是否首次告警(默认规则保留)", "是否末次告警": "是否末次告警"})
    if A.回补键列: cols["回补键"] = "是否回补键"
    ad = d[d.可评估 & (d.告警 == 1)][list(cols)].rename(columns=cols).sort_values(["航班号", "航线", "出发日", "观察日"])
    ad.to_csv(f"{out}/{A.名称}_告警明细_含重复标注.csv", index=False, encoding="utf-8-sig")

    # ================= 输出4：整体汇总（口径 × 是否去重），k 并列；区间用按航班分组自助抽样
    E = d[d.可评估].copy()
    rng = np.random.default_rng(20261026)
    def 区间(sub, 列s):
        """按航班分组自助抽样 → 精度、各k每单净省钱 的 95% 区间"""
        if A.抽样次数 <= 0 or len(sub) == 0: return {}
        gs = sub.groupby("航班键").agg(n=("真降", "size"), h=("真降", "sum"), **{f"s{k:g}": (f"净省钱_k{k:g}", "sum") for k in KS})
        arr = gs.to_numpy(float); G = len(arr); res = []
        for _ in range(A.抽样次数):
            idx = rng.integers(0, G, G); s = arr[idx].sum(0); res.append([s[1] / s[0]] + [s[2 + i] / s[0] for i in range(len(KS))])
        res = np.array(res); o = {}
        o["精度_区间下限"], o["精度_区间上限"] = np.percentile(res[:, 0], [2.5, 97.5])
        for i, k in enumerate(KS): o[f"每单净省钱_k{k:g}_区间下限"], o[f"每单净省钱_k{k:g}_区间上限"] = np.percentile(res[:, 1 + i], [2.5, 97.5])
        return o
    汇 = []
    口径们 = [("全部可评估行(含近端)", E), (f"只看距起飞≥{A.近端线}天(窗口不被起飞截短)", E[~E.窗口被起飞截短])]
    for 口径名, X in 口径们:
        XA = X[X.告警 == 1]
        规则们 = [("同航班每个告警日都算(行级，不去重)", XA, X, True), ("同航班只算第一次告警(默认规则)", XA[XA.是否首次告警 == 1], None, True),
                  ("同航班只算最后一次告警", XA[XA.是否末次告警 == 1], None, True), ("航班等权(每航班内先平均、再对航班平均)", XA, None, False),
                  ("航班任一告警日命中即算命中(只对精度有意义)", XA, None, False)]
        for 规则名, 告警部分, 全集, 要区间 in 规则们:
            r = {"口径": 口径名, "是否去重": 规则名, "阈值": A.阈值, "阈值来源": A.阈值来源, "窗口天数": W, "数据截止日": 截止日,
                 "计入的观察日数": X.obs_day.nunique(), "计入观察日": ",".join(sorted(X.obs_day.unique()))}
            if 全集 is not None: r.update(指标(全集, 告警部分, 含召回=有全量))
            elif 规则名.startswith("航班等权"):
                gq = 告警部分.groupby("航班键").agg(h=("真降", "mean"), **{f"n{k:g}": (f"净省钱_k{k:g}", "mean") for k in KS})
                r.update({"告警行数": len(告警部分), "精度": gq.h.mean(), "召回": np.nan})
                for k in KS: r[f"每单净省钱_k{k:g}"] = gq[f"n{k:g}"].mean()
            elif 规则名.startswith("航班任一"):
                gq = 告警部分.groupby("航班键").真降.max()
                r.update({"告警行数": len(告警部分), "命中行数": int(gq.sum()), "精度": gq.mean(), "召回": np.nan})
            else:
                r.update(指标(X, 告警部分, 含召回=False)); r["召回"] = np.nan
            r["涉及航班数"] = 告警部分.航班键.nunique(); r["平均每航班告警次数"] = len(告警部分) / max(r["涉及航班数"], 1)
            if "命中行数" in r and 规则名[:4] != "航班等权":
                lo = wilson(r["命中行数"], r["告警行数"] if 规则名[:2] != "航班" else r["涉及航班数"]); r["精度_简单区间下限(行当独立)"], r["精度_简单区间上限(行当独立)"] = lo
            if 要区间: r.update(区间(告警部分, KS))
            汇.append(r)
    汇 = pd.DataFrame(汇)
    汇.to_csv(f"{out}/{A.名称}_整体汇总.csv", index=False, encoding="utf-8-sig")

    # ================= 输出5：阶梯对照（旧口径→干净口径）
    阶梯 = []
    if A.旧标签列:
        def 行(名, x, 标签, 说明):
            xa = x[x.告警 == 1]; n = len(xa)
            阶梯.append({"步骤": 名, "做了什么": 说明, "涉及观察日": x.obs_day.nunique(), "告警行数": n, "精度": xa[标签].mean() if n else np.nan,
                         "日均告警": n / max(x.obs_day.nunique(), 1)})
        base = d[d.old_y.notna()]
        行("0 旧口径(全部日、旧标签、每个告警日都算)", base, "old_y", "含未满窗日；真降标签=打分表自带旧标签")
        s1 = base[~base.未满窗]; 行("1 去掉未满窗日", s1, "old_y", f"只留 观察日+{W}天 ≤ 数据截止日{截止日} 的日子")
        s2 = d[d.可评估 & d.old_y.notna()]; 行("2 再去掉异常/无快照/旧价结转/无后续价的行", s2, "old_y", "按本脚本的剔除清单；标签仍是旧标签")
        行("3 干净标签(按每天末批价重算真降)", d[d.可评估], "真降", "标签改为按价格路径重算，价格只用该次快照；阈值不变")
        s4 = d[d.可评估 & (d.是否首次告警 == 1) | (d.可评估 & (d.告警 == 0))]; 行("4 同航班只算第一次告警(默认规则)", s4, "真降", "同一航班多次告警只留第一次")
        阶梯 = pd.DataFrame(阶梯); 阶梯["比上一步变化(百分点)"] = 阶梯.精度.diff() * 100; 阶梯["比旧口径变化(百分点)"] = (阶梯.精度 - 阶梯.精度.iloc[0]) * 100
        阶梯.to_csv(f"{out}/{A.名称}_旧口径到干净口径阶梯.csv", index=False, encoding="utf-8-sig")
    # ================= 输出6：运行说明
    说明 = {"名称": A.名称, "运行时间(北京时间)": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "命令行参数": vars(A),
            "输入文件md5": {"打分表": md5(A.打分表)}, "识别到的列": 列, "表内行数": n_表内, "无分数行数(如缺价未打分)": n_未打分,
            "价格路径": 路径说明, "实际使用的数据截止日": 截止日, "满窗规则": f"观察日+{W}天 <= 数据截止日 才算满窗",
            "阈值": A.阈值, "阈值来源": A.阈值来源, "阈值说明": A.阈值说明, "k档": KS,
            "名单价与当天末批价核对": d.名单价是否当天末批价.value_counts().to_dict(),
            "旧标签与重算标签不一致的可评估行数": (int((d[d.可评估 & d.old_y.notna()].old_y != d[d.可评估 & d.old_y.notna()].真降).sum()) if A.旧标签列 else "未给旧标签"),
            "表内是否含未告警行(决定能否算召回)": bool(有全量),
            "真降定义": f"观察日之后{W}个日历天内，某天末批价 <= 基准价-{A.降价元数:g}元 或 <= 基准价×{1-A.降价比例:g}",
            "净省钱定义": "真降：省的钱=基准价-首次降到线的价，亏=0；没降：按窗口最后一次价买，价格比基准价高才算亏(亏=终点价-基准价)；每单净省钱=省的钱-k×亏的钱"}
    for p in [A.价格路径]: pass
    json.dump(说明, open(f"{out}/{A.名称}_运行说明.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    shutil.rmtree(tmpd, ignore_errors=True)
    # 屏幕摘要
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("数据截止日", 截止日, "| 价格路径", 路径说明)
    print(日表[["观察日", "满窗状态", "可评估行数", "告警行数", "精度", "每单净省钱_k2", "每单净省钱_k1"] if "每单净省钱_k2" in 日表 else 日表.columns].to_string())
    print(汇[["口径", "是否去重", "告警行数", "精度", "召回", "每单净省钱_k2", "每单净省钱_k1"]].to_string())
    if A.旧标签列: print(阶梯.to_string())

if __name__ == "__main__":
    main()
