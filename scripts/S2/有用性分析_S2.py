#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S2 有用性分析（11-02 重训）：同义组只计 1 次；尺子=净省钱 k1/k2；训练期内时间切分；不碰测试集挑特征；不训练正式模型、不上线。
口径对齐 clean_eval v1.5 / 构建重训数据集 v2.1：降到线=价<=p0-100 或 <=p0*0.85；窗=7；读法B dtd>=14；exclude-from=2026-10-28；gap>=7。
"""
from __future__ import annotations
import os, sys, json, time, hashlib, gc
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
S0 = ROOT / "S0_底表/数据/主表_全航司日价_dtd0_30.parquet"
S1 = ROOT / "S1_特征/数据/特征矩阵_全航司_dtd0_30.parquet"
DEDUP = ROOT / "S1_特征/核对/去重对照表.csv"
COV = ROOT / "S1_特征/核对/特征覆盖率.csv"
OUT = ROOT / "S2_有用性"
DATA, CHK, LOG, MID, SCR = OUT / "数据", OUT / "核对", OUT / "日志", OUT / "中间", OUT / "脚本"
for d in (DATA, CHK, LOG, MID):
    d.mkdir(parents=True, exist_ok=True)

EXCLUDE_FROM = "2026-10-28"
W = 7
DROP_YUAN, DROP_RATIO = 100.0, 0.15
DTD_MIN_B = 14  # 读法B
CUTOFF = "2026-11-01"  # S0 最晚观察日=数据截止
# gap>=7：train 末到 val 初
TRAIN_DAYS = ["2026-10-15", "2026-10-16"]
VAL_DAYS = ["2026-10-24", "2026-10-25"]
# 报告用冻结出手比例网格（在 VAL 上选峰，不在测试挑）
XS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
MIN_N = 500
PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=63, feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
              min_data_in_leaf=200, verbosity=-1, n_jobs=2, seed=42)
ROUNDS, EARLY = 400, 30
SEED = 42


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG / "运行日志.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_labels():
    """从 S0 日价路径打 7 天窗标签；只写满窗行。"""
    outp = DATA / "标签_全航司_满窗.parquet"
    if outp.exists() and (MID / "标签统计.json").exists():
        log(f"标签已存在，跳过重建: {outp}")
        return outp
    log("开始打标签…")
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='8GB'")
    con.execute(f"""
    CREATE TABLE pp AS
    SELECT 观察日 AS dy, 航班号 AS flight_no, 航线 AS route, 起飞日 AS dep_date,
           CAST(价格 AS DOUBLE) AS price, 距起飞天数 AS dtd, 是否满7天窗口, 是否回补键
    FROM read_parquet('{S0}')
    WHERE 价格 > 0
    """)
    # 窗内后续价
    con.execute(f"""
    CREATE TABLE lab AS
    WITH base AS (
      SELECT dy AS obs_day, flight_no, route, dep_date, price AS p0, dtd, 是否满7天窗口, 是否回补键
      FROM pp
      WHERE 是否满7天窗口='是'
        AND obs_day < '{EXCLUDE_FROM}'
        AND dtd >= {DTD_MIN_B}
        AND dtd <= 30
    ),
    j AS (
      SELECT b.obs_day, b.flight_no, b.route, b.dep_date, b.p0, b.dtd, b.是否回补键,
             p.dy AS bd, p.price AS bp,
             date_diff('day', CAST(b.obs_day AS DATE), CAST(p.dy AS DATE)) AS h
      FROM base b
      JOIN pp p
        ON p.flight_no=b.flight_no AND p.route=b.route AND p.dep_date=b.dep_date
       AND p.dy > b.obs_day
       AND CAST(p.dy AS DATE) <= CAST(b.obs_day AS DATE) + INTERVAL {W} DAY
    ),
    g AS (
      SELECT obs_day, flight_no, route, dep_date, p0, dtd, 是否回补键,
             count(*) AS n_win,
             arg_min(bp, bd) FILTER (WHERE bp <= p0 - {DROP_YUAN} OR bp <= p0 * {1-DROP_RATIO}) AS p_drop,
             min(h) FILTER (WHERE bp <= p0 - {DROP_YUAN} OR bp <= p0 * {1-DROP_RATIO}) AS h_drop,
             arg_max(bp, bd) AS p_end,
             min(bp) AS p_min
      FROM j
      GROUP BY 1,2,3,4,5,6,7
    )
    SELECT *,
      CASE WHEN p_drop IS NOT NULL THEN 1 ELSE 0 END AS 标签_7天降到线,
      CASE WHEN p_drop IS NOT NULL THEN p0 - p_drop ELSE 0 END AS 降价收益_若收单,
      CASE WHEN p_drop IS NULL THEN greatest(p_end - p0, 0) ELSE 0 END AS 涨价亏损_若收单,
      CASE WHEN p_drop IS NOT NULL THEN p0 - p_drop ELSE 0 END
        - 1 * (CASE WHEN p_drop IS NULL THEN greatest(p_end - p0, 0) ELSE 0 END) AS 每单净省钱_k1,
      CASE WHEN p_drop IS NOT NULL THEN p0 - p_drop ELSE 0 END
        - 2 * (CASE WHEN p_drop IS NULL THEN greatest(p_end - p0, 0) ELSE 0 END) AS 每单净省钱_k2,
      '满窗已定' AS 标签状态,
      '读法B_dtd>=14' AS 出手口径
    FROM g
    WHERE n_win IS NOT NULL AND n_win > 0
    """)
    con.execute(f"COPY lab TO '{outp}' (FORMAT PARQUET)")
    st = con.execute("""
    SELECT count(*) 行数,
           count(DISTINCT obs_day) 观察日数,
           min(obs_day) 最早, max(obs_day) 最晚,
           round(avg(标签_7天降到线)*100,2) 正例率pct,
           round(avg(每单净省钱_k1),2) 全收单k1,
           round(avg(每单净省钱_k2),2) 全收单k2,
           sum((是否回补键='是')::INT) 回补行
    FROM lab
    """).df().iloc[0].to_dict()
    by = con.execute("""
    SELECT obs_day 观察日, count(*) 行数,
           round(avg(标签_7天降到线)*100,2) 正例率pct,
           round(avg(每单净省钱_k1),2) k1, round(avg(每单净省钱_k2),2) k2
    FROM lab GROUP BY 1 ORDER BY 1
    """).df()
    by.to_csv(CHK / "标签_按观察日.csv", index=False, encoding="utf-8-sig")
    gap = (pd.Timestamp(VAL_DAYS[0]) - pd.Timestamp(TRAIN_DAYS[-1])).days - 1
    st.update({
        "exclude_from": EXCLUDE_FROM, "数据截止日": CUTOFF, "窗口天": W,
        "降价元": DROP_YUAN, "降价比例": DROP_RATIO, "读法B_dtd下限": DTD_MIN_B,
        "训练观察日": TRAIN_DAYS, "验证观察日": VAL_DAYS,
        "实际gap天数_训练末到验证初": gap, "gap要求": 7,
        "gap是否满足": bool(gap >= 7),
        "标签文件": str(outp), "标签md5": md5(outp),
    })
    json.dump(st, open(MID / "标签统计.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"标签完成: 行={st['行数']}, 正例率={st['正例率pct']}%, gap={gap}, md5={st['标签md5']}")
    con.close()
    return outp


def load_groups():
    g = pd.read_csv(DEDUP)
    reps = (g.groupby("同义组号", as_index=False)
            .agg(代表特征=("代表特征", "first"),
                 成员特征=("成员特征", lambda s: "|".join(s)),
                 成员数=("成员特征", "size")))
    cov = pd.read_csv(COV)
    reps = reps.merge(cov.rename(columns={"特征名": "代表特征"}), on="代表特征", how="left")
    return reps


def risk_tags(name: str, cov_row) -> list[str]:
    tags = []
    n = name
    if any(k in n for k in ["chunyun", "春运", "spring_festival"]):
        tags.append("春运外推")
    if (cov_row is not None and cov_row.get("是否需航管历史") == "是") or any(
        k in n for k in ["ly_", "tr_p", "train_", "prev_vs_ly", "prev_vs_train",
                         "dest_avg", "dest_drop", "orig_avg", "orig_drop", "route_recent"]
    ):
        tags.append("跨期过拟合_航管2025")
    if any(k in n for k in ["shuyun", "dep_month", "dep_season", "dep_is_off",
                            "golden_week", "holiday", "hol_", "g1_", "hd_dep",
                            "is_holiday", "days_to_hol", "days_from_hol"]):
        tags.append("季节_年份绑死风险")
    cov = float(cov_row["覆盖率"]) if cov_row is not None and pd.notna(cov_row.get("覆盖率")) else 1.0
    if cov == 0:
        tags.append("零覆盖")
    elif cov < 0.5:
        tags.append("低覆盖")
    return tags


def pick_net(score, y, save, loss, xs=XS, min_n=MIN_N):
    order = np.argsort(-score)
    y, save, loss, score = y[order], save[order], loss[order], score[order]
    n = len(y)
    rows, best = [], None
    for x in xs:
        k = max(1, int(round(n * x)))
        s1 = float((save[:k] - loss[:k]).mean())
        s2 = float((save[:k] - 2 * loss[:k]).mean())
        row = {"出手比例": x, "条数": k, "阈值": float(score[k - 1]),
               "每单净省钱_k1": s1, "每单净省钱_k2": s2,
               "精度": float(y[:k].mean()), "召回": float(y[:k].sum() / max(y.sum(), 1))}
        rows.append(row)
        if k >= min_n and (best is None or s2 > best["每单净省钱_k2"] or (
                abs(s2 - best["每单净省钱_k2"]) < 1e-12 and s1 > best["每单净省钱_k1"])):
            # 允许 s2<=0 时仍取最大，后面再标
            if s2 > 0 or best is None:
                best = row
    if best is None:
        ok = [r for r in rows if r["条数"] >= min_n]
        best = max(ok or rows, key=lambda r: (r["每单净省钱_k2"], r["每单净省钱_k1"]))
    return best, rows


def net_at_x(score, save, loss, x, day_index=None):
    """按天取前 x%（若给 day_index），否则全局前 x%。返回 k1/k2 每单。"""
    if day_index is None:
        n = len(score)
        k = max(1, int(round(n * x)))
        order = np.argsort(-score)[:k]
        s, l = save[order], loss[order]
        return float((s - l).mean()), float((s - 2 * l).mean()), k
    # 按天
    sel = []
    for d in np.unique(day_index):
        m = np.where(day_index == d)[0]
        if len(m) == 0:
            continue
        k = max(1, int(round(len(m) * x)))
        sc = score[m]
        order = m[np.argsort(-sc)[:k]]
        sel.append(order)
    if not sel:
        return np.nan, np.nan, 0
    idx = np.concatenate(sel)
    s, l = save[idx], loss[idx]
    return float((s - l).mean()), float((s - 2 * l).mean()), int(len(idx))


def main():
    t0 = time.time()
    log("=== S2 有用性分析开始 ===")
    lab_path = build_labels()
    groups = load_groups()
    feat_cols = groups["代表特征"].tolist()
    log(f"同义组代表数={len(feat_cols)}")

    # 读标签
    lab = pd.read_parquet(lab_path)
    lab = lab.rename(columns={"obs_day": "观察日", "flight_no": "航班号", "route": "航线",
                              "dep_date": "起飞日", "dtd": "距起飞天数", "p0": "价格"})
    # 切分
    tr = lab[lab["观察日"].isin(TRAIN_DAYS)].copy()
    va = lab[lab["观察日"].isin(VAL_DAYS)].copy()
    log(f"训练行={len(tr):,} 验证行={len(va):,} 正例率 train={tr['标签_7天降到线'].mean():.4f} val={va['标签_7天降到线'].mean():.4f}")

    # 读特征（只代表列+键）
    keys = ["观察日", "航班号", "航线", "起飞日"]
    log("读取特征矩阵代表列…")
    # 分块 merge 降内存：用 duckdb
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='10GB'")
    feat_list_sql = ", ".join(f'"{c}"' for c in feat_cols)
    con.execute(f"""
    CREATE TABLE lab AS SELECT * FROM read_parquet('{lab_path}')
    """)
    # rename in SQL view
    con.execute("""
    CREATE VIEW labk AS
    SELECT obs_day AS 观察日, flight_no AS 航班号, route AS 航线, dep_date AS 起飞日,
           dtd AS 距起飞天数, p0 AS 基准价, 标签_7天降到线, 降价收益_若收单, 涨价亏损_若收单,
           每单净省钱_k1, 每单净省钱_k2
    FROM lab
    """)
    con.execute(f"""
    CREATE TABLE feat AS
    SELECT 观察日, 航班号, 航线, 起飞日, {feat_list_sql}
    FROM read_parquet('{S1}')
    WHERE 观察日 IN ({",".join(repr(d) for d in TRAIN_DAYS+VAL_DAYS)})
      AND 距起飞天数 >= {DTD_MIN_B}
    """)
    con.execute("""
    CREATE TABLE j AS
    SELECT l.*, f.* EXCLUDE (观察日, 航班号, 航线, 起飞日)
    FROM labk l
    INNER JOIN feat f USING (观察日, 航班号, 航线, 起飞日)
    """)
    n_j = con.execute("SELECT count(*) FROM j").fetchone()[0]
    log(f"标签⋈特征 行数={n_j:,}")
    # 物化到 numpy 友好 parquet
    joined = MID / "分析样本_train_val.parquet"
    con.execute(f"COPY (SELECT * FROM j) TO '{joined}' (FORMAT PARQUET)")
    log(f"分析样本已写 {joined} md5={md5(joined)}")

    df = pd.read_parquet(joined)
    # 若 join 产生 航线_1 等后缀，映射回代表名
    for c in list(feat_cols):
        if c not in df.columns:
            alt = c + "_1"
            if alt in df.columns:
                df[c] = df[alt]
    # 数值化：类别特征（航司/航线等）按训练集频率编码；其余 to_numeric
    is_tr_mask = df["观察日"].isin(TRAIN_DAYS)
    cat_enc = {}
    for c in feat_cols:
        if c not in df.columns:
            raise SystemExit(f"缺少代表列: {c}")
        s = df[c]
        as_str = s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "str"
        # Decimal / 数字字符串
        if as_str or str(s.dtype) == "object":
            nn = pd.to_numeric(s, errors="coerce")
            # 若大部分能转数字，当数值；否则当类别
            frac = float(nn.notna().mean()) if len(s) else 0.0
            if frac >= 0.9:
                df[c] = nn
            else:
                # 类别：用训练期出现的值编码
                tr_vals = df.loc[is_tr_mask, c].astype(str).fillna("缺失")
                freq = tr_vals.value_counts()
                mp = {k: i for i, k in enumerate(freq.index)}
                cat_enc[c] = mp
                df[c] = df[c].astype(str).fillna("缺失").map(mp).fillna(-1)
        else:
            df[c] = pd.to_numeric(s, errors="coerce")
    df[feat_cols] = df[feat_cols].astype(np.float32)
    json.dump({k: len(v) for k, v in cat_enc.items()}, open(MID / "类别编码规模.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if cat_enc:
        log(f"类别编码列: {list(cat_enc.keys())}")

    is_tr = df["观察日"].isin(TRAIN_DAYS).to_numpy()
    is_va = df["观察日"].isin(VAL_DAYS).to_numpy()
    y = df["标签_7天降到线"].to_numpy(np.int8)
    save = df["降价收益_若收单"].to_numpy(np.float64)
    loss = df["涨价亏损_若收单"].to_numpy(np.float64)
    days_va = df.loc[is_va, "观察日"].astype(str).to_numpy()

    Xtr = df.loc[is_tr, feat_cols]
    ytr = y[is_tr]
    Xva = df.loc[is_va, feat_cols]
    yva = y[is_va]
    save_va = save[is_va]
    loss_va = loss[is_va]

    # 丢掉全空/零方差代表（仍记入明细为建议剔除）
    keep_feats = []
    drop_zero = []
    for c in feat_cols:
        s = Xtr[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            drop_zero.append(c)
        else:
            keep_feats.append(c)
    log(f"可用代表={len(keep_feats)} 零方差/全空剔除={len(drop_zero)}: {drop_zero}")

    # -------- 基线模型 --------
    log("训练基线 LightGBM（仅训练日）…")
    dtr = lgb.Dataset(Xtr[keep_feats], ytr, feature_name=keep_feats, free_raw_data=False)
    dva = lgb.Dataset(Xva[keep_feats], yva, feature_name=keep_feats, free_raw_data=False)
    booster = lgb.train(
        PARAMS, dtr, num_boost_round=ROUNDS, valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(50)]
    )
    booster.save_model(str(MID / "baseline_lgbm.txt"))
    gain = pd.Series(booster.feature_importance(importance_type="gain"), index=keep_feats, name="gain")
    split_imp = pd.Series(booster.feature_importance(importance_type="split"), index=keep_feats, name="split")

    score_va = booster.predict(Xva[keep_feats])
    # 按天取前X% 选峰（读法B主报）
    day_codes = pd.factorize(days_va)[0]
    curve = []
    best = None
    for x in XS:
        k1, k2, n = net_at_x(score_va, save_va, loss_va, x, day_index=day_codes)
        row = {"出手比例": x, "条数": n, "每单净省钱_k1": k1, "每单净省钱_k2": k2}
        curve.append(row)
        if n >= MIN_N and (best is None or (k2 is not None and k2 > best["每单净省钱_k2"])):
            best = row
    if best is None:
        best = max(curve, key=lambda r: (r["每单净省钱_k2"] if r["每单净省钱_k2"]==r["每单净省钱_k2"] else -1e18, r["每单净省钱_k1"] or -1e18))
    freeze_x = best["出手比例"]
    base_k1, base_k2 = best["每单净省钱_k1"], best["每单净省钱_k2"]
    pd.DataFrame(curve).to_csv(CHK / "基线_VAL出手比例曲线.csv", index=False, encoding="utf-8-sig")
    json.dump({"冻结出手比例": freeze_x, "基线每单k1": base_k1, "基线每单k2": base_k2,
               "early_stopping_best_iteration": booster.best_iteration},
              open(MID / "基线选参.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"基线 VAL 冻结X={freeze_x} k1={base_k1:.3f} k2={base_k2:.3f} best_iter={booster.best_iteration}")

    # -------- 置换重要性（同义组=代表列）--------
    log("置换重要性（按同义组代表，VAL，Δ净省钱）…")
    rng = np.random.default_rng(SEED)
    Xva_np = Xva[keep_feats].to_numpy(np.float32, copy=True)
    feat_index = {c: i for i, c in enumerate(keep_feats)}
    perm_rows = []
    # 也做 leave-one-group-out 分数：用 gain 近似太弱；真正 LOGO 重训太慢——对 gain 前 40 + 风险组做 LOGO
    for i, c in enumerate(keep_feats):
        col = Xva_np[:, feat_index[c]].copy()
        shuf = col.copy()
        rng.shuffle(shuf)
        Xva_np[:, feat_index[c]] = shuf
        sc = booster.predict(Xva_np)
        k1, k2, n = net_at_x(sc, save_va, loss_va, freeze_x, day_index=day_codes)
        Xva_np[:, feat_index[c]] = col  # restore
        perm_rows.append({
            "代表特征": c,
            "置换后_每单净省钱_k1": k1,
            "置换后_每单净省钱_k2": k2,
            "Δ净省钱_k1": base_k1 - k1,  # 正=有用（打乱后变差）
            "Δ净省钱_k2": base_k2 - k2,
            "条数": n,
        })
        if (i + 1) % 20 == 0:
            log(f"  perm {i+1}/{len(keep_feats)}")
            pd.DataFrame(perm_rows).to_csv(MID / "perm_partial.csv", index=False, encoding="utf-8-sig")
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(MID / "perm_full.csv", index=False, encoding="utf-8-sig")

    # -------- 单特征有用性（仅该特征训小模型）--------
    log("单特征短训（每代表 80 轮）…")
    uni_rows = []
    # 随机分基线
    rnd = rng.random(len(yva))
    rk1, rk2, _ = net_at_x(rnd, save_va, loss_va, freeze_x, day_index=day_codes)
    for i, c in enumerate(keep_feats):
        dtr_u = lgb.Dataset(Xtr[[c]], ytr, feature_name=[c], free_raw_data=True)
        b = lgb.train({**PARAMS, "feature_fraction": 1.0}, dtr_u, num_boost_round=80)
        sc = b.predict(Xva[[c]])
        k1, k2, n = net_at_x(sc, save_va, loss_va, freeze_x, day_index=day_codes)
        # 与标签 AUC
        try:
            from sklearn.metrics import roc_auc_score
            # 处理常数
            if len(np.unique(yva)) > 1 and np.nanstd(sc) > 0:
                auc = float(roc_auc_score(yva, sc))
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")
        uni_rows.append({
            "代表特征": c,
            "单特征_每单净省钱_k1": k1,
            "单特征_每单净省钱_k2": k2,
            "单特征_相对随机_Δk1": k1 - rk1,
            "单特征_相对随机_Δk2": k2 - rk2,
            "单特征_AUC": auc,
            "条数": n,
        })
        if (i + 1) % 30 == 0:
            log(f"  uni {i+1}/{len(keep_feats)}")
            pd.DataFrame(uni_rows).to_csv(MID / "uni_partial.csv", index=False, encoding="utf-8-sig")
    uni = pd.DataFrame(uni_rows)
    uni.to_csv(MID / "uni_full.csv", index=False, encoding="utf-8-sig")
    json.dump({"随机基线_k1": rk1, "随机基线_k2": rk2, "冻结X": freeze_x},
              open(MID / "随机基线.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # -------- Leave-one-group-out（增益 Top40 + 全部风险组，限时）--------
    log("Leave-one-group-out 重训（增益前40 + 风险组）…")
    top_gain = list(gain.sort_values(ascending=False).head(40).index)
    risk_feats = []
    for _, r in groups.iterrows():
        tags = risk_tags(r["代表特征"], r)
        if any(t.startswith("跨期") or t.startswith("春运") or t.startswith("季节") for t in tags):
            if r["代表特征"] in keep_feats:
                risk_feats.append(r["代表特征"])
    logo_targets = list(dict.fromkeys(top_gain + risk_feats))
    logo_rows = []
    for i, drop_c in enumerate(logo_targets):
        cols = [c for c in keep_feats if c != drop_c]
        dtr_l = lgb.Dataset(Xtr[cols], ytr, feature_name=cols, free_raw_data=True)
        dva_l = lgb.Dataset(Xva[cols], yva, feature_name=cols, free_raw_data=True)
        b = lgb.train(PARAMS, dtr_l, num_boost_round=min(ROUNDS, booster.best_iteration or ROUNDS),
                      valid_sets=[dva_l], valid_names=["val"],
                      callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        sc = b.predict(Xva[cols])
        k1, k2, n = net_at_x(sc, save_va, loss_va, freeze_x, day_index=day_codes)
        logo_rows.append({
            "代表特征": drop_c,
            "LOGO后_每单净省钱_k1": k1,
            "LOGO后_每单净省钱_k2": k2,
            "LOGO_Δk1": base_k1 - k1,
            "LOGO_Δk2": base_k2 - k2,
            "条数": n,
        })
        if (i + 1) % 10 == 0:
            log(f"  logo {i+1}/{len(logo_targets)}")
            pd.DataFrame(logo_rows).to_csv(MID / "logo_partial.csv", index=False, encoding="utf-8-sig")
    logo = pd.DataFrame(logo_rows)
    logo.to_csv(MID / "logo_full.csv", index=False, encoding="utf-8-sig")

    # -------- 年份-季节不一致性（航管特征：与 dep_month 相关 + 置换已覆盖）--------
    log("年份/季节风险附加分…")
    year_rows = []
    if "dep_month" in df.columns or "dep_month" in feat_cols:
        dm = df.loc[is_va, "dep_month"] if "dep_month" in df.columns else Xva["dep_month"] if "dep_month" in Xva.columns else None
    else:
        dm = None
    # 从特征矩阵再取 dep_month（它是特征之一）
    if "dep_month" in keep_feats:
        dm = Xva["dep_month"]
    for c in keep_feats:
        tags = []
        # 与月份相关
        if dm is not None and Xva[c].notna().sum() > 100:
            try:
                corr = float(pd.Series(Xva[c].to_numpy()).corr(pd.Series(dm.to_numpy())))
            except Exception:
                corr = float("nan")
        else:
            corr = float("nan")
        # 主表月份只有 10/11/12，暑运/春运恒 0 → 标记
        nuniq = int(Xva[c].nunique(dropna=True))
        year_rows.append({"代表特征": c, "与dep_month相关": corr, "VAL唯一值数": nuniq})
    year_df = pd.DataFrame(year_rows)

    # -------- 汇总明细 --------
    log("汇总名单…")
    detail = groups.copy()
    detail = detail.merge(gain.reset_index().rename(columns={"index": "代表特征"}), on="代表特征", how="left")
    detail = detail.merge(split_imp.reset_index().rename(columns={"index": "代表特征"}), on="代表特征", how="left")
    detail = detail.merge(perm, on="代表特征", how="left")
    detail = detail.merge(uni, on="代表特征", how="left")
    detail = detail.merge(logo, on="代表特征", how="left")
    detail = detail.merge(year_df, on="代表特征", how="left")

    def decide(row):
        name = row["代表特征"]
        tags = risk_tags(name, row)
        # 零覆盖 / 全空
        if name in drop_zero or (pd.notna(row.get("覆盖率")) and row["覆盖率"] == 0):
            return "建议剔除", tags + (["零覆盖"] if "零覆盖" not in tags else []), "覆盖率为0或训练期零方差，无信息"
        dk2 = row.get("Δ净省钱_k2")
        uk2 = row.get("单特征_相对随机_Δk2")
        lk2 = row.get("LOGO_Δk2")
        g_ = row.get("gain")
        reason_parts = []
        # 春运：样本学不到
        if "春运外推" in tags:
            return "建议剔除", tags, "样本无春运（dep_is_chunyun恒0），无法学习，默认不入模"
        # 强风险 + 弱增益
        useful_signal = False
        if pd.notna(dk2) and dk2 > 0.5:  # 打乱后每单少赚 >0.5 元
            useful_signal = True
            reason_parts.append(f"置换Δk2={dk2:.2f}")
        if pd.notna(uk2) and uk2 > 1.0:
            useful_signal = True
            reason_parts.append(f"单特征相对随机Δk2={uk2:.2f}")
        if pd.notna(lk2) and lk2 > 0.5:
            useful_signal = True
            reason_parts.append(f"LOGOΔk2={lk2:.2f}")
        if pd.notna(g_) and g_ > 0 and gain.max() > 0 and g_ >= gain.quantile(0.7):
            useful_signal = True
            reason_parts.append(f"增益分位高({g_:.1f})")

        weak = (pd.isna(dk2) or dk2 <= 0) and (pd.isna(uk2) or uk2 <= 0) and (pd.isna(g_) or g_ <= 0)

        if "跨期过拟合_航管2025" in tags or "季节_年份绑死风险" in tags:
            # 需要更强证据才入选；否则观察或剔除
            if useful_signal and pd.notna(dk2) and dk2 > 1.0 and (pd.isna(lk2) or lk2 > 0):
                return "观察", tags, "有净省钱信号但依赖2025/季节，跨期风险高，建议观察不默认入模；" + "；".join(reason_parts)
            if weak:
                return "建议剔除", tags, "依赖2025历史或季节且净省钱增量弱/不稳，建议先拿掉"
            return "观察", tags, "有季节/航管年份风险，信号一般；" + ("；".join(reason_parts) if reason_parts else "增量有限")

        if "低覆盖" in tags and not useful_signal:
            return "建议剔除", tags, "覆盖率低且无稳定净省钱增量"

        if useful_signal and (pd.notna(dk2) and dk2 > 0.3) and (pd.isna(uk2) or uk2 >= -0.5):
            return "建议入选", tags, "有助于净省钱（k1/k2）：" + "；".join(reason_parts)
        if useful_signal:
            return "观察", tags, "有一定信号但不稳或偏弱：" + "；".join(reason_parts)
        if weak:
            return "建议剔除", tags, "置换/单特征/增益均未显示稳定净省钱增量"
        return "观察", tags, "信号边缘，留观察"

    decisions, tag_lists, reasons = [], [], []
    for _, row in detail.iterrows():
        d, t, r = decide(row)
        decisions.append(d)
        tag_lists.append("|".join(t) if t else "")
        reasons.append(r)
    detail["建议档"] = decisions
    detail["风险标签"] = tag_lists
    detail["入选理由"] = reasons

    # 综合分：主要用于排序展示
    detail["综合有用分_k2"] = (
        detail["Δ净省钱_k2"].fillna(0) * 0.5
        + detail["单特征_相对随机_Δk2"].fillna(0) * 0.3
        + detail["LOGO_Δk2"].fillna(0) * 0.2
    )
    detail = detail.sort_values(["建议档", "综合有用分_k2", "Δ净省钱_k2"],
                                ascending=[True, False, False],
                                key=lambda s: s.map({"建议入选": 0, "观察": 1, "建议剔除": 2}) if s.name == "建议档" else s)

    # 输出列中文
    out_detail = detail.rename(columns={
        "覆盖率": "覆盖率",
        "是否需航管历史": "是否需航管历史",
        "gain": "模型增益重要性",
        "split": "模型分裂次数",
        "Δ净省钱_k1": "置换Δ每单净省钱_k1",
        "Δ净省钱_k2": "置换Δ每单净省钱_k2",
        "单特征_相对随机_Δk1": "单特征相对随机Δk1",
        "单特征_相对随机_Δk2": "单特征相对随机Δk2",
        "单特征_AUC": "单特征AUC",
        "LOGO_Δk1": "去组重训Δk1",
        "LOGO_Δk2": "去组重训Δk2",
        "与dep_month相关": "与起飞月相关",
    })
    cols_detail = [
        "同义组号", "代表特征", "成员特征", "成员数", "建议档", "入选理由", "风险标签",
        "覆盖率", "是否需航管历史", "综合有用分_k2",
        "置换Δ每单净省钱_k1", "置换Δ每单净省钱_k2",
        "单特征相对随机Δk1", "单特征相对随机Δk2", "单特征AUC",
        "去组重训Δk1", "去组重训Δk2",
        "模型增益重要性", "模型分裂次数",
        "与起飞月相关", "VAL唯一值数",
        "置换后_每单净省钱_k1", "置换后_每单净省钱_k2",
        "单特征_每单净省钱_k1", "单特征_每单净省钱_k2",
    ]
    cols_detail = [c for c in cols_detail if c in out_detail.columns]
    out_detail[cols_detail].to_csv(OUT / "逐特征有用性明细.csv", index=False, encoding="utf-8-sig")

    short = out_detail[["同义组号", "代表特征", "成员特征", "建议档", "入选理由", "风险标签",
                        "置换Δ每单净省钱_k1", "置换Δ每单净省钱_k2",
                        "单特征相对随机Δk1", "单特征相对随机Δk2",
                        "去组重训Δk2", "覆盖率", "是否需航管历史", "综合有用分_k2"]].copy()
    # 只保留存在的列
    short = short[[c for c in short.columns if c in out_detail.columns or c in short.columns]]
    short.to_csv(OUT / "有用特征名单.csv", index=False, encoding="utf-8-sig")

    vc = short["建议档"].value_counts().to_dict()
    log(f"档位计数: {vc}")

    # -------- 文档 --------
    top_in = short[short["建议档"] == "建议入选"].head(25)
    top_watch = short[short["建议档"] == "观察"].head(20)
    top_drop = short[short["建议档"] == "建议剔除"]

    risk_md = f"""# 风险标签说明（S2）

- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai
- 授权：Yangyang t1350u 开工；闸门 t1353u：有用名单须本人过目后才能开 S3
- 本文件只解释风险标签含义与本窗证据，**不是过门、不是上线**

## 标签定义

| 标签 | 含义 | 本窗证据 |
|---|---|---|
| 春运外推 | 特征依赖春运，但主表起飞日无春运 | `dep_is_chunyun` 恒为 0；`g1_days_to_spring_festival` 仅 67..113（距春运很远） |
| 跨期过拟合_航管2025 | 特征主要来自航管 2025 扩样本 | S1 覆盖表「是否需航管历史=是」共 27 列；主表观察日全在 2026-10-15..11-01 |
| 季节_年份绑死风险 | 起飞月/暑运/节假日相位等，易把「年份」学成「季节」 | 主表 `dep_month` 仅 10/11/12；`dep_is_shuyun` 恒 0 |
| 零覆盖 | 特征全空 | 见名单 |
| 低覆盖 | 非空率 &lt; 50% | 见名单 |

## 评估口径（防误解）

1. 主评估只用 **全航司** 主表，**不把航管行混进评估闸门**。
2. 时间切分（训练期内）：训练观察日 {TRAIN_DAYS}；验证 {VAL_DAYS}；实际 gap={(pd.Timestamp(VAL_DAYS[0]) - pd.Timestamp(TRAIN_DAYS[-1])).days - 1}（要求 ≥7，已满足）。
3. `--exclude-from={EXCLUDE_FROM}`：该日及以后不入本分析；满窗日本就止于 10-25。
4. 读法B：只保留 `距起飞天数>={DTD_MIN_B}`。
5. 有用性主尺子：验证集上按天取前 X% 的 **每单净省钱 k1/k2**（X 在验证集选峰后冻结为 {freeze_x}）。
6. **从未用测试集挑特征**；本 S2 无单独测试集报告（正式训练 S3 另批后才切测试）。

## 方法

- 同义组 172：每组只用代表特征计 1 次。
- 置换重要性：打乱该代表列后，基线模型分数重算，Δ净省钱 = 基线 − 打乱后（正值=有用）。
- 单特征短训：只用该列训 80 轮，相对随机分的 Δ净省钱。
- Leave-one-group-out：对增益前 40 + 风险组，去掉该列重训后的 Δ净省钱。

## 基线参考（验证集，离线口径，不是真实盈亏）

- 冻结出手比例 X={freeze_x}
- 基线每单净省钱 k1={base_k1:.4f}，k2={base_k2:.4f}
- 随机分基线 k1={rk1:.4f}，k2={rk2:.4f}
"""
    (OUT / "风险标签说明.md").write_text(risk_md, encoding="utf-8")

    # 一页纸
    def fmt_top(df_, n=12):
        lines = []
        for _, r in df_.head(n).iterrows():
            dk2 = r.get("置换Δ每单净省钱_k2")
            uk2 = r.get("单特征相对随机Δk2")
            lines.append(
                f"- **{r['代表特征']}**：档={r['建议档']}；置换Δk2={dk2 if pd.notna(dk2) else '—'}；"
                f"单特征Δk2={uk2 if pd.notna(uk2) else '—'}；风险={r['风险标签'] or '无'}；{r['入选理由']}"
            )
        return "\n".join(lines) if lines else "（无）"

    page = f"""# 给 Yangyang 的一页纸 · S2 有用特征（大白话）

> 时间：{time.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai  
> 目的：按你的闸门（t1353u），训练前先让你过目「有用特征」短名单。  
> **你点头之前不开 S3 训练。** 数字都是离线模拟净省钱，不是真收。

## 我们怎么衡量「有用」

- 只看能不能 **多净省钱**（k1 / k2，涨价亏损要扣）。
- 做法：在较早的两天（10-15、10-16）上拟合一个临时模型，拿到较晚两天（10-24、10-25）上按天取前 {int(freeze_x*100) if freeze_x<=1 else freeze_x}% 名单，看每单净省钱。
- 中间空了 7 天以上，避免标签窗重叠（gap 已满足）。
- 同名/同义特征合并成 **172 组，一组只算一次**。

## 建议先留下的（能多赚）

共 **{vc.get('建议入选', 0)}** 组。前几名：

{fmt_top(top_in, 15)}

## 建议先拿掉的

共 **{vc.get('建议剔除', 0)}** 组。主要三类：

1. **春运相关**：这个窗口根本没有春运，学了也是瞎编 → 默认剔除。  
2. **零覆盖 / 几乎没值**：例如 argmin_*、n_post、n_stale、route_recent_dep_level。  
3. **靠 2025 航管硬扛季节、但净省钱增量又弱** 的：容易把「去年」当成「旺季」。

剔除清单节选：

{fmt_top(top_drop, 15)}

## 观察（先别进默认，也不急着删）

共 **{vc.get('观察', 0)}** 组。多半是：有一点用，但 **像把年份当成季节**，或依赖航管 2025。你若坚持要试，S3 可做「加/不加」对照，不默认入模。

{fmt_top(top_watch, 12)}

## 最大风险（请务必看）

1. **没有春运**：春运特征现在学不到；将来春运临近要么禁用，要么另开窗重评。  
2. **起飞月只有 10–12 月，暑运全是 0**：季节类特征的变异很多来自 2025 扩样本，跨年可能翻车。  
3. **满窗观察日只有 11 天**，再挖掉 gap，真正用来选特征的更少 → 名单偏「这一小段窗口」，不是全年真理。  
4. 航管扩样本 **不进评估闸门**；主尺子始终只看全航司。

## 请你拍板

- [ ] 同意「建议入选」进 S3 默认候选  
- [ ] 观察组：默认不进 / 或指定某几个进「加/不加」对照  
- [ ] 建议剔除：确认拿掉（尤其春运 + 零覆盖）  
- [ ] 其他要加减的特征：________

你回复「开始」或点名调整后，再开 S3。
"""
    (OUT / "给Yangyang的一页纸.md").write_text(page, encoding="utf-8")

    # 说明
    explain = f"""# 11-02 新模型重训 · S2 有用性分析说明

- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai
- 授权：t1350u 开工；**t1353u 闸门：有用名单交 Yangyang 过目后才能 S3**
- 红线：未碰线上推理服务 / [REDACTED_SERVICE]；未写服务器库；未读 ana_extract/pull8001；未打印 store.json；**未做正式训练/apply/上线/真收**；主评估未混航管行

## 1. 输入

| 项 | 路径 / 值 |
|---|---|
| 特征矩阵 | `S1_特征/数据/特征矩阵_全航司_dtd0_30.parquet` md5 `a7394ec84b551f3f236729732f7ae305` |
| 去重对照 | `S1_特征/核对/去重对照表.csv`（172 同义组） |
| 底表价格 | `S0_底表/数据/主表_全航司日价_dtd0_30.parquet` |
| 评估尺子 | clean_eval v1.5 净省钱定义；读法B；gap≥7；exclude-from={EXCLUDE_FROM} |

## 2. 标签

- 降到线：窗口内首次价 ≤ 基准价−{DROP_YUAN} 或 ≤ 基准价×{1-DROP_RATIO}
- 窗长 {W} 天；满窗=观察日+{W}≤数据截止日 {CUTOFF}
- 净省钱_k = 降价收益 − k×涨价亏损（未降时亏损=max(窗末价−基准价,0)）
- 子集：满窗 ∧ dtd≥{DTD_MIN_B} ∧ 观察日&lt;{EXCLUDE_FROM}
- 标签文件：`数据/标签_全航司_满窗.parquet`

## 3. 切分（只在训练期内选特征）

| 集 | 观察日 | 说明 |
|---|---|---|
| 训练 | {', '.join(TRAIN_DAYS)} | 拟合临时模型 |
| 验证 | {', '.join(VAL_DAYS)} | 置换/单特征/LOGO 与选 X |
| gap | {(pd.Timestamp(VAL_DAYS[0]) - pd.Timestamp(TRAIN_DAYS[-1])).days - 1} | ≥7 已满足 |
| 测试 | （本 S2 不切、不用） | 防泄漏 |

## 4. 产出计数

| 建议档 | 组数 |
|---|---|
| 建议入选 | {vc.get('建议入选', 0)} |
| 观察 | {vc.get('观察', 0)} |
| 建议剔除 | {vc.get('建议剔除', 0)} |
| 合计同义组 | {len(short)} |

基线验证集：冻结 X={freeze_x}，每单 k1={base_k1:.4f}，k2={base_k2:.4f}（离线口径，不是真实盈亏）。

## 5. 产出文件

- `有用特征名单.csv`
- `逐特征有用性明细.csv`
- `风险标签说明.md`
- `给Yangyang的一页纸.md`
- `md5清单.txt`
- 本说明

## 6. 下一步

**停止于 S2。** 将「有用特征名单」+一页纸交 Grok 转 Yangyang；点头前不开 S3。
"""
    (OUT / "有用性分析说明.md").write_text(explain, encoding="utf-8")

    # md5 清单
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "md5清单.txt" and "脚本/__pycache__" not in str(p) and p.suffix != ".pyc":
            if "/中间/" in str(p) or p.parent.name == "中间":
                # 中间大文件也列
                pass
            rel = str(p.relative_to(OUT))
            if rel.startswith("日志/"):
                continue
            files.append(rel)
    # regenerate list properly
    lines = [f"# S2 md5清单", f"# 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai", ""]
    for p in sorted(OUT.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "md5清单.txt":
            continue
        if "__pycache__" in str(p) or p.suffix == ".pyc":
            continue
        rel = str(p.relative_to(OUT))
        if rel.startswith("日志/"):
            continue
        lines.append(f"{md5(p)}  {rel}")
    (OUT / "md5清单.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "建议入选": int(vc.get("建议入选", 0)),
        "观察": int(vc.get("观察", 0)),
        "建议剔除": int(vc.get("建议剔除", 0)),
        "冻结X": freeze_x,
        "基线k1": base_k1,
        "基线k2": base_k2,
        "gap": (pd.Timestamp(VAL_DAYS[0]) - pd.Timestamp(TRAIN_DAYS[-1])).days - 1,
        "耗时秒": round(time.time() - t0),
        "top入选": top_in[["代表特征", "置换Δ每单净省钱_k2", "单特征相对随机Δk2"]].head(15).to_dict(orient="records"),
    }
    json.dump(summary, open(MID / "汇总.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    log(f"=== S2 完成 耗时{summary['耗时秒']}s 入选={summary['建议入选']} 观察={summary['观察']} 剔除={summary['建议剔除']} ===")


if __name__ == "__main__":
    main()
