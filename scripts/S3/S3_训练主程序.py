#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 正式离线训练（11-02 重训）
闸门：Yangyang t1360u 经 Grok 解除（建议入选38默认；剔除24；观察里跨期/去年类自行加不加对照）。
红线：不碰线上推理服务、不上线、不真收、不写服务器库。评估过门只用全航司。
"""
from __future__ import annotations
import os, sys, json, time, hashlib, gc, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

PY = sys.executable
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
S0_MAIN = ROOT / "S0_底表/数据/主表_全航司日价_dtd0_30.parquet"
S0_EXP = ROOT / "S0_底表/数据/扩样本表_航管2025与2026新_对齐_dtd0_30_含子来源.parquet"
S0_DTD3150 = ROOT / "S0_底表/数据/对照表_全航司日价_dtd31_50.parquet"
S0_HC = ROOT / "S0_底表/数据/回补表_全航司日价_另存.parquet"
S0_TL5 = ROOT / "S0_底表/数据/对照表_timeline_2026五家.parquet"
S1 = ROOT / "S1_特征/数据/特征矩阵_全航司_dtd0_30.parquet"
S2_LIST = ROOT / "S2_有用性/有用特征名单.csv"
S2_LAB = ROOT / "S2_有用性/数据/标签_全航司_满窗.parquet"
OUT = ROOT / "S3_训练"
DATA, MODEL, MET, CURVE, CHK, LOG, MID, SCR = (
    OUT / "数据", OUT / "模型", OUT / "指标", OUT / "曲线",
    OUT / "核对", OUT / "日志", OUT / "中间", OUT / "脚本",
)
for d in (DATA, MODEL, MET, CURVE, CHK, LOG, MID):
    d.mkdir(parents=True, exist_ok=True)

EXCLUDE_FROM = "2026-10-28"
W = 7
DROP_YUAN, DROP_RATIO = 100.0, 0.15
DTD_MIN_B = 14
# 多日切分：训15-17 / 验25（gap=7）；测试另报24作冻结参考（不参与选阈）
TRAIN_DAYS = ["2026-10-15", "2026-10-16", "2026-10-17"]
VAL_DAYS = ["2026-10-25"]
TEST_DAYS = ["2026-10-24"]  # 冻结参考；阈值只在 VAL 选
XS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
MIN_N = 500
PARAMS = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.05,
    num_leaves=63, feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
    min_data_in_leaf=200, verbosity=-1, n_jobs=2, seed=42,
)
ROUNDS, EARLY, SEED = 600, 40, 42
# 航管扩样本：每子来源最多抽这么多训练行（内存友好）
EXP_SAMPLE_CAP = 800_000
EXP_WEIGHT = 0.35  # 航管样本权重（降权）；MF/SC 再乘 0.5


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


def dump_json(obj, path):
    def _c(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=_c)


def load_feature_plan():
    df = pd.read_csv(S2_LIST)
    keep = df[df["建议档"] == "建议入选"]["代表特征"].tolist()
    drop = df[df["建议档"] == "建议剔除"]["代表特征"].tolist()
    obs = df[df["建议档"] == "观察"].copy()
    # 跨期/去年/航管历史类 → 加不加对照候选
    mask_ly = (
        obs["代表特征"].astype(str).str.contains(
            r"ly_|prev_vs_ly|train_bin|tr_p|dest_avg_vol|dest_drop|orig_avg|orig_drop|route_recent|tiaoxiu|g1_|holiday|hol_|hd_dep|days_to_hol|days_from_hol|is_holiday|dep_month|dep_is_|shuyun|golden_week|season",
            case=False, na=False, regex=True,
        )
        | obs["风险标签"].astype(str).str.contains("跨期|春运|过拟合|航管|去年|年份|季节", na=False)
        | obs["是否需航管历史"].astype(str).eq("是")
    )
    ly_cand = obs.loc[mask_ly].sort_values("综合有用分_k2", ascending=False)
    # 冷启动观察：默认不入
    cold = obs.loc[~mask_ly & obs["风险标签"].astype(str).str.contains("冷启动|零覆盖", na=False)]
    other_obs = obs.loc[~mask_ly & ~obs.index.isin(cold.index)]
    plan = {
        "默认38": keep,
        "剔除24": drop,
        "跨期对照候选": ly_cand["代表特征"].tolist(),
        "跨期对照明细": ly_cand[["代表特征", "风险标签", "是否需航管历史", "综合有用分_k2", "覆盖率"]].to_dict("records"),
        "冷启动观察_默认不入": cold["代表特征"].tolist(),
        "其他观察_默认不入": other_obs["代表特征"].tolist(),
    }
    dump_json(plan, MID / "特征计划.json")
    ly_cand.to_csv(CHK / "跨期对照候选.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"代表特征": keep, "来源": "建议入选"}).to_csv(CHK / "默认38.csv", index=False, encoding="utf-8-sig")
    return plan


def ensure_labels():
    """复用 S2 标签；校验 md5/行数。"""
    assert S2_LAB.exists(), "缺少 S2 标签"
    st = json.load(open(ROOT / "S2_有用性/中间/标签统计.json"))
    lab_md5 = md5(S2_LAB)
    info = {
        "路径": str(S2_LAB),
        "md5": lab_md5,
        "S2统计md5": st.get("标签md5"),
        "一致": lab_md5 == st.get("标签md5"),
        "行数": st["行数"],
        "exclude_from": EXCLUDE_FROM,
        "读法B": DTD_MIN_B,
    }
    dump_json(info, MID / "标签复用核对.json")
    log(f"标签复用: 行={st['行数']:,} md5={lab_md5[:8]} 一致={info['一致']}")
    return S2_LAB


def build_join_table(feat_cols, days, out_parquet):
    """标签 ⋈ 特征，只取给定观察日、dtd>=14。"""
    if out_parquet.exists():
        log(f"已存在联结表，跳过: {out_parquet.name}")
        return out_parquet
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='10GB'")
    feat_sql = ", ".join(f'"{c}"' for c in feat_cols)
    days_sql = ", ".join(repr(d) for d in days)
    con.execute(f"""
    CREATE TABLE lab AS
    SELECT obs_day AS 观察日, flight_no AS 航班号, route AS 航线, dep_date AS 起飞日,
           dtd AS 距起飞天数_标签, p0 AS 基准价,
           标签_7天降到线, 降价收益_若收单, 涨价亏损_若收单, 每单净省钱_k1, 每单净省钱_k2,
           是否回补键
    FROM read_parquet('{S2_LAB}')
    WHERE obs_day IN ({days_sql})
      AND dtd >= {DTD_MIN_B}
    """)
    con.execute(f"""
    CREATE TABLE feat AS
    SELECT 观察日, 航班号, 航线, 起飞日, 航司, 距起飞天数, {feat_sql}
    FROM read_parquet('{S1}')
    WHERE 观察日 IN ({days_sql})
      AND 距起飞天数 >= {DTD_MIN_B}
    """)
    feat_sel = ", ".join([f'f."{c}"' for c in feat_cols])
    con.execute(f"""
    COPY (
      SELECT l.观察日, l.航班号, l.航线, l.起飞日, l.距起飞天数_标签, l.基准价,
             l.标签_7天降到线, l.降价收益_若收单, l.涨价亏损_若收单, l.每单净省钱_k1, l.每单净省钱_k2,
             l.是否回补键, f.航司, f.距起飞天数, {feat_sel}
      FROM lab l
      INNER JOIN feat f
        USING (观察日, 航班号, 航线, 起飞日)
    ) TO '{out_parquet}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out_parquet}')").fetchone()[0]
    log(f"联结表 {out_parquet.name}: {n:,} 行, 特征={len(feat_cols)}")
    con.close()
    return out_parquet


def make_X(df, feats):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            # 类别：简单因子化（按出现序）；未知→-1
            X[c] = pd.Categorical(X[c].astype(str)).codes.astype(np.int32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(np.float32)


def pick_thr(y, score, save, loss, xs=XS, min_n=MIN_N):
    order = np.argsort(-score)
    y, score, save, loss = y[order], score[order], save[order], loss[order]
    n = len(y)
    rows, best = [], None
    for x in xs:
        k = max(1, int(round(n * x)))
        s1 = float((save[:k] - loss[:k]).mean())
        s2 = float((save[:k] - 2 * loss[:k]).mean())
        row = {
            "出手比例": x, "条数": int(k), "阈值": float(score[k - 1]),
            "每单净省钱_k1": s1, "每单净省钱_k2": s2,
            "精度": float(y[:k].mean()),
            "召回": float(y[:k].sum() / max(y.sum(), 1)),
        }
        rows.append(row)
        if k >= min_n and (best is None or s2 > best["每单净省钱_k2"] or (
                abs(s2 - best["每单净省钱_k2"]) < 1e-12 and s1 > best["每单净省钱_k1"])):
            best = row
    if best is None:
        ok = [r for r in rows if r["条数"] >= min_n]
        best = max(ok or rows, key=lambda r: (r["每单净省钱_k2"], r["每单净省钱_k1"]))
    return best, rows


def net_at_mask(df, mask):
    sub = df.loc[mask]
    n = len(sub)
    if n == 0:
        return {"条数": 0, "每单净省钱_k1": np.nan, "每单净省钱_k2": np.nan, "精度": np.nan, "召回": np.nan}
    y = sub["标签_7天降到线"].to_numpy(float)
    save = sub["降价收益_若收单"].to_numpy(float)
    loss = sub["涨价亏损_若收单"].to_numpy(float)
    all_pos = (df["标签_7天降到线"] == 1).sum()
    return {
        "条数": int(n),
        "每单净省钱_k1": float((save - loss).mean()),
        "每单净省钱_k2": float((save - 2 * loss).mean()),
        "净省钱合计_k1": float((save - loss).sum()),
        "净省钱合计_k2": float((save - 2 * loss).sum()),
        "精度": float(y.mean()),
        "召回": float(y.sum() / all_pos) if all_pos else np.nan,
        "正例率_全集": float(df["标签_7天降到线"].mean()),
    }


def topx_metrics(df, score, x):
    """按天取前 x%。"""
    days = df["观察日"].to_numpy()
    sel = []
    for d in np.unique(days):
        m = np.where(days == d)[0]
        k = max(1, int(round(len(m) * x)))
        order = m[np.argsort(-score[m])[:k]]
        sel.append(order)
    idx = np.concatenate(sel) if sel else np.array([], dtype=int)
    mask = np.zeros(len(df), dtype=bool)
    mask[idx] = True
    return net_at_mask(df, mask)


def train_eval(df_tr, df_va, df_te, feats, tag, sample_weight_tr=None):
    log(f"训练 {tag}: 特征={len(feats)} 训={len(df_tr):,} 验={len(df_va):,} 测={len(df_te):,}")
    Xtr = make_X(df_tr, feats)
    Xva = make_X(df_va, feats)
    Xte = make_X(df_te, feats)
    ytr = df_tr["标签_7天降到线"].to_numpy(np.int8)
    yva = df_va["标签_7天降到线"].to_numpy(np.int8)
    yte = df_te["标签_7天降到线"].to_numpy(np.int8)
    dtr = lgb.Dataset(Xtr, ytr, feature_name=list(feats), weight=sample_weight_tr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        {**PARAMS, "seed": SEED}, dtr, num_boost_round=ROUNDS,
        valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY, verbose=False)],
    )
    bi = int(booster.best_iteration or booster.num_trees())
    mp = MODEL / f"model_{tag}_seed{SEED}.txt"
    booster.save_model(str(mp), num_iteration=bi)
    pv = booster.predict(Xva, num_iteration=bi)
    pt = booster.predict(Xte, num_iteration=bi)
    ptr = booster.predict(Xtr, num_iteration=bi)
    save_v = df_va["降价收益_若收单"].to_numpy(float)
    loss_v = df_va["涨价亏损_若收单"].to_numpy(float)
    thr_info, curve = pick_thr(yva, pv, save_v, loss_v)
    pd.DataFrame(curve).to_csv(CURVE / f"净省钱曲线_VAL_{tag}.csv", index=False, encoding="utf-8-sig")
    # 测试曲线（阈值不重选，只扫出手比例）
    te_curve = []
    for x in XS:
        te_curve.append({"出手比例": x, **{k: v for k, v in topx_metrics(df_te, pt, x).items()}})
    pd.DataFrame(te_curve).to_csv(CURVE / f"净省钱曲线_TEST_{tag}.csv", index=False, encoding="utf-8-sig")
    thr = thr_info["阈值"]
    # 主报告：VAL 所选阈值 → TEST；同时报 VAL 上 topX 峰
    mask_te = pt >= thr
    mask_va = pv >= thr
    # 也报 VAL 峰出手比例对应的 TEST topX
    x_star = thr_info["出手比例"]
    test_topx = topx_metrics(df_te, pt, x_star)
    val_topx = topx_metrics(df_va, pv, x_star)
    auc_v = float(roc_auc_score(yva, pv)) if yva.min() != yva.max() else float("nan")
    auc_t = float(roc_auc_score(yte, pt)) if yte.min() != yte.max() else float("nan")
    imp = pd.DataFrame({
        "特征": feats,
        "增益": booster.feature_importance(importance_type="gain"),
        "分裂": booster.feature_importance(importance_type="split"),
    }).sort_values("增益", ascending=False)
    imp.to_csv(MET / f"特征重要性_{tag}.csv", index=False, encoding="utf-8-sig")
    # 分数落地（测试）
    sc = df_te[["观察日", "航班号", "航线", "起飞日", "距起飞天数", "标签_7天降到线",
                 "降价收益_若收单", "涨价亏损_若收单"]].copy()
    sc["分数"] = pt
    sc["是否出手_阈值"] = mask_te.astype(int)
    sc.to_parquet(DATA / f"测试打分_{tag}.parquet", index=False)
    report = {
        "标签": tag,
        "特征数": len(feats),
        "特征": feats,
        "best_iteration": bi,
        "模型路径": str(mp),
        "模型md5": md5(mp),
        "训练行": int(len(df_tr)), "验证行": int(len(df_va)), "测试行": int(len(df_te)),
        "验证AUC": auc_v, "测试AUC": auc_t,
        "验证选参": thr_info,
        "验证_阈值出手": net_at_mask(df_va, mask_va),
        "验证_峰出手比例topX": val_topx,
        "测试_阈值出手": net_at_mask(df_te, mask_te),
        "测试_同比例topX": test_topx,
        "测试_全收单k1": float(df_te["每单净省钱_k1"].mean()),
        "测试_全收单k2": float(df_te["每单净省钱_k2"].mean()),
    }
    dump_json(report, MET / f"报告_{tag}.json")
    # 瘦身
    del Xtr, Xva, Xte, dtr, dva
    gc.collect()
    return report, booster, thr_info


def run_ly_contrast(df_all, base_feats, ly_cands):
    """在全航司上：默认38 vs +跨期特征（分批/整体）。"""
    days_tr = set(TRAIN_DAYS)
    days_va = set(VAL_DAYS)
    days_te = set(TEST_DAYS)
    tr = df_all[df_all["观察日"].isin(days_tr)]
    va = df_all[df_all["观察日"].isin(days_va)]
    te = df_all[df_all["观察日"].isin(days_te)]
    rows = []
    # 基线
    base_rep, _, _ = train_eval(tr, va, te, base_feats, "A0_默认38")
    rows.append({
        "方案": "A0_默认38", "加特征": "",
        "验证_峰k1": base_rep["验证_峰出手比例topX"]["每单净省钱_k1"],
        "验证_峰k2": base_rep["验证_峰出手比例topX"]["每单净省钱_k2"],
        "测试_同比例k1": base_rep["测试_同比例topX"]["每单净省钱_k1"],
        "测试_同比例k2": base_rep["测试_同比例topX"]["每单净省钱_k2"],
        "测试AUC": base_rep["测试AUC"],
        "决策": "基线",
    })
    base_k2 = rows[0]["验证_峰k2"]

    # 单加：只测综合分>0 且覆盖>=0.5 的前 15 个，控制时长
    ly_df = pd.read_csv(CHK / "跨期对照候选.csv")
    ly_df = ly_df[(ly_df["综合有用分_k2"] > 0) & (ly_df["覆盖率"].fillna(0) >= 0.5)].head(15)
    promote = []
    for _, r in ly_df.iterrows():
        f = r["代表特征"]
        if f in base_feats:
            continue
        feats = base_feats + [f]
        tag = f"A_ly单加_{f}"[:80]
        try:
            rep, _, _ = train_eval(tr, va, te, feats, tag)
            dk2 = rep["验证_峰出手比例topX"]["每单净省钱_k2"] - base_k2
            decision = "并入" if dk2 >= 1.0 else "不加"
            if decision == "并入":
                promote.append(f)
            rows.append({
                "方案": tag, "加特征": f,
                "验证_峰k1": rep["验证_峰出手比例topX"]["每单净省钱_k1"],
                "验证_峰k2": rep["验证_峰出手比例topX"]["每单净省钱_k2"],
                "测试_同比例k1": rep["测试_同比例topX"]["每单净省钱_k1"],
                "测试_同比例k2": rep["测试_同比例topX"]["每单净省钱_k2"],
                "测试AUC": rep["测试AUC"],
                "Δ验证k2_相对基线": dk2,
                "决策": decision,
            })
            log(f"ly单加 {f}: Δk2={dk2:+.2f} → {decision}")
        except Exception as e:
            log(f"ly单加失败 {f}: {e}")
            rows.append({"方案": tag, "加特征": f, "决策": f"失败:{e}", "Δ验证k2_相对基线": np.nan})

    # 打包：一次加上全部 promote
    if promote:
        feats = base_feats + promote
        rep, _, _ = train_eval(tr, va, te, feats, "A1_默认38加晋升ly")
        rows.append({
            "方案": "A1_默认38加晋升ly", "加特征": "|".join(promote),
            "验证_峰k1": rep["验证_峰出手比例topX"]["每单净省钱_k1"],
            "验证_峰k2": rep["验证_峰出手比例topX"]["每单净省钱_k2"],
            "测试_同比例k1": rep["测试_同比例topX"]["每单净省钱_k1"],
            "测试_同比例k2": rep["测试_同比例topX"]["每单净省钱_k2"],
            "测试AUC": rep["测试AUC"],
            "Δ验证k2_相对基线": rep["验证_峰出手比例topX"]["每单净省钱_k2"] - base_k2,
            "决策": "A版默认特征集",
        })
        final_feats = feats
        final_tag = "A1_默认38加晋升ly"
        final_rep = rep
    else:
        final_feats = base_feats
        final_tag = "A0_默认38"
        final_rep = base_rep

    # 负分跨期打包「全加」对照（预期不加）
    neg = ly_df  # already positive only; also try top-8 as bundle without selection
    bundle = [f for f in ly_df["代表特征"].tolist()[:8] if f not in base_feats]
    if bundle:
        try:
            rep, _, _ = train_eval(tr, va, te, base_feats + bundle, "A_ly打包前8")
            rows.append({
                "方案": "A_ly打包前8", "加特征": "|".join(bundle),
                "验证_峰k1": rep["验证_峰出手比例topX"]["每单净省钱_k1"],
                "验证_峰k2": rep["验证_峰出手比例topX"]["每单净省钱_k2"],
                "测试_同比例k1": rep["测试_同比例topX"]["每单净省钱_k1"],
                "测试_同比例k2": rep["测试_同比例topX"]["每单净省钱_k2"],
                "测试AUC": rep["测试AUC"],
                "Δ验证k2_相对基线": rep["验证_峰出手比例topX"]["每单净省钱_k2"] - base_k2,
                "决策": "对照_不默认并入" if rep["验证_峰出手比例topX"]["每单净省钱_k2"] < base_k2 + 1 else "对照_整体有用但以单加晋升为准",
            })
        except Exception as e:
            log(f"ly打包失败: {e}")

    tab = pd.DataFrame(rows)
    tab.to_csv(MET / "ly加不加对照表.csv", index=False, encoding="utf-8-sig")
    dump_json({"晋升特征": promote, "最终A特征": final_feats, "最终A标签": final_tag}, MID / "A版定案.json")
    return final_feats, final_tag, final_rep, promote


def build_expand_relative_train(cap=EXP_SAMPLE_CAP):
    """从航管扩样本构造轻量相对特征+标签，仅作训练扩样本；评估不用。
    口径：相对自身历史；剔 MF/SC 或降权；剔价格异常；只要 dtd>=14；子来源先 2026新 再尝试加 2025。
    """
    out_2026 = DATA / "扩样本训练_航管2026新_相对特征.parquet"
    out_2025 = DATA / "扩样本训练_航管2025_相对特征.parquet"
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='10GB'")

    def _build(sub_filter_sql, outp, seed_offset=0):
        if outp.exists():
            log(f"扩样本已存在: {outp.name}")
            return outp
        log(f"构建扩样本相对特征 → {outp.name}")
        # 价格路径 + 7 天窗标签（与主口径同降到线）
        con.execute(f"""
        CREATE OR REPLACE TABLE exp_raw AS
        SELECT 观察日, 航班号, 航线, 起飞日, 航司, 距起飞天数 AS dtd, 价格 AS price,
               子来源, 数据源, 价格异常标记
        FROM read_parquet('{S0_EXP}')
        WHERE {sub_filter_sql}
          AND 距起飞天数 >= {DTD_MIN_B} AND 距起飞天数 <= 30
          AND 价格 > 0
          AND (价格异常标记 IS NULL OR 价格异常标记 = '' OR 价格异常标记 = '否')
          AND 航司 NOT IN ('MF', 'SC')
        """)
        n0 = con.execute("SELECT count(*) FROM exp_raw").fetchone()[0]
        log(f"  原始扩样本行={n0:,}")
        # 自身历史相对特征
        con.execute("""
        CREATE OR REPLACE TABLE exp_feat AS
        SELECT *,
          lag(price, 1) OVER w AS price_prev,
          min(price) OVER (PARTITION BY 航班号, 航线, 起飞日 ORDER BY 观察日
                           ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS roll7_min,
          avg(price) OVER (PARTITION BY 航班号, 航线, 起飞日 ORDER BY 观察日
                           ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS roll7_avg,
          min(price) OVER (PARTITION BY 航班号, 航线, 起飞日 ORDER BY 观察日
                           ROWS BETWEEN 13 PRECEDING AND 1 PRECEDING) AS roll14_min
        FROM exp_raw
        WINDOW w AS (PARTITION BY 航班号, 航线, 起飞日 ORDER BY 观察日)
        """)
        con.execute("""
        CREATE OR REPLACE TABLE exp_feat2 AS
        SELECT *,
          CASE WHEN price_prev > 0 THEN price / price_prev ELSE NULL END AS x_over_prev,
          CASE WHEN roll7_min > 0 THEN price / roll7_min ELSE NULL END AS x_over_roll7min,
          CASE WHEN roll7_avg > 0 THEN price / roll7_avg ELSE NULL END AS x_over_roll7avg,
          CASE WHEN roll14_min > 0 THEN price / roll14_min ELSE NULL END AS x_over_roll14min,
          CASE WHEN price_prev IS NOT NULL THEN price - price_prev ELSE NULL END AS 上一观察价差_元_扩
        FROM exp_feat
        """)
        # 标签
        con.execute(f"""
        CREATE OR REPLACE TABLE exp_lab AS
        WITH base AS (
          SELECT 观察日 AS obs_day, 航班号 AS flight_no, 航线 AS route, 起飞日 AS dep_date,
                 price AS p0, dtd, 子来源, 数据源, 航司,
                 x_over_prev, x_over_roll7min, x_over_roll7avg, x_over_roll14min, 上一观察价差_元_扩,
                 price_prev, roll7_min, roll7_avg
          FROM exp_feat2
          WHERE price_prev IS NOT NULL
        ),
        j AS (
          SELECT b.*, p.观察日 AS bd, p.price AS bp,
                 date_diff('day', CAST(b.obs_day AS DATE), CAST(p.观察日 AS DATE)) AS h
          FROM base b
          JOIN exp_raw p
            ON p.航班号=b.flight_no AND p.航线=b.route AND p.起飞日=b.dep_date
           AND p.观察日 > b.obs_day
           AND CAST(p.观察日 AS DATE) <= CAST(b.obs_day AS DATE) + INTERVAL {W} DAY
        ),
        g AS (
          SELECT obs_day, flight_no, route, dep_date, p0, dtd, 子来源, 数据源, 航司,
                 x_over_prev, x_over_roll7min, x_over_roll7avg, x_over_roll14min, 上一观察价差_元_扩,
                 count(*) AS n_win,
                 arg_min(bp, bd) FILTER (WHERE bp <= p0 - {DROP_YUAN} OR bp <= p0 * {1-DROP_RATIO}) AS p_drop,
                 arg_max(bp, bd) AS p_end
          FROM j GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14
        )
        SELECT *,
          CASE WHEN p_drop IS NOT NULL THEN 1 ELSE 0 END AS 标签_7天降到线,
          CASE WHEN p_drop IS NOT NULL THEN p0 - p_drop ELSE 0 END AS 降价收益_若收单,
          CASE WHEN p_drop IS NULL THEN greatest(p_end - p0, 0) ELSE 0 END AS 涨价亏损_若收单
        FROM g
        WHERE n_win > 0
        """)
        n1 = con.execute("SELECT count(*) FROM exp_lab").fetchone()[0]
        log(f"  满窗标签行={n1:,}")
        # 随机抽样封顶
        con.execute(f"""
        COPY (
          SELECT obs_day AS 观察日, flight_no AS 航班号, route AS 航线, dep_date AS 起飞日,
                 dtd AS 距起飞天数, p0 AS 基准价, 航司, 子来源, 数据源,
                 x_over_prev, x_over_roll7min, x_over_roll7avg, x_over_roll14min, 上一观察价差_元_扩,
                 标签_7天降到线, 降价收益_若收单, 涨价亏损_若收单,
                 降价收益_若收单 - 涨价亏损_若收单 AS 每单净省钱_k1,
                 降价收益_若收单 - 2*涨价亏损_若收单 AS 每单净省钱_k2
          FROM exp_lab
          USING SAMPLE {cap}
        ) TO '{outp}' (FORMAT PARQUET)
        """)
        n2 = con.execute(f"SELECT count(*) FROM read_parquet('{outp}')").fetchone()[0]
        log(f"  写出扩样本 {n2:,} → {outp}")
        return outp

    p2026 = _build("子来源 = '航管2026_03..09(39家)'", out_2026, 0)
    # 2025 两段合并抽
    p2025 = _build("子来源 LIKE '%2025%'", out_2025, 1)
    con.close()
    return p2026, p2025


def run_B(df_af, a_feats, p2026, p2025):
    """B版：全航司训行 + 航管相对特征扩样本（特征交集用占位对齐）。
    因扩样本只有相对价特征，B 用「共享相对特征子集 + 来源id」训练，评估仍全航司全特征 A 集。
    更稳妥做法：扩样本只增强「相对价」头，与 A 特征并集时对扩样本缺失列填 NaN。
    """
    rel_feats = ["x_over_prev", "x_over_roll7min", "x_over_roll7avg", "x_over_roll14min", "上一观察价差_元_扩", "距起飞天数"]
    # 映射：全航司侧找近似列
    af_map = {
        "x_over_prev": "上一观察价差_元",  # 没有比值就用价差；另造
        "上一观察价差_元_扩": "上一观察价差_元",
        "距起飞天数": "距起飞天数",
    }
    # 在全航司上构造同名相对列
    def add_rel_af(df):
        out = df.copy()
        # 用已有列近似
        if "log_price_prev" in out.columns:
            out["x_over_prev"] = np.nan  # 占位
        # prev_over_min_day 等已在 A 中；为扩样本对齐构造：
        out["x_over_prev"] = np.where(out.get("log_price_prev") is not None, np.nan, np.nan)
        if "上一观察价差_元" in out.columns:
            out["上一观察价差_元_扩"] = out["上一观察价差_元"]
        # 用 prev_over_min_day / roll 近似
        for src, dst in [
            ("prev_over_min_day", "x_over_roll7min"),
            ("roll7_p20", "x_over_roll7avg"),
            ("roll14_p80", "x_over_roll14min"),
        ]:
            if src in out.columns:
                out[dst] = out[src]
        if "上一观察价差_元" in out.columns and out["上一观察价差_元"].notna().any():
            # 粗比值：若有基准价
            if "基准价" in out.columns:
                out["x_over_prev"] = 1.0 + out["上一观察价差_元"] / out["基准价"].replace(0, np.nan)
        out["来源id"] = 0  # 全航司
        out["样本权重"] = 1.0
        return out

    tr_af = add_rel_af(df_af[df_af["观察日"].isin(TRAIN_DAYS)])
    va_af = add_rel_af(df_af[df_af["观察日"].isin(VAL_DAYS)])
    te_af = add_rel_af(df_af[df_af["观察日"].isin(TEST_DAYS)])

    results = []

    def _load_exp(path, src_id, weight):
        e = pd.read_parquet(path)
        e["来源id"] = src_id
        e["样本权重"] = weight
        # 统一列名
        rename = {}
        if "距起飞天数" not in e.columns and "dtd" in e.columns:
            rename["dtd"] = "距起飞天数"
        e = e.rename(columns=rename)
        return e

    # B1: 全航司 + 2026新
    exp26 = _load_exp(p2026, 1, EXP_WEIGHT)
    # 训练特征 = A特征在扩样本上多数缺失→对扩样本只用 rel + 来源id，对全航司用 A+rel+来源id
    # 简化：B 模型特征 = a_feats 中存在于两侧可对齐的 + rel_feats + 来源id
    # 扩样本缺 A 特征 → 填 NaN（LGBM 可处理）
    def align(df, feats_all):
        out = df.copy()
        for c in feats_all:
            if c not in out.columns:
                out[c] = np.nan
        return out

    feats_B = list(dict.fromkeys(a_feats + rel_feats + ["来源id"]))
    tr_b1 = align(pd.concat([tr_af, exp26], ignore_index=True), feats_B)
    w_b1 = tr_b1["样本权重"].to_numpy(np.float32)
    va_b = align(va_af, feats_B)
    te_b = align(te_af, feats_B)
    rep_b1, _, _ = train_eval(tr_b1, va_b, te_b, feats_B, "B1_全航司加航管2026新", sample_weight_tr=w_b1)
    results.append(("B1_全航司加航管2026新", rep_b1))

    # B2: +2025
    exp25 = _load_exp(p2025, 2, EXP_WEIGHT * 0.8)
    tr_b2 = align(pd.concat([tr_af, exp26, exp25], ignore_index=True), feats_B)
    w_b2 = tr_b2["样本权重"].to_numpy(np.float32)
    rep_b2, _, _ = train_eval(tr_b2, va_b, te_b, feats_B, "B2_全航司加航管2026新与2025", sample_weight_tr=w_b2)
    results.append(("B2_全航司加航管2026新与2025", rep_b2))

    # 对照表
    rows = []
    for name, rep in results:
        rows.append({
            "方案": name,
            "验证_峰k1": rep["验证_峰出手比例topX"]["每单净省钱_k1"],
            "验证_峰k2": rep["验证_峰出手比例topX"]["每单净省钱_k2"],
            "测试_同比例k1": rep["测试_同比例topX"]["每单净省钱_k1"],
            "测试_同比例k2": rep["测试_同比例topX"]["每单净省钱_k2"],
            "测试AUC": rep["测试AUC"],
            "训练行": rep["训练行"],
        })
    pd.DataFrame(rows).to_csv(MET / "B版扩样本对照表.csv", index=False, encoding="utf-8-sig")
    return results


def offline_dtd3150_note():
    """dtd31-50 对照：主训特征矩阵无 31-50（S1 只 dtd0-30）。只记账说明。"""
    con = duckdb.connect()
    n = con.execute(f"SELECT count(*), min(观察日), max(观察日), min(距起飞天数), max(距起飞天数) FROM read_parquet('{S0_DTD3150}')").fetchone()
    note = {
        "说明": "S1 特征矩阵仅 dtd0-30，本轮不对 dtd31-50 重打特征/重训；仅记录底表规模，后续若要对照需另批构造特征。",
        "对照表行数": n[0], "观察日": f"{n[1]}..{n[2]}", "dtd": f"{n[3]}..{n[4]}",
    }
    dump_json(note, CHK / "dtd31_50对照说明.json")
    con.close()
    return note


def write_docs(plan, a_feats, a_tag, a_rep, promote, b_results):
    # 最终特征清单
    fin = pd.DataFrame({
        "代表特征": a_feats,
        "来源": ["建议入选38" if f in plan["默认38"] else ("ly晋升" if f in promote else "其他") for f in a_feats],
    })
    fin.to_csv(CHK / "最终入模特征清单.csv", index=False, encoding="utf-8-sig")

    # 一页纸
    def _fmt(rep):
        if not rep:
            return "无"
        v = rep["验证_峰出手比例topX"]
        t = rep["测试_同比例topX"]
        return (f"验证峰 出手{v.get('条数')}条 k1={v.get('每单净省钱_k1'):.2f} k2={v.get('每单净省钱_k2'):.2f}；"
                f"测试同比例 k1={t.get('每单净省钱_k1'):.2f} k2={t.get('每单净省钱_k2'):.2f} AUC={rep.get('测试AUC'):.4f}")

    b1 = next((r for n, r in b_results if n.startswith("B1")), None)
    b2 = next((r for n, r in b_results if n.startswith("B2")), None)
    # 选 B：验证 k2 更高者
    best_b = None
    if b_results:
        best_b = max(b_results, key=lambda x: x[1]["验证_峰出手比例topX"]["每单净省钱_k2"])

    page = f"""# 11-02 重训 S3 · 给 Yangyang 的一页纸（大白话）

- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（Asia/Shanghai）
- 授权：t1360u 特征定案后开 S3；红线未碰（（线上推理服务） / 上线 / 真收）
- 口径：全航司末批价、读法B（出手 dtd≥14）、gap≥7、exclude-from=2026-10-28、净省钱 k1/k2 主尺子
- 切分：训练 10-15..10-17；验证 10-25（只在这里选出手比例）；测试参考 10-24（阈值不重选）

## 先说结论

1. **A版（全航司）**：默认 38 个有用特征；跨期/去年类自己做了「加/不加」，晋升进默认的有：{('、'.join(promote) if promote else '无（都不够稳）')}。
2. **能不能多赚**：看验证/测试的每单净省钱。A版：{_fmt(a_rep)}。
3. **B版（加航管扩样本）**：按 S0 提醒只加相对变化、航管降权、评估仍只用全航司。B1（+2026新）：{_fmt(b1)}。B2（再+2025）：{_fmt(b2)}。
4. **亏在哪**：全收单本身 k2 为负（涨价惩罚重）；模型靠「只推前 X%」把钱从平均亏里捞回来。若测试 k2 仍明显差于验证，就是分布外推/过拟合风险。
5. **A vs B**：谁验证峰 k2 更高用谁做离线候选——当前更好的是 **{(best_b[0] if best_b else a_tag)}**。通过≠上线≠真收。
6. **冷启动观察特征**（特征B等）默认不进；dtd31–50 本轮只记账不重训（缺 S1 特征）。

## 数字摘要（主）

| 方案 | 验证峰k2 | 测试同比例k2 | 测试AUC |
|---|---:|---:|---:|
| {a_tag} | {a_rep['验证_峰出手比例topX']['每单净省钱_k2']:.2f} | {a_rep['测试_同比例topX']['每单净省钱_k2']:.2f} | {a_rep['测试AUC']:.4f} |
"""
    if b1:
        page += f"| B1_加2026新 | {b1['验证_峰出手比例topX']['每单净省钱_k2']:.2f} | {b1['测试_同比例topX']['每单净省钱_k2']:.2f} | {b1['测试AUC']:.4f} |\n"
    if b2:
        page += f"| B2_再加2025 | {b2['验证_峰出手比例topX']['每单净省钱_k2']:.2f} | {b2['测试_同比例topX']['每单净省钱_k2']:.2f} | {b2['测试AUC']:.4f} |\n"
    page += """
详细曲线见 `曲线/`，对照表见 `指标/ly加不加对照表.csv`、`指标/B版扩样本对照表.csv`。
"""
    (OUT / "大白话一页纸.md").write_text(page, encoding="utf-8")

    # 说明.md
    sm = f"""# S3 训练说明

## 授权与红线
- Yangyang t1360u：建议入选 38 进默认；剔除 24；观察里去年/航管/跨期自行加不加。
- 未碰线上推理服务、未写服务器库、未上线、未真收。评估过门数据 = 全航司。

## 输入 md5（前 8 位）
- S1 特征矩阵：{md5(S1)[:8]}
- S2 名单：{md5(S2_LIST)[:8]}
- S2 标签：{md5(S2_LAB)[:8]}

## 切分
- 训练观察日：{TRAIN_DAYS}
- 验证观察日：{VAL_DAYS}（选出手比例 / 阈值）
- 测试参考观察日：{TEST_DAYS}（不重选阈值）
- gap（训末→验初）= {(pd.Timestamp(VAL_DAYS[0]) - pd.Timestamp(TRAIN_DAYS[-1])).days - 1} 天（要求≥7）
- exclude-from={EXCLUDE_FROM}；读法B dtd≥{DTD_MIN_B}；窗={W}；降到线 = −100元 或 ×0.85

## 特征
- 默认 38：见 核对/默认38.csv
- 剔除 24：不入模
- 跨期对照：见 指标/ly加不加对照表.csv；晋升={promote}
- 冷启动观察：默认不入（{plan['冷启动观察_默认不入'][:8]}…）
- 最终 A 特征数={len(a_feats)}，清单 核对/最终入模特征清单.csv

## A / B
- A：`{a_tag}` 模型 md5={a_rep['模型md5'][:8]}
- B1/B2：航管相对特征扩样本，权重={EXP_WEIGHT}，剔 MF/SC，评估全航司

## 限制
- dtd31–50、回补、五家 tl2026：本轮仅说明/未做完整特征重训（见 核对/dtd31_50对照说明.json）
- 航管与全航司无重叠观察日，B 版扩的是样本多样性不是同日拼接绝对价
"""
    (OUT / "说明.md").write_text(sm, encoding="utf-8")


def write_md5_list():
    lines = ["文件\tmd5\t字节\n"]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "md5清单.txt" and "脚本" not in p.parts:
            # include scripts too actually
            pass
    lines = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "md5清单.txt":
            continue
        rel = str(p.relative_to(OUT))
        lines.append(f"{md5(p)}\t{p.stat().st_size}\t{rel}")
    text = "md5\t字节\t相对路径\n" + "\n".join(lines) + "\n"
    (OUT / "md5清单.txt").write_text(text, encoding="utf-8")
    log(f"md5清单: {len(lines)} 项, 清单自身 md5={md5(OUT / 'md5清单.txt')}")


def main():
    t0 = time.time()
    (LOG / "运行日志.txt").write_text("", encoding="utf-8")
    log("=== S3 训练开始 ===")
    plan = load_feature_plan()
    ensure_labels()
    base = plan["默认38"]
    # 跨期候选也要进联结表
    ly_all = plan["跨期对照候选"]
    all_feats_needed = list(dict.fromkeys(base + ly_all + ["上一观察价差_元", "log_price_prev", "prev_over_min_day", "roll7_p20", "roll14_p80"]))
    days = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
    join_path = DATA / "联结_全航司_训验测.parquet"
    build_join_table(all_feats_needed, days, join_path)
    df = pd.read_parquet(join_path)
    log(f"联结加载: {len(df):,} 行, 日={sorted(df['观察日'].unique().tolist())}")
    # 断言无回补（S2 标签回补行=0）
    if "是否回补键" in df.columns:
        log(f"回补行={((df['是否回补键']=='是').sum())}")

    a_feats, a_tag, a_rep, promote = run_ly_contrast(df, base, ly_all)

    offline_dtd3150_note()

    # B 扩样本
    try:
        p2026, p2025 = build_expand_relative_train()
        b_results = run_B(df, a_feats, p2026, p2025)
    except Exception as e:
        log(f"B版失败: {traceback.format_exc()}")
        b_results = []
        (MID / "B版错误.txt").write_text(traceback.format_exc(), encoding="utf-8")

    write_docs(plan, a_feats, a_tag, a_rep, promote, b_results)
    write_md5_list()
    log(f"=== S3 完成 用时 {(time.time()-t0)/60:.1f} 分钟 ===")
    # 摘要
    summary = {
        "A标签": a_tag,
        "A特征数": len(a_feats),
        "晋升ly": promote,
        "A验证峰k2": a_rep["验证_峰出手比例topX"]["每单净省钱_k2"],
        "A测试同比例k2": a_rep["测试_同比例topX"]["每单净省钱_k2"],
        "B方案数": len(b_results),
        "目录": str(OUT),
        "说明md5": md5(OUT / "说明.md"),
        "一页纸md5": md5(OUT / "大白话一页纸.md"),
        "清单md5": md5(OUT / "md5清单.txt"),
    }
    dump_json(summary, MID / "完成摘要.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
