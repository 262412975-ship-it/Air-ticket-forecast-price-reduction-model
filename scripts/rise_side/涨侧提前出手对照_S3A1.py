#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 A1 之上的「涨侧/提前出手」离线对照。
仅离线；不碰线上推理服务 / 不上线 / 不真收 / 不写服务器库 / 不重启 [REDACTED_SERVICE]。
通过≠上线。seats 不入模。未读 ana_extract/pull8001/store.json。

授权：Yangyang t1380u（经经理 Grok 转达）问涨侧/提前出手可否做、做完看对现有模型有无作用。
规则/路径结算逻辑对齐 v4b；降模型=S3 A1；涨模型优先在 A1 特征集上新训。
入场池两档：尖端约0.5%；合计利润最大附近（约前33%，VAL 上按净省钱合计选）。
"""
import os, sys, json, time, hashlib, gc
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import lightgbm as lgb
import duckdb
from sklearn.metrics import roc_auc_score

ROOT = "${PROJECT_ROOT}"
OUT = os.path.join(ROOT, "涨侧提前出手对照_20261102")
S0 = os.path.join(ROOT, "S0_底表/数据/主表_全航司日价_dtd0_30.parquet")
S1 = os.path.join(ROOT, "S1_特征/数据/特征矩阵_全航司_dtd0_30.parquet")
S2_LAB = os.path.join(ROOT, "S2_有用性/数据/标签_全航司_满窗.parquet")
S3 = os.path.join(ROOT, "S3_训练")
DROP_MODEL = os.path.join(S3, "模型/model_A1_默认38加晋升ly_seed42.txt")
A_JSON = os.path.join(S3, "中间/A版定案.json")
JOINED = os.path.join(S3, "数据/联结_全航司_训验测.parquet")

TRAIN_DAYS = ["2026-10-15", "2026-10-16", "2026-10-17"]
VAL_DAYS = ["2026-10-25"]
TEST_DAYS = ["2026-10-24"]
EXCLUDE_FROM = "2026-10-28"
CUTOFF = "2026-11-01"
SWITCH = "2026-10-29"
DTD_MIN = 14  # 读法B 主切点
W = 7
DROP_YUAN, DROP_RATIO = 100.0, 0.15
RISE_YUAN, RISE_RATIO = 100.0, 0.15
SEED = 42
PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1, n_jobs=1)
ROUNDS, EARLY = 800, 40
# θ 网格（v4b）
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
CTRL_N = [1, 2, 3, 4]
TIE_YUAN, SUPPORT_K2_GAIN = 1.0, 5.0
# 入场比例扫描（含尖端与约前33%）
XS_POOL = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.60]
SETTLE_MIN = 7


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(OUT, "结果/运行日志.txt"), "a", encoding="utf-8") as f:
        f.write(s + "\n")


def make_X(df, feats):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = pd.Categorical(X[c].astype(str)).codes.astype(np.int32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(np.float32)


def kstr(df):
    return (df["航班号"].astype(str) + "|" + df["航线"].astype(str) + "|" + df["起飞日"].astype(str))


def build_paths(E, PR):
    """E: idx,键,t,p0；PR: 键,d,p,s,dd。返回 P,S,V,D 矩阵 n×W。"""
    m = E[["idx", "键", "t", "p0"]].merge(PR[["键", "d", "p", "s", "dd"]], on="键", how="inner")
    h = (m["d"] - m["t"]).dt.days
    m = m.loc[(h >= 1) & (h <= W)].sort_values(["idx", "d"], kind="stable")
    pos = m.groupby("idx").cumcount().to_numpy()
    n = len(E)
    P = np.full((n, W), np.nan)
    S = np.full((n, W), np.nan)
    V = np.zeros((n, W), bool)
    D = np.full((n, W), np.nan)
    if len(m):
        ii = m["idx"].to_numpy()
        P[ii, pos] = m["p"].to_numpy()
        S[ii, pos] = m["s"].to_numpy()
        V[ii, pos] = True
        D[ii, pos] = m["dd"].to_numpy()
    return P, S, V, D


def outcome(P, S, V, p0, theta):
    n = len(p0)
    ar = np.arange(n)
    H = V & ((P <= p0[:, None] - DROP_YUAN) | (P <= p0[:, None] * (1.0 - DROP_RATIO)))
    ev = H | (V & (S >= theta)) if np.isfinite(theta) else H
    has = ev.any(1)
    j = ev.argmax(1)
    hit = has & H[ar, j]
    early = has & ~hit
    pj = P[ar, j]
    last = V.sum(1) - 1
    pend = P[ar, np.maximum(last, 0)]
    gain = np.where(hit, p0 - pj, 0.0)
    loss = np.where(early, np.maximum(pj - p0, 0), np.where(hit, 0.0, np.maximum(pend - p0, 0)))
    sidx = np.where(has, j, np.maximum(last, 0))
    return gain, loss, early, hit, j, sidx


def mets(gain, loss, early, msk=None, sd=None):
    if msk is not None:
        gain, loss, early = gain[msk], loss[msk], early[msk]
        if sd is not None:
            sd = sd[msk]
    n = len(gain)
    if n == 0:
        return {"条数": 0}
    ex_ = {} if sd is None else {
        "结算时距起飞最少天数": float(np.nanmin(sd)),
        "结算时距起飞不足7天占比": float(np.mean(sd < SETTLE_MIN)),
    }
    return {
        **ex_,
        "条数": int(n),
        "每单净省钱_k1": float((gain - loss).mean()),
        "每单净省钱_k2": float((gain - 2 * loss).mean()),
        "净省钱合计_k1": float((gain - loss).sum()),
        "净省钱合计_k2": float((gain - 2 * loss).sum()),
        "提前出占比": float(early.mean()),
        "平均涨价亏损_元": float(loss.mean()),
        "平均降价收益_元": float(gain.mean()),
    }


def schemes(Pm, Sm, Vm):
    out_ = [("不提前出(基线)", "不提前出", Sm, np.inf)]
    for th in GRID:
        out_.append(("涨侧提前出", th, Sm, th))
    for N in CTRL_N:
        SN = np.where(np.arange(W)[None, :] == N - 1, 1.0, -1.0) * np.ones((Sm.shape[0], 1))
        out_.append((f"对照:无模型_入场后第{N}个观察日必出", "—", SN, 0.5))
    U = np.random.default_rng(SEED).random(Sm.shape)
    for th in GRID:
        q = float((Sm >= th)[Vm].mean()) if Vm.any() else 0.0
        out_.append(("对照:无模型_随机提前出(与该门槛同每日提前出概率)", th, U, 1 - q))
    return out_


def pool_curve(y, score, save, loss, xs):
    o = np.argsort(-score, kind="stable")
    y, score, save, loss = y[o], score[o], save[o], loss[o]
    n = len(y)
    rows = []
    for x in xs:
        k = max(1, int(round(n * x)))
        s1 = float((save[:k] - loss[:k]).mean())
        s2 = float((save[:k] - 2 * loss[:k]).mean())
        rows.append({
            "出手比例": x, "条数": int(k), "阈值": float(score[k - 1]),
            "每单净省钱_k1": s1, "每单净省钱_k2": s2,
            "净省钱合计_k1": float((save[:k] - loss[:k]).sum()),
            "净省钱合计_k2": float((save[:k] - 2 * loss[:k]).sum()),
            "精度": float(y[:k].mean()),
        })
    return rows


def main():
    os.makedirs(os.path.join(OUT, "结果"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "模型"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "中间"), exist_ok=True)
    open(os.path.join(OUT, "结果/运行日志.txt"), "w", encoding="utf-8").write("")
    T0 = time.time()
    feats = json.load(open(A_JSON, encoding="utf-8"))["最终A特征"]
    assert len(feats) == 46
    drop_md5 = md5(DROP_MODEL)
    assert drop_md5.startswith("f4a120a9"), drop_md5
    log("降模型 A1 md5", drop_md5, "特征数", len(feats))

    con = duckdb.connect()
    # ---------- 1) 涨到线标签（与 v2.1 对称：价>=p0+100 或 >=p0*1.15）----------
    rise_lab_path = os.path.join(OUT, "中间/标签_涨到线_满窗.parquet")
    if not os.path.exists(rise_lab_path):
        log("构建涨到线标签…")
        con.execute(f"""
        COPY (
          WITH base AS (
            SELECT CAST(观察日 AS VARCHAR) AS obs_day, 航班号 AS flight_no, 航线 AS route,
                   CAST(起飞日 AS VARCHAR) AS dep_date, CAST(价格 AS DOUBLE) AS p0,
                   距起飞天数 AS dtd, 是否回补键
            FROM read_parquet('{S0}')
            WHERE 是否满7天窗口='是' AND 观察日 < '{EXCLUDE_FROM}'
              AND 距起飞天数 >= {DTD_MIN} AND 距起飞天数 <= 30
              AND length(CAST(航班号 AS VARCHAR)) <> 7
          ),
          fut AS (
            SELECT b.*, p.观察日 AS bd, CAST(p.价格 AS DOUBLE) AS bp,
                   date_diff('day', CAST(b.obs_day AS DATE), CAST(p.观察日 AS DATE)) AS h
            FROM base b
            JOIN read_parquet('{S0}') p
              ON p.航班号 = b.flight_no AND p.航线 = b.route AND CAST(p.起飞日 AS VARCHAR) = b.dep_date
             AND CAST(p.观察日 AS VARCHAR) > b.obs_day
             AND CAST(p.观察日 AS DATE) <= CAST(b.obs_day AS DATE) + INTERVAL {W} DAY
             AND length(CAST(p.航班号 AS VARCHAR)) <> 7
          ),
          agg AS (
            SELECT obs_day, flight_no, route, dep_date, p0, dtd, 是否回补键,
                   count(*) AS n_win,
                   bool_or(bp >= p0 + {RISE_YUAN} OR bp >= p0 * {1.0 + RISE_RATIO}) AS rise
            FROM fut GROUP BY 1,2,3,4,5,6,7
          )
          SELECT obs_day AS 观察日, flight_no AS 航班号, route AS 航线, dep_date AS 起飞日,
                 p0 AS 基准价, dtd AS 距起飞天数, 是否回补键, n_win,
                 CASE WHEN rise THEN 1 ELSE 0 END AS 标签_7天涨到线
          FROM agg WHERE n_win > 0
        ) TO '{rise_lab_path}' (FORMAT PARQUET)
        """)
        log("涨标签落盘", rise_lab_path, md5(rise_lab_path))
    else:
        log("复用涨标签", rise_lab_path, md5(rise_lab_path))

    # ---------- 2) 训练涨模型（A1 特征对齐）----------
    rise_model_path = os.path.join(OUT, "模型/model_涨到线_A1特征_seed42.txt")
    rise_report = {}
    if not os.path.exists(rise_model_path):
        log("加载训/验特征+涨标签…")
        days_tr_va = TRAIN_DAYS + VAL_DAYS
        days_sql = ",".join(f"'{d}'" for d in days_tr_va)
        feat_sql = ", ".join(f'f."{c}"' for c in feats)
        df = con.execute(f"""
          SELECT l.观察日, l.航班号, l.航线, l.起飞日, l.距起飞天数,
                 l.标签_7天涨到线, {feat_sql}
          FROM read_parquet('{rise_lab_path}') l
          JOIN read_parquet('{S1}') f
            ON f.观察日 = l.观察日 AND f.航班号 = l.航班号
           AND f.航线 = l.航线 AND CAST(f.起飞日 AS VARCHAR) = l.起飞日
          WHERE l.观察日 IN ({days_sql})
            AND l.距起飞天数 >= {DTD_MIN}
        """).fetchdf()
        log("涨训联结行", len(df), "日", sorted(df["观察日"].unique().tolist()))
        tr = df[df["观察日"].isin(TRAIN_DAYS)].reset_index(drop=True)
        va = df[df["观察日"].isin(VAL_DAYS)].reset_index(drop=True)
        del df; gc.collect()
        Xtr, Xva = make_X(tr, feats), make_X(va, feats)
        ytr = tr["标签_7天涨到线"].to_numpy(np.int8)
        yva = va["标签_7天涨到线"].to_numpy(np.int8)
        log("涨正例率 训", float(ytr.mean()), "验", float(yva.mean()), "训行", len(tr), "验行", len(va))
        dtr = lgb.Dataset(Xtr, ytr, feature_name=list(feats), free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
        booster = lgb.train(
            {**PARAMS, "seed": SEED}, dtr, num_boost_round=ROUNDS,
            valid_sets=[dva], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY, verbose=False)],
        )
        bi = int(booster.best_iteration or booster.num_trees())
        booster.save_model(rise_model_path, num_iteration=bi)
        pv = booster.predict(Xva, num_iteration=bi)
        auc_v = float(roc_auc_score(yva, pv))
        rise_report = {
            "路径": rise_model_path, "md5": md5(rise_model_path),
            "best_iteration": bi, "验证AUC": auc_v,
            "训练行": len(tr), "验证行": len(va),
            "训练正例率": float(ytr.mean()), "验证正例率": float(yva.mean()),
            "特征数": len(feats), "特征来源": "S3 A1 最终A特征（与降模型对齐）",
            "标签定义": "7天内价>=p0+100 或 >=p0*1.15",
            "LightGBM": PARAMS, "种子": SEED,
        }
        json.dump(rise_report, open(os.path.join(OUT, "中间/涨模型训练报告.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        log("涨模型已训", rise_model_path, "md5", rise_report["md5"], "VAL AUC", auc_v, "bi", bi)
        del tr, va, Xtr, Xva, dtr, dva, booster; gc.collect()
    else:
        rise_report = json.load(open(os.path.join(OUT, "中间/涨模型训练报告.json"), encoding="utf-8"))
        log("复用涨模型", rise_model_path, rise_report.get("md5"), "AUC", rise_report.get("验证AUC"))

    br = lgb.Booster(model_file=rise_model_path)
    bd = lgb.Booster(model_file=DROP_MODEL)
    bi_r = int(br.num_trees())

    # ---------- 3) 入场日：降分 + 基线赚亏（复用 S2/S3 标签）----------
    log("加载 VAL/TEST 入场行（联结 S2 降标签 + S1 特征）…")
    days_et = VAL_DAYS + TEST_DAYS
    days_sql = ",".join(f"'{d}'" for d in days_et)
    feat_sql = ", ".join(f'f."{c}"' for c in feats)
    entry = con.execute(f"""
      SELECT l.obs_day AS 观察日, l.flight_no AS 航班号, l.route AS 航线,
             CAST(l.dep_date AS VARCHAR) AS 起飞日, l.dtd AS 距起飞天数,
             l.p0 AS 价格, l.是否回补键,
             l.标签_7天降到线, l.降价收益_若收单, l.涨价亏损_若收单, l.n_win,
             {feat_sql}
      FROM read_parquet('{S2_LAB}') l
      JOIN read_parquet('{S1}') f
        ON f.观察日 = l.obs_day AND f.航班号 = l.flight_no
       AND f.航线 = l.route AND CAST(f.起飞日 AS VARCHAR) = CAST(l.dep_date AS VARCHAR)
      WHERE l.obs_day IN ({days_sql})
        AND l.dtd >= {DTD_MIN} AND l.dtd <= 30
        AND l.是否回补键 = '否'
    """).fetchdf()
    log("入场候选行", len(entry), "日", sorted(entry["观察日"].astype(str).unique().tolist()))

    X_e = make_X(entry, feats)
    entry["降分"] = bd.predict(X_e)
    entry["涨分_入场日"] = br.predict(X_e)  # 入场日涨分不用于提前出，仅备查
    del X_e; gc.collect()

    va = entry[entry["观察日"].astype(str).isin(VAL_DAYS)].reset_index(drop=True)
    te = entry[entry["观察日"].astype(str).isin(TEST_DAYS)].reset_index(drop=True)
    log("VAL", len(va), "TEST", len(te))

    # 测涨 AUC（需涨标签）
    te_rise = con.execute(f"""
      SELECT 观察日, 航班号, 航线, 起飞日, 标签_7天涨到线
      FROM read_parquet('{rise_lab_path}')
      WHERE 观察日 IN ({",".join(repr(d) for d in TEST_DAYS)}) AND 距起飞天数 >= {DTD_MIN}
    """).fetchdf()
    te_m = te.merge(te_rise, on=["观察日", "航班号", "航线", "起飞日"], how="inner")
    auc_te = float(roc_auc_score(te_m["标签_7天涨到线"], te_m["涨分_入场日"])) if len(te_m) and te_m["标签_7天涨到线"].nunique() > 1 else float("nan")
    rise_report["测试AUC"] = auc_te
    log("涨模型测试AUC", auc_te)

    # ---------- 4) VAL 选两档入场池 ----------
    yva = va["标签_7天降到线"].to_numpy(np.int8)
    sva = va["降价收益_若收单"].to_numpy(float)
    lva = va["涨价亏损_若收单"].to_numpy(float)
    pdv = va["降分"].to_numpy(float)
    curve = pool_curve(yva, pdv, sva, lva, XS_POOL)
    pd.DataFrame(curve).round(6).to_csv(os.path.join(OUT, "结果/VAL入场池扫描_读法B_dtd14.csv"),
                                        index=False, encoding="utf-8-sig")
    # 尖端 0.5%：对齐 S3 A1 验证选出
    tip = [r for r in curve if abs(r["出手比例"] - 0.005) < 1e-12][0]
    # 合计利润最大：在扫描上取 净省钱合计_k2 最大；若并列取更小比例
    best_tot = max(curve, key=lambda r: (r["净省钱合计_k2"], -r["出手比例"]))
    pools = {
        "尖端约0.5%": {"出手比例": tip["出手比例"], "阈值": tip["阈值"], "VAL条数": tip["条数"],
                      "选择依据": "对齐 S3 A1 验证选参 tip0.5%（每单k2峰）", **{k: tip[k] for k in ("每单净省钱_k1","每单净省钱_k2","净省钱合计_k1","净省钱合计_k2")}},
        "合计利润最大附近": {"出手比例": best_tot["出手比例"], "阈值": best_tot["阈值"], "VAL条数": best_tot["条数"],
                      "选择依据": f"VAL 读法B dtd>=14 上净省钱合计_k2 最大（扫描含0.33）；实际比例={best_tot['出手比例']}",
                      **{k: best_tot[k] for k in ("每单净省钱_k1","每单净省钱_k2","净省钱合计_k1","净省钱合计_k2")}},
    }
    log("入场池", {k: (v["出手比例"], round(v["阈值"], 4), v["VAL条数"]) for k, v in pools.items()})

    # ---------- 5) 为入场键构建路径价 + 路径日涨分 ----------
    def collect_keys(df, thr):
        return set(kstr(df.loc[df["降分"] >= thr]))

    all_keys = set()
    for p in pools.values():
        all_keys |= collect_keys(va, p["阈值"])
        all_keys |= collect_keys(te, p["阈值"])
    log("需路径键数", len(all_keys))

    # 路径观察日范围：入场日后 1..7 天，且 <= CUTOFF；可跨 exclude-from（只作路径价，不作入场）
    path_days_needed = []
    for d0 in VAL_DAYS + TEST_DAYS:
        t0 = pd.Timestamp(d0)
        for h in range(1, W + 1):
            d = str(t0 + pd.Timedelta(days=h))[:10]
            if d <= CUTOFF:
                path_days_needed.append(d)
    path_days_needed = sorted(set(path_days_needed))
    log("路径观察日", path_days_needed)

    # 写出键列表供 duckdb 过滤
    keys_df = pd.DataFrame({"键": list(all_keys)})
    keys_path = os.path.join(OUT, "中间/入场键列表.parquet")
    keys_df.to_parquet(keys_path, index=False)

    path_feat_path = os.path.join(OUT, "中间/路径行_价与特征.parquet")
    if not os.path.exists(path_feat_path):
        log("抽取路径行价+特征…")
        pdays_sql = ",".join(f"'{d}'" for d in path_days_needed)
        # also need 价格, 距起飞天数
        con.execute(f"""
        COPY (
          SELECT CAST(f.观察日 AS VARCHAR) AS 观察日, f.航班号, f.航线, CAST(f.起飞日 AS VARCHAR) AS 起飞日,
                 CAST(f.价格 AS DOUBLE) AS 价格, f.距起飞天数,
                 f.航班号 || '|' || f.航线 || '|' || CAST(f.起飞日 AS VARCHAR) AS 键,
                 {", ".join(f'f."{c}"' for c in feats)}
          FROM read_parquet('{S1}') f
          WHERE f.观察日 IN ({pdays_sql})
            AND length(CAST(f.航班号 AS VARCHAR)) <> 7
            AND (f.航班号 || '|' || f.航线 || '|' || CAST(f.起飞日 AS VARCHAR))
                IN (SELECT 键 FROM read_parquet('{keys_path}'))
        ) TO '{path_feat_path}' (FORMAT PARQUET)
        """)
        log("路径行落盘", path_feat_path, "行", con.execute(f"SELECT count(*) FROM read_parquet('{path_feat_path}')").fetchone()[0])
    else:
        log("复用路径行", path_feat_path)

    # 打涨分（分块）
    path_score_path = os.path.join(OUT, "中间/路径行_含涨分.parquet")
    if not os.path.exists(path_score_path):
        log("对路径行打涨分…")
        PR = pd.read_parquet(path_feat_path)
        log("路径行载入", len(PR))
        # 分块预测
        scores = np.empty(len(PR), dtype=np.float64)
        BS = 200_000
        for i in range(0, len(PR), BS):
            sl = PR.iloc[i:i+BS]
            scores[i:i+BS] = br.predict(make_X(sl, feats))
            log("  涨分块", i, "-", min(i+BS, len(PR)))
        PR = PR[["观察日", "航班号", "航线", "起飞日", "价格", "距起飞天数", "键"]].copy()
        PR["涨分"] = scores
        PR.to_parquet(path_score_path, index=False)
        del PR, scores; gc.collect()
        log("路径涨分落盘", path_score_path, md5(path_score_path))
    else:
        log("复用路径涨分", path_score_path)

    PRall = pd.read_parquet(path_score_path)
    PRall = PRall.assign(
        d=pd.to_datetime(PRall["观察日"]),
        p=PRall["价格"].astype(float),
        dd=PRall["距起飞天数"].astype(float),
        s=PRall["涨分"].astype(float),
    )
    PRall = PRall.sort_values(["键", "d"], kind="stable").drop_duplicates(["键", "d"]).reset_index(drop=True)
    log("路径去重后", len(PRall))

    # ---------- 6) 对每个入场池：VAL 选 θ → 落盘 → 再读测试 ----------
    rep = {
        "性质": "仅离线影子对照；不是过门结论，不上线，不动现网；通过≠上线",
        "授权": "Yangyang t1380u（经经理 Grok 转达）：涨侧/提前出手可否做、做完看对现有模型有无作用",
        "红线": "未碰线上推理服务/未上线/未真收/未写服务器库/未重启[REDACTED_SERVICE]/未读ana_extract·pull8001/未打印store.json；seats不入模",
        "衔接": {
            "v4路径": f"{ROOT.replace('重训_20261102','重训准备_20261102')}/v4_涨侧提前出手_20261029/",
            "v4b路径": "${PRED_ROOT}/重训准备_20261102/v4b_说明更正_20261029/",
            "规则说明": "路径结算/θ网格/R1–R5/无模型对照以 v4b 为准；数值逻辑同 v4；本包降模已换 S3 A1，v4b 结论不自动继承",
            "降模型": {"路径": DROP_MODEL, "md5": drop_md5, "特征数": 46, "标签": "A1_默认38加晋升ly"},
            "涨模型": rise_report,
            "默认38说明": "默认38/剔除24=S2短名单+经理安排，不得写成 t1360u 定案38",
        },
        "切分": {"训练": TRAIN_DAYS, "验证": VAL_DAYS, "测试": TEST_DAYS,
                "gap要求": ">=7", "exclude_from": EXCLUDE_FROM, "cutoff": CUTOFF,
                "采集窗切换日": SWITCH, "主评估dtd": "0..30", "读法B": DTD_MIN, "窗": W},
        "尺子": {"k1": "降价赚-1*涨价亏", "k2": "降价赚-2*涨价亏",
                "降到线": f"价<=p0-{DROP_YUAN} 或 <=p0*{1-DROP_RATIO}",
                "涨到线": f"价>=p0+{RISE_YUAN} 或 >=p0*{1+RISE_RATIO}"},
        "入场池": pools,
        "VAL入场池扫描": curve,
    }

    val_rows_all = []
    sel_all = {}
    # 先全部做完 VAL（两池），落盘后才碰测试结果
    val_mats = {}  # pool_name -> (E, Pm, Sm, Vm, Dm, p0, base_ref)

    for pool_name, pool in pools.items():
        thr = pool["阈值"]
        msk = (va["降分"] >= thr).to_numpy()
        Eraw = va.loc[msk].reset_index(drop=True)
        E = pd.DataFrame({
            "idx": np.arange(len(Eraw)),
            "键": kstr(Eraw),
            "t": pd.to_datetime(Eraw["观察日"]),
            "p0": Eraw["价格"].astype(float),
        })
        Pm, Sm, Vm, Dm = build_paths(E, PRall)
        p0 = E["p0"].to_numpy()
        covered = Vm.sum(1) >= 1
        if not covered.all():
            log("警告: VAL", pool_name, "无后续价", int((~covered).sum()), "/", len(E))
        # 仅保留有路径的
        E = E.loc[covered].reset_index(drop=True)
        Eraw = Eraw.loc[covered].reset_index(drop=True)
        E["idx"] = np.arange(len(E))
        Pm, Sm, Vm, Dm = build_paths(E, PRall)
        p0 = E["p0"].to_numpy()
        g0, l0, e0, h0, j0, s0 = outcome(Pm, Sm, Vm, p0, np.inf)
        ref_g = Eraw["降价收益_若收单"].to_numpy(float)
        ref_l = Eraw["涨价亏损_若收单"].to_numpy(float)
        bad = int((np.abs(g0 - ref_g) > 1e-6).sum() + (np.abs(l0 - ref_l) > 1e-6).sum())
        log("VAL基线核对", pool_name, "入场", len(E), "不一致", bad)
        # 允许少量因路径日缺测不一致，但记录
        rep.setdefault("基线路径重算核对_VAL", {})[pool_name] = {
            "入场单数": int(len(E)), "与数据集降价收益/涨价亏损不一致的数": bad,
            "说明": "路径价来自 S0/S1；若缺日可能导致少量不一致",
        }
        rows = []
        for sch, lab_, Sx, thx in schemes(Pm, Sm, Vm):
            g, l, e, h, j, sx = outcome(Pm, Sx, Vm, p0, thx)
            m = mets(g, l, e, None, Dm[np.arange(len(p0)), sx])
            rows.append({"入场池": pool_name, "入场口径": "读法B_dtd>=14", "方案": sch, "涨价几率门槛": lab_, **m})
        val_rows_all += rows
        base = rows[0]
        cand = [r for r in rows if r["方案"] == "涨侧提前出"]
        ctrl = [r for r in rows if r["方案"].startswith("对照")]
        top = max(r["每单净省钱_k2"] for r in cand)
        tied = [r for r in cand if top - r["每单净省钱_k2"] < TIE_YUAN]
        best = max(tied, key=lambda r: r["涨价几率门槛"] if isinstance(r["涨价几率门槛"], float) else -1)
        # 门槛可能是 float
        best = max(tied, key=lambda r: float(r["涨价几率门槛"]) if r["涨价几率门槛"] != "—" else -1)
        sup = (best["每单净省钱_k2"] - base["每单净省钱_k2"] >= SUPPORT_K2_GAIN) and (best["每单净省钱_k1"] >= base["每单净省钱_k1"])
        bc_ = max(ctrl, key=lambda r: r["每单净省钱_k2"])
        inc = (best["每单净省钱_k2"] - bc_["每单净省钱_k2"] >= SUPPORT_K2_GAIN) and (best["每单净省钱_k1"] >= bc_["每单净省钱_k1"])
        sel_all[pool_name] = {
            "入场池": pool_name,
            "降分阈值": float(thr),
            "出手比例": pool["出手比例"],
            "VAL入场单数": int(len(E)),
            "VAL选出门槛": float(best["涨价几率门槛"]),
            "VAL选出门槛_指标": best,
            "VAL不提前出_指标": base,
            "VAL是否支持提前出(R3)": "VAL支持提前出" if sup else "VAL不支持提前出",
            "VAL最好的无模型对照": {"方案": bc_["方案"], "门槛标签": bc_["涨价几率门槛"],
                              "k1": bc_["每单净省钱_k1"], "k2": bc_["每单净省钱_k2"]},
            "涨分是否有增量(R3b)": "涨分相对无模型对照有增量" if inc else "涨分相对朴素缩短等待无增量",
            "选定方案": (f"提前出(门槛{best['涨价几率门槛']})" if sup else "不提前出"),
            "VAL选定方案k1": (best if sup else base)["每单净省钱_k1"],
            "VAL选定方案k2": (best if sup else base)["每单净省钱_k2"],
            "基线核对不一致数": bad,
        }
        val_mats[pool_name] = {"thr": thr, "Eraw_cols": ["观察日", "航班号", "航线", "起飞日", "价格", "航司"] if "航司" in va.columns else ["观察日", "航班号", "航线", "起飞日", "价格"]}
        log("VAL选择", pool_name, sel_all[pool_name]["选定方案"], "θ", sel_all[pool_name]["VAL选出门槛"],
            sel_all[pool_name]["VAL是否支持提前出(R3)"])

    # 落盘 VAL（先于测试）
    pd.DataFrame(val_rows_all).round(4).to_csv(os.path.join(OUT, "结果/VAL选门槛过程表.csv"),
                                               index=False, encoding="utf-8-sig")
    val_sel_path = os.path.join(OUT, "结果/VAL选择_先于测试集读取.json")
    json.dump({
        "R规则": [
            "R1 θ网格与无模型对照同 v4b",
            "R2 VAL 每单k2最大；k2差<1元并列→更高θ",
            "R3 VAL支持提前出：相对基线 k2提高>=5 且 k1不下降；否则选定=不提前出",
            "R3b 涨分增量：相对最好无模型对照 k2高>=5 且 k1不低于",
            "R4 本文件落盘并记md5后才读测试入场单结果",
            "R5 VAL入场单路径可用其后7天价（可跨 exclude-from，只作路径）",
        ],
        "入场池": pools,
        "VAL选择": sel_all,
        "涨模型": {"路径": rise_model_path, "md5": md5(rise_model_path), "验证AUC": rise_report.get("验证AUC")},
        "降模型": {"路径": DROP_MODEL, "md5": drop_md5},
    }, open(val_sel_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    val_md5 = {
        "VAL选门槛过程表.csv": md5(os.path.join(OUT, "结果/VAL选门槛过程表.csv")),
        "VAL选择_先于测试集读取.json": md5(val_sel_path),
    }
    rep["VAL选择"] = sel_all
    rep["VAL选择文件md5(落盘后、读测试入场单前)"] = val_md5
    log("VAL选择已落盘（尚未读测试入场单结果）", val_md5)

    # ---------- 7) 测试 ----------
    # 补充航司列（若有）
    if "航司" not in te.columns:
        air = con.execute(f"""
          SELECT CAST(观察日 AS VARCHAR) AS 观察日, 航班号, 航线,
                 CAST(起飞日 AS VARCHAR) AS 起飞日, 航司
          FROM read_parquet('{S1}')
          WHERE CAST(观察日 AS VARCHAR) IN ({",".join(repr(d) for d in TEST_DAYS)})
        """).fetchdf()
        te = te.copy()
        te["观察日"] = te["观察日"].astype(str)
        te["起飞日"] = te["起飞日"].astype(str)
        te = te.merge(air, on=["观察日", "航班号", "航线", "起飞日"], how="left")

    test_rows_all = []
    daily = []
    air_rows = []
    delta_rows = []
    for pool_name, pool in pools.items():
        thr = pool["阈值"]
        th_sel = sel_all[pool_name]["VAL选出门槛"]
        msk = (te["降分"] >= thr).to_numpy()
        Eraw = te.loc[msk].reset_index(drop=True)
        E = pd.DataFrame({
            "idx": np.arange(len(Eraw)),
            "键": kstr(Eraw),
            "t": pd.to_datetime(Eraw["观察日"]),
            "p0": Eraw["价格"].astype(float),
        })
        Pm, Sm, Vm, Dm = build_paths(E, PRall)
        covered = Vm.sum(1) >= 1
        if not covered.all():
            log("警告: TEST", pool_name, "无后续价", int((~covered).sum()))
        E = E.loc[covered].reset_index(drop=True)
        Eraw = Eraw.loc[covered].reset_index(drop=True)
        E["idx"] = np.arange(len(E))
        Pm, Sm, Vm, Dm = build_paths(E, PRall)
        p0 = E["p0"].to_numpy()
        g0, l0, e0, h0, j0, s0 = outcome(Pm, Sm, Vm, p0, np.inf)
        ref_g = Eraw["降价收益_若收单"].to_numpy(float)
        ref_l = Eraw["涨价亏损_若收单"].to_numpy(float)
        bad = int((np.abs(g0 - ref_g) > 1e-6).sum() + (np.abs(l0 - ref_l) > 1e-6).sum())
        rep.setdefault("基线路径重算核对_测试", {})[pool_name] = {
            "入场单数": int(len(E)), "与数据集不一致的数": bad
        }
        log("TEST基线核对", pool_name, "入场", len(E), "不一致", bad)
        rows = []
        for sch, lab_, Sx, thx in schemes(Pm, Sm, Vm):
            g, l, e, h, j, sx = outcome(Pm, Sx, Vm, p0, thx)
            m = mets(g, l, e, None, Dm[np.arange(len(p0)), sx])
            b = mets(g0, l0, e0, None, Dm[np.arange(len(p0)), s0])
            rows.append({
                "入场池": pool_name, "入场口径": "读法B_dtd>=14",
                "方案": sch, "涨价几率门槛": lab_,
                "是否VAL选出门槛": bool(sch == "涨侧提前出" and lab_ == th_sel),
                "是否选定方案行": bool(
                    (sel_all[pool_name]["选定方案"] == "不提前出" and sch.startswith("不提前出"))
                    or (sch == "涨侧提前出" and lab_ == th_sel and sel_all[pool_name]["选定方案"].startswith("提前出"))
                ),
                **m,
                "相对不提前出_每单k1变化": m["每单净省钱_k1"] - b["每单净省钱_k1"],
                "相对不提前出_每单k2变化": m["每单净省钱_k2"] - b["每单净省钱_k2"],
                "相对不提前出_合计k1变化": m["净省钱合计_k1"] - b["净省钱合计_k1"],
                "相对不提前出_合计k2变化": m["净省钱合计_k2"] - b["净省钱合计_k2"],
            })
        test_rows_all += rows
        # 选定门槛 vs 基线摘要
        g, l, e, h, j, sx = outcome(Pm, Sm, Vm, p0, th_sel)
        bmet = mets(g0, l0, e0)
        smet = mets(g, l, e)
        delta_rows.append({
            "入场池": pool_name,
            "降分阈值": thr,
            "出手比例": pool["出手比例"],
            "测试入场单数": bmet["条数"],
            "VAL选出门槛": th_sel,
            "VAL判定": sel_all[pool_name]["VAL是否支持提前出(R3)"],
            "选定方案": sel_all[pool_name]["选定方案"],
            "不提前出_每单k1": bmet["每单净省钱_k1"],
            "不提前出_每单k2": bmet["每单净省钱_k2"],
            "不提前出_合计k1": bmet["净省钱合计_k1"],
            "不提前出_合计k2": bmet["净省钱合计_k2"],
            "VAL门槛提前出_每单k1": smet["每单净省钱_k1"],
            "VAL门槛提前出_每单k2": smet["每单净省钱_k2"],
            "VAL门槛提前出_合计k1": smet["净省钱合计_k1"],
            "VAL门槛提前出_合计k2": smet["净省钱合计_k2"],
            "Δ每单k1": smet["每单净省钱_k1"] - bmet["每单净省钱_k1"],
            "Δ每单k2": smet["每单净省钱_k2"] - bmet["每单净省钱_k2"],
            "Δ合计k1": smet["净省钱合计_k1"] - bmet["净省钱合计_k1"],
            "Δ合计k2": smet["净省钱合计_k2"] - bmet["净省钱合计_k2"],
            "提前出占比": smet["提前出占比"],
            "对A1是否起正面作用_按R3选定": (
                "按R3选定不提前出→对A1本档不建议加提前出"
                if sel_all[pool_name]["选定方案"] == "不提前出"
                else ("测试Δk2>0且Δk1>=0→有正面作用(仍仅离线)" if (smet["每单净省钱_k2"] - bmet["每单净省钱_k2"] > 0 and smet["每单净省钱_k1"] - bmet["每单净省钱_k1"] >= 0)
                      else "测试未同时改善k1/k2→未起可靠正面作用")
            ),
        })
        # 逐观察日 / 航司
        days = Eraw["观察日"].astype(str).to_numpy()
        for d in sorted(set(days)):
            mk = days == d
            a, b = mets(g0, l0, e0, mk), mets(g, l, e, mk)
            daily.append({"入场池": pool_name, "观察日": d, "入场单数": a["条数"],
                          "不提前出_k1": a.get("每单净省钱_k1"), "不提前出_k2": a.get("每单净省钱_k2"),
                          "VAL选出门槛": th_sel, "提前出_k1": b.get("每单净省钱_k1"),
                          "提前出_k2": b.get("每单净省钱_k2"), "提前出占比": b.get("提前出占比")})
        if "航司" in Eraw.columns:
            air = Eraw["航司"].astype(str).to_numpy()
            for a_ in sorted(set(air)):
                mk = air == a_
                b_, m_ = mets(g0, l0, e0, mk), mets(g, l, e, mk)
                if b_.get("条数", 0) == 0:
                    continue
                air_rows.append({
                    "入场池": pool_name, "航司": a_, "入场单数": b_["条数"],
                    "VAL选出门槛": th_sel,
                    "不提前出_k1": b_["每单净省钱_k1"], "不提前出_k2": b_["每单净省钱_k2"],
                    "提前出_k1": m_["每单净省钱_k1"], "提前出_k2": m_["每单净省钱_k2"],
                    "k1变化": m_["每单净省钱_k1"] - b_["每单净省钱_k1"],
                    "k2变化": m_["每单净省钱_k2"] - b_["每单净省钱_k2"],
                    "提前出占比": m_["提前出占比"],
                })

    pd.DataFrame(test_rows_all).round(4).to_csv(os.path.join(OUT, "结果/测试结果表_读法B_dtd14_两档入场池.csv"),
                                                index=False, encoding="utf-8-sig")
    pd.DataFrame(delta_rows).round(4).to_csv(os.path.join(OUT, "结果/相对基线_Δk1Δk2_两档摘要.csv"),
                                             index=False, encoding="utf-8-sig")
    pd.DataFrame(daily).round(4).to_csv(os.path.join(OUT, "结果/测试逐观察日_VAL选出门槛.csv"),
                                        index=False, encoding="utf-8-sig")
    if air_rows:
        pd.DataFrame(air_rows).round(4).to_csv(os.path.join(OUT, "结果/测试按航司_VAL选出门槛.csv"),
                                               index=False, encoding="utf-8-sig")

    # 短对照：读法A dtd>=8 / 全量 — 只报 tip 池基线 vs VAL θ（可选，省时只做 tip）
    # 跳过以控时；正文以读法B 为主。在说明里写明。

    rep["测试摘要_相对基线"] = delta_rows
    rep["涨模型AUC核对"] = {
        "验证AUC": rise_report.get("验证AUC"),
        "测试AUC": auc_te,
    }
    rep["耗时秒"] = round(time.time() - T0, 1)
    json.dump(rep, open(os.path.join(OUT, "结果/涨侧提前出手报告_S3A1.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    json.dump(rise_report, open(os.path.join(OUT, "中间/涨模型训练报告.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log("完成主计算", round(time.time() - T0, 1), "s")
    print("DONE")


if __name__ == "__main__":
    main()
