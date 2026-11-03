#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 A1 之上的「价格回归 + 日期回归」离线对照。
仅离线；不碰线上推理服务 / 不上线 / 不真收 / 不写服务器库 / 不重启 [REDACTED_SERVICE]。
通过≠上线。seats 不入模。未读 ana_extract/pull8001/store.json。

授权：Yangyang t1389（经经理 Grok 转达）离线做价格回归（还能降多少）+ 日期回归（大概哪天降到线/适合出），叠在 A1 上看赚不赚。
读法B：入场 dtd>=14；卖出须在起飞前7天前完成（结算时 dtd>=7）。
"""
import os, sys, json, time, hashlib, gc, math
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import lightgbm as lgb
import duckdb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "${PROJECT_ROOT}"
OUT = os.path.join(ROOT, "价格日期回归对照_20261102")
S0 = os.path.join(ROOT, "S0_底表/数据/主表_全航司日价_dtd0_30.parquet")
S1 = os.path.join(ROOT, "S1_特征/数据/特征矩阵_全航司_dtd0_30.parquet")
S2_LAB = os.path.join(ROOT, "S2_有用性/数据/标签_全航司_满窗.parquet")
S3 = os.path.join(ROOT, "S3_训练")
DROP_MODEL = os.path.join(S3, "模型/model_A1_默认38加晋升ly_seed42.txt")
A_JSON = os.path.join(S3, "中间/A版定案.json")

TRAIN_DAYS = ["2026-10-15", "2026-10-16", "2026-10-17"]
VAL_DAYS = ["2026-10-25"]
TEST_DAYS = ["2026-10-24"]
EXCLUDE_FROM = "2026-10-28"
CUTOFF = "2026-11-01"
SWITCH = "2026-10-29"
DTD_MIN = 14
W = 7
DROP_YUAN, DROP_RATIO = 100.0, 0.15
SEED = 42
SENTINEL_DAYS = W + 1  # 未触达降到线 → 8
PARAMS_REG = dict(objective="regression", metric="l2", learning_rate=0.05,
                  num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbosity=-1, n_jobs=1)
ROUNDS, EARLY = 800, 40
# 价格辅助收：预测还能降多少 门槛（元）
PRICE_FILTER_GRID = [30, 50, 80, 100, 120, 150, 200, 250, 300, 400]
# 价格辅助出：预测降幅低于此则第1观察日提前出
PRICE_EARLY_GRID = [30, 50, 80, 100, 150, 200]
# 日期晚触达提前出：预测天数 > θ 则第1观察日提前出
DATE_EARLY_GRID = [2, 3, 4, 5, 6, 7]
TIE_YUAN, SUPPORT_K2_GAIN = 1.0, 5.0
XS_POOL = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.60]
SETTLE_MIN = 7
CTRL_N = [1, 2, 3, 4]


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
    """E: idx,键,t,p0；PR: 键,d,p,dd。返回 P,V,D 矩阵 n×W（无涨分）。"""
    m = E[["idx", "键", "t", "p0"]].merge(PR[["键", "d", "p", "dd"]], on="键", how="inner")
    h = (m["d"] - m["t"]).dt.days
    m = m.loc[(h >= 1) & (h <= W)].sort_values(["idx", "d"], kind="stable")
    pos = m.groupby("idx").cumcount().to_numpy()
    n = len(E)
    P = np.full((n, W), np.nan)
    V = np.zeros((n, W), bool)
    D = np.full((n, W), np.nan)
    if len(m):
        ii = m["idx"].to_numpy()
        P[ii, pos] = m["p"].to_numpy()
        V[ii, pos] = True
        D[ii, pos] = m["dd"].to_numpy()
    return P, V, D


def outcome_baseline(P, V, p0):
    """不提前出：首次降到线按该价；否则窗末价。"""
    n = len(p0)
    ar = np.arange(n)
    H = V & ((P <= p0[:, None] - DROP_YUAN) | (P <= p0[:, None] * (1.0 - DROP_RATIO)))
    has = H.any(1)
    j = H.argmax(1)
    hit = has
    pj = P[ar, j]
    last = V.sum(1) - 1
    pend = P[ar, np.maximum(last, 0)]
    gain = np.where(hit, p0 - pj, 0.0)
    loss = np.where(hit, 0.0, np.maximum(pend - p0, 0.0))
    early = np.zeros(n, dtype=bool)
    sidx = np.where(has, j, np.maximum(last, 0))
    return gain, loss, early, hit, sidx


def outcome_early_day1(P, V, p0):
    """第1个有价观察日提前出（赚=0，亏=max(p1-p0,0)）；若无价则退回基线。"""
    n = len(p0)
    ar = np.arange(n)
    # 先看是否有路径价
    has_any = V.any(1)
    j0 = V.argmax(1)  # 第一个 True
    pj = P[ar, j0]
    # 也可在第1日已降到线 → 算命中
    H0 = has_any & ((pj <= p0 - DROP_YUAN) | (pj <= p0 * (1.0 - DROP_RATIO)))
    gain = np.where(H0, p0 - pj, 0.0)
    loss = np.where(H0, 0.0, np.where(has_any, np.maximum(pj - p0, 0.0), 0.0))
    early = has_any & ~H0
    # 无路径：0
    gain = np.where(has_any, gain, 0.0)
    loss = np.where(has_any, loss, 0.0)
    sidx = np.where(has_any, j0, 0)
    return gain, loss, early, H0, sidx


def outcome_date_pred_settle(P, V, p0, pred_days):
    """按预测触达日结算：到降到线立刻出；否则在 clip(round(pred),1,W) 对应有价日结算；
    若该日无价则向后找有价日，再没有则窗末。pred>=SENTINEL 视为倾向等满窗（目标日=W）。"""
    n = len(p0)
    ar = np.arange(n)
    H = V & ((P <= p0[:, None] - DROP_YUAN) | (P <= p0[:, None] * (1.0 - DROP_RATIO)))
    # 目标出手日（1..W）
    pd_ = np.asarray(pred_days, dtype=float)
    target = np.clip(np.rint(pd_), 1, W).astype(int)
    # 哨兵/很大 → 等满窗
    target = np.where(pd_ >= SENTINEL_DAYS - 0.5, W, target)
    # 事件：降到线 或 到达目标日（有价）
    day_idx = np.arange(W)[None, :]  # 0..W-1
    at_target = V & (day_idx + 1 >= target[:, None])
    ev = H | at_target
    has = ev.any(1)
    j = ev.argmax(1)
    hit = has & H[ar, j]
    early = has & ~hit  # 按预测日提前/定时出且未降到线
    pj = P[ar, j]
    last = V.sum(1) - 1
    pend = P[ar, np.maximum(last, 0)]
    # 若完全无事件（无价），用窗末
    use_j = np.where(has, j, np.maximum(last, 0))
    pj2 = P[ar, use_j]
    gain = np.where(hit, p0 - pj, 0.0)
    loss = np.where(hit, 0.0, np.where(has | (last >= 0), np.maximum(pj2 - p0, 0.0), 0.0))
    # 无路径
    has_path = V.any(1)
    gain = np.where(has_path, gain, 0.0)
    loss = np.where(has_path, loss, 0.0)
    early = early & has_path
    return gain, loss, early, hit, use_j


def outcome_cond_early(P, V, p0, do_early):
    """对 do_early=True 的样本第1日提前出，其余走基线。"""
    g0, l0, e0, h0, s0 = outcome_baseline(P, V, p0)
    g1, l1, e1, h1, s1 = outcome_early_day1(P, V, p0)
    do = np.asarray(do_early, dtype=bool)
    gain = np.where(do, g1, g0)
    loss = np.where(do, l1, l0)
    early = np.where(do, e1, e0)
    hit = np.where(do, h1, h0)
    sidx = np.where(do, s1, s0)
    return gain, loss, early, hit, sidx


def mets(gain, loss, early, msk=None, sd=None):
    if msk is not None:
        gain, loss, early = gain[msk], loss[msk], early[msk]
        if sd is not None:
            sd = sd[msk]
    n = len(gain)
    if n == 0:
        return {"条数": 0, "每单净省钱_k1": 0.0, "每单净省钱_k2": 0.0,
                "净省钱合计_k1": 0.0, "净省钱合计_k2": 0.0, "提前出占比": 0.0,
                "平均涨价亏损_元": 0.0, "平均降价收益_元": 0.0}
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


def reg_metrics(y, p, name):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "名称": name,
        "条数": int(len(y)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(math.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "标签均值": float(np.mean(y)),
        "预测均值": float(np.mean(p)),
    }


def pick_best(cands, base, support_k2=SUPPORT_K2_GAIN):
    """在候选里按每单k2最大选；并列(差<1元)取更保守门槛（更大的价格θ / 更大的日期θ）。
    R3：相对基线 Δk2>=support 且 k1不降 → 支持该方案；否则选定基线。"""
    if not cands:
        return None, False, base
    top = max(r["每单净省钱_k2"] for r in cands)
    tied = [r for r in cands if top - r["每单净省钱_k2"] < TIE_YUAN]
    # 保守：门槛数值更大
    def thr_key(r):
        t = r.get("门槛数值", 0)
        try:
            return float(t)
        except Exception:
            return 0.0
    best = max(tied, key=thr_key)
    sup = (best["每单净省钱_k2"] - base["每单净省钱_k2"] >= support_k2) and (
        best["每单净省钱_k1"] >= base["每单净省钱_k1"] - 1e-9)
    return best, bool(sup), (best if sup else base)


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

    # ---------- 1) 价格/日期回归标签（观察日之后未来窗，防泄露）----------
    lab_path = os.path.join(OUT, "中间/标签_价格日期回归_满窗.parquet")
    if not os.path.exists(lab_path):
        log("构建价格/日期回归标签…")
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
            SELECT b.*, CAST(p.价格 AS DOUBLE) AS bp,
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
                   min(bp) AS p_min,
                   min(CASE WHEN bp <= p0 - {DROP_YUAN} OR bp <= p0 * {1.0 - DROP_RATIO}
                            THEN h END) AS h_drop
            FROM fut GROUP BY 1,2,3,4,5,6,7
          )
          SELECT obs_day AS 观察日, flight_no AS 航班号, route AS 航线, dep_date AS 起飞日,
                 p0 AS 基准价, dtd AS 距起飞天数, 是否回补键, n_win,
                 p_min AS 窗内最低未来价,
                 p0 - p_min AS 还能降多少_元,
                 CASE WHEN h_drop IS NULL THEN 0 ELSE 1 END AS 是否触达降到线,
                 CASE WHEN h_drop IS NULL THEN {SENTINEL_DAYS} ELSE CAST(h_drop AS INTEGER) END
                   AS 触达降到线天数,
                 CASE WHEN h_drop IS NULL THEN '未触达_哨兵{SENTINEL_DAYS}' ELSE '已触达' END
                   AS 触达状态
          FROM agg WHERE n_win > 0
        ) TO '{lab_path}' (FORMAT PARQUET)
        """)
        log("回归标签落盘", lab_path, md5(lab_path))
    else:
        log("复用回归标签", lab_path, md5(lab_path))
    nlab = con.execute(f"SELECT count(*) FROM read_parquet('{lab_path}')").fetchone()[0]
    log("标签行数", nlab)

    # ---------- 2) 训练价格回归 / 日期回归 ----------
    price_model = os.path.join(OUT, "模型/model_价格回归_还能降多少_A1特征_seed42.txt")
    date_model = os.path.join(OUT, "模型/model_日期回归_触达降到线天数_A1特征_seed42.txt")
    reg_report = {"价格回归": {}, "日期回归": {}}

    def train_reg(target_col, model_path, report_key):
        if os.path.exists(model_path) and os.path.exists(os.path.join(OUT, f"中间/{report_key}训练报告.json")):
            rep = json.load(open(os.path.join(OUT, f"中间/{report_key}训练报告.json"), encoding="utf-8"))
            log("复用", report_key, model_path, rep.get("md5"))
            return rep
        log("训练", report_key, "目标", target_col)
        days_tr_va = TRAIN_DAYS + VAL_DAYS
        days_sql = ",".join(f"'{d}'" for d in days_tr_va)
        feat_sql = ", ".join(f'f."{c}"' for c in feats)
        df = con.execute(f"""
          SELECT l.观察日, l.航班号, l.航线, l.起飞日, l.距起飞天数,
                 l."{target_col}" AS y, {feat_sql}
          FROM read_parquet('{lab_path}') l
          JOIN read_parquet('{S1}') f
            ON f.观察日 = l.观察日 AND f.航班号 = l.航班号
           AND f.航线 = l.航线 AND CAST(f.起飞日 AS VARCHAR) = l.起飞日
          WHERE l.观察日 IN ({days_sql})
            AND l.距起飞天数 >= {DTD_MIN}
            AND l.是否回补键 = '否'
        """).fetchdf()
        log(report_key, "联结行", len(df))
        tr = df[df["观察日"].astype(str).isin(TRAIN_DAYS)].reset_index(drop=True)
        va = df[df["观察日"].astype(str).isin(VAL_DAYS)].reset_index(drop=True)
        del df; gc.collect()
        Xtr, Xva = make_X(tr, feats), make_X(va, feats)
        ytr = tr["y"].to_numpy(float)
        yva = va["y"].to_numpy(float)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=list(feats), free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
        booster = lgb.train(
            {**PARAMS_REG, "seed": SEED}, dtr, num_boost_round=ROUNDS,
            valid_sets=[dva], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY, verbose=False)],
        )
        bi = int(booster.best_iteration or booster.num_trees())
        booster.save_model(model_path, num_iteration=bi)
        pv = booster.predict(Xva, num_iteration=bi)
        vm = reg_metrics(yva, pv, "验证")
        # 测试日指标稍后补
        rep = {
            "路径": model_path, "md5": md5(model_path),
            "目标列": target_col, "best_iteration": bi,
            "训练行": len(tr), "验证行": len(va),
            "验证指标": vm,
            "特征数": len(feats), "特征来源": "S3 A1 最终A特征（46列，与降模型对齐）",
            "LightGBM": PARAMS_REG, "种子": SEED,
        }
        json.dump(rep, open(os.path.join(OUT, f"中间/{report_key}训练报告.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        log(report_key, "完成", "md5", rep["md5"], "VAL MAE", vm["MAE"], "bi", bi)
        del tr, va, Xtr, Xva, dtr, dva, booster; gc.collect()
        return rep

    # 若首次标签 SQL 含坏窗口函数，确保 lab 可用
    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{lab_path}')").fetchdf()["column_name"].tolist()
    if "还能降多少_元" not in cols:
        raise SystemExit("标签缺列 还能降多少_元: " + str(cols))

    reg_report["价格回归"] = train_reg("还能降多少_元", price_model, "价格回归")
    reg_report["价格回归"]["标签定义"] = (
        "从观察日起至满窗终点（读法B可卖窗W=7）内，仅用观察日之后可见的未来价："
        "还能降多少_元 = 基准价p0 − 窗内最低未来价；特征不含未来价/未来标签。"
    )
    reg_report["日期回归"] = train_reg("触达降到线天数", date_model, "日期回归")
    reg_report["日期回归"]["标签定义"] = (
        f"窗内首次触达降到线（价<=p0-{DROP_YUAN} 或 <=p0*{1-DROP_RATIO}）相对观察日的天数；"
        f"未触达 → 哨兵 {SENTINEL_DAYS}（截尾，不删除样本）。特征不含未来价/未来标签。"
    )

    bp = lgb.Booster(model_file=price_model)
    bd_date = lgb.Booster(model_file=date_model)
    bd = lgb.Booster(model_file=DROP_MODEL)

    # ---------- 3) VAL/TEST 入场行：降分 + 回归预测 + S2 基线赚亏 ----------
    log("加载 VAL/TEST 入场行…")
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
    log("入场候选行", len(entry))

    # 联结回归真值（评估用）
    lab_et = con.execute(f"""
      SELECT 观察日, 航班号, 航线, 起飞日, 还能降多少_元, 触达降到线天数, 是否触达降到线
      FROM read_parquet('{lab_path}')
      WHERE 观察日 IN ({days_sql}) AND 距起飞天数 >= {DTD_MIN}
    """).fetchdf()
    entry = entry.merge(lab_et, on=["观察日", "航班号", "航线", "起飞日"], how="left")

    X_e = make_X(entry, feats)
    entry["降分"] = bd.predict(X_e)
    entry["预测_还能降多少"] = bp.predict(X_e)
    entry["预测_触达天数"] = bd_date.predict(X_e)
    del X_e; gc.collect()

    va = entry[entry["观察日"].astype(str).isin(VAL_DAYS)].reset_index(drop=True)
    te = entry[entry["观察日"].astype(str).isin(TEST_DAYS)].reset_index(drop=True)
    log("VAL", len(va), "TEST", len(te))

    # 回归测试指标
    te_ok = te.dropna(subset=["还能降多少_元", "触达降到线天数"])
    reg_report["价格回归"]["测试指标"] = reg_metrics(
        te_ok["还能降多少_元"], te_ok["预测_还能降多少"], "测试")
    reg_report["日期回归"]["测试指标"] = reg_metrics(
        te_ok["触达降到线天数"], te_ok["预测_触达天数"], "测试")
    # 补验证指标若复用
    va_ok = va.dropna(subset=["还能降多少_元", "触达降到线天数"])
    if "验证指标" not in reg_report["价格回归"] or not reg_report["价格回归"].get("验证指标"):
        reg_report["价格回归"]["验证指标"] = reg_metrics(
            va_ok["还能降多少_元"], va_ok["预测_还能降多少"], "验证")
        reg_report["日期回归"]["验证指标"] = reg_metrics(
            va_ok["触达降到线天数"], va_ok["预测_触达天数"], "验证")
    log("价格回归 VAL MAE", reg_report["价格回归"].get("验证指标", {}).get("MAE"),
        "TEST MAE", reg_report["价格回归"]["测试指标"]["MAE"])
    log("日期回归 VAL MAE", reg_report["日期回归"].get("验证指标", {}).get("MAE"),
        "TEST MAE", reg_report["日期回归"]["测试指标"]["MAE"])

    json.dump(reg_report["价格回归"], open(os.path.join(OUT, "中间/价格回归训练报告.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(reg_report["日期回归"], open(os.path.join(OUT, "中间/日期回归训练报告.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---------- 4) VAL 选两档入场池（对齐涨侧/S3）----------
    yva = va["标签_7天降到线"].to_numpy(np.int8)
    sva = va["降价收益_若收单"].to_numpy(float)
    lva = va["涨价亏损_若收单"].to_numpy(float)
    pdv = va["降分"].to_numpy(float)
    curve = pool_curve(yva, pdv, sva, lva, XS_POOL)
    pd.DataFrame(curve).round(6).to_csv(os.path.join(OUT, "结果/VAL入场池扫描_读法B_dtd14.csv"),
                                        index=False, encoding="utf-8-sig")
    tip = [r for r in curve if abs(r["出手比例"] - 0.005) < 1e-12][0]
    best_tot = max(curve, key=lambda r: (r["净省钱合计_k2"], -r["出手比例"]))
    pools = {
        "尖端约0.5%": {
            "出手比例": tip["出手比例"], "阈值": tip["阈值"], "VAL条数": tip["条数"],
            "选择依据": "对齐 S3 A1 验证选参 tip0.5%（每单k2峰）",
            **{k: tip[k] for k in ("每单净省钱_k1", "每单净省钱_k2", "净省钱合计_k1", "净省钱合计_k2")},
        },
        "合计利润最大附近": {
            "出手比例": best_tot["出手比例"], "阈值": best_tot["阈值"], "VAL条数": best_tot["条数"],
            "选择依据": f"VAL 读法B dtd>=14 上净省钱合计_k2 最大（扫描含0.33）；实际比例={best_tot['出手比例']}",
            **{k: best_tot[k] for k in ("每单净省钱_k1", "每单净省钱_k2", "净省钱合计_k1", "净省钱合计_k2")},
        },
    }
    log("入场池", {k: (v["出手比例"], round(v["阈值"], 4), v["VAL条数"]) for k, v in pools.items()})

    # ---------- 5) 路径价（入场键）----------
    def collect_keys(df, thr):
        return set(kstr(df.loc[df["降分"] >= thr]))

    all_keys = set()
    for p in pools.values():
        all_keys |= collect_keys(va, p["阈值"])
        all_keys |= collect_keys(te, p["阈值"])
    log("需路径键数", len(all_keys))

    path_days_needed = []
    for d0 in VAL_DAYS + TEST_DAYS:
        t0 = pd.Timestamp(d0)
        for h in range(1, W + 1):
            d = str(t0 + pd.Timedelta(days=h))[:10]
            if d <= CUTOFF:
                path_days_needed.append(d)
    path_days_needed = sorted(set(path_days_needed))
    log("路径观察日", path_days_needed)

    keys_path = os.path.join(OUT, "中间/入场键列表.parquet")
    pd.DataFrame({"键": list(all_keys)}).to_parquet(keys_path, index=False)

    path_feat_path = os.path.join(OUT, "中间/路径行_价.parquet")
    if not os.path.exists(path_feat_path):
        log("抽取路径行价…")
        pdays_sql = ",".join(f"'{d}'" for d in path_days_needed)
        # 优先从 S0 取价（更轻）；dtd 也从 S0
        con.execute(f"""
        COPY (
          SELECT CAST(观察日 AS VARCHAR) AS 观察日, 航班号, 航线, CAST(起飞日 AS VARCHAR) AS 起飞日,
                 CAST(价格 AS DOUBLE) AS 价格, 距起飞天数,
                 航班号 || '|' || 航线 || '|' || CAST(起飞日 AS VARCHAR) AS 键
          FROM read_parquet('{S0}')
          WHERE CAST(观察日 AS VARCHAR) IN ({pdays_sql})
            AND length(CAST(航班号 AS VARCHAR)) <> 7
            AND (航班号 || '|' || 航线 || '|' || CAST(起飞日 AS VARCHAR))
                IN (SELECT 键 FROM read_parquet('{keys_path}'))
        ) TO '{path_feat_path}' (FORMAT PARQUET)
        """)
        log("路径行落盘", path_feat_path,
            con.execute(f"SELECT count(*) FROM read_parquet('{path_feat_path}')").fetchone()[0])
    else:
        log("复用路径行", path_feat_path)

    PRall = pd.read_parquet(path_feat_path)
    PRall = PRall.assign(
        d=pd.to_datetime(PRall["观察日"]),
        p=PRall["价格"].astype(float),
        dd=PRall["距起飞天数"].astype(float),
    )
    PRall = PRall.sort_values(["键", "d"], kind="stable").drop_duplicates(["键", "d"]).reset_index(drop=True)
    log("路径去重后", len(PRall))

    # ---------- 6) 对每个池：扫方案 → VAL 选择落盘 → 再测 ----------
    def eval_schemes_on(Eraw, pool_name):
        E = pd.DataFrame({
            "idx": np.arange(len(Eraw)),
            "键": kstr(Eraw),
            "t": pd.to_datetime(Eraw["观察日"]),
            "p0": Eraw["价格"].astype(float),
        })
        P, V, D = build_paths(E, PRall)
        covered = V.any(1)
        E = E.loc[covered].reset_index(drop=True)
        Eraw2 = Eraw.loc[covered].reset_index(drop=True)
        E["idx"] = np.arange(len(E))
        P, V, D = build_paths(E, PRall)
        p0 = E["p0"].to_numpy()
        pred_drop = Eraw2["预测_还能降多少"].to_numpy(float)
        pred_days = Eraw2["预测_触达天数"].to_numpy(float)

        rows = []
        # 基线
        g, l, e, h, sx = outcome_baseline(P, V, p0)
        base = {"入场池": pool_name, "方案族": "纯A1基线", "方案": "纯A1阈值出手_不提前出",
                "门槛标签": "—", "门槛数值": 0, **mets(g, l, e, None, D[np.arange(len(p0)), sx])}
        rows.append(base)

        # 对照：第N日必出
        for N in CTRL_N:
            do = np.ones(len(p0), dtype=bool)  # 简化：用 early day1 仅 N=1；N>1 用 date settle target=N
            if N == 1:
                g, l, e, h, sx = outcome_early_day1(P, V, p0)
            else:
                g, l, e, h, sx = outcome_date_pred_settle(P, V, p0, np.full(len(p0), float(N)))
            rows.append({"入场池": pool_name, "方案族": "无模型对照",
                         "方案": f"对照_入场后第{N}个观察日必出", "门槛标签": "—", "门槛数值": N,
                         **mets(g, l, e, None, D[np.arange(len(p0)), sx])})

        # A1+价格回归辅助收
        for th in PRICE_FILTER_GRID:
            msk = pred_drop >= th
            if msk.sum() == 0:
                continue
            g, l, e, h, sx = outcome_baseline(P, V, p0)
            rows.append({"入场池": pool_name, "方案族": "A1+价格回归辅助收",
                         "方案": f"价格过滤_预测降幅>={th}", "门槛标签": f">={th}", "门槛数值": th,
                         **mets(g, l, e, msk, D[np.arange(len(p0)), sx])})

        # A1+价格回归辅助出
        for th in PRICE_EARLY_GRID:
            do = pred_drop < th
            g, l, e, h, sx = outcome_cond_early(P, V, p0, do)
            rows.append({"入场池": pool_name, "方案族": "A1+价格回归辅助出",
                         "方案": f"预测降幅<{th}则第1日提前出", "门槛标签": f"<{th}", "门槛数值": th,
                         **mets(g, l, e, None, D[np.arange(len(p0)), sx])})

        # A1+日期回归：按预测日结算
        g, l, e, h, sx = outcome_date_pred_settle(P, V, p0, pred_days)
        rows.append({"入场池": pool_name, "方案族": "A1+日期回归按预测日出手",
                     "方案": "按预测触达日结算_未触达则倾向满窗", "门槛标签": "round(pred)",
                     "门槛数值": 0,
                     **mets(g, l, e, None, D[np.arange(len(p0)), sx])})

        # A1+日期回归：晚触达提前出
        for th in DATE_EARLY_GRID:
            do = pred_days > th
            g, l, e, h, sx = outcome_cond_early(P, V, p0, do)
            rows.append({"入场池": pool_name, "方案族": "A1+日期回归晚触达提前出",
                         "方案": f"预测触达天数>{th}则第1日提前出", "门槛标签": f">{th}", "门槛数值": th,
                         **mets(g, l, e, None, D[np.arange(len(p0)), sx])})

        # 两者：价格过滤 + 按预测日结算
        for th in PRICE_FILTER_GRID:
            msk = pred_drop >= th
            if msk.sum() == 0:
                continue
            g, l, e, h, sx = outcome_date_pred_settle(P, V, p0, pred_days)
            rows.append({"入场池": pool_name, "方案族": "A1+价格过滤+日期按预测日",
                         "方案": f"价格>={th}且按预测日结算", "门槛标签": f">={th}+pred日",
                         "门槛数值": th,
                         **mets(g, l, e, msk, D[np.arange(len(p0)), sx])})

        # 两者：价格过滤 + 晚触达提前出（固定日期θ=4 作代表网格交叉）
        for th_p in PRICE_FILTER_GRID:
            for th_d in [3, 4, 5]:
                msk = pred_drop >= th_p
                if msk.sum() == 0:
                    continue
                do = pred_days > th_d
                g, l, e, h, sx = outcome_cond_early(P, V, p0, do)
                rows.append({"入场池": pool_name, "方案族": "A1+价格过滤+日期晚触达提前出",
                             "方案": f"价格>={th_p}且预测天数>{th_d}提前出",
                             "门槛标签": f"p>={th_p},d>{th_d}",
                             "门槛数值": th_p + th_d / 100.0,
                             **mets(g, l, e, msk, D[np.arange(len(p0)), sx])})

        # 基线核对
        g0, l0, e0, h0, s0 = outcome_baseline(P, V, p0)
        ref_g = Eraw2["降价收益_若收单"].to_numpy(float)
        ref_l = Eraw2["涨价亏损_若收单"].to_numpy(float)
        bad = int((np.abs(g0 - ref_g) > 1e-6).sum() + (np.abs(l0 - ref_l) > 1e-6).sum())
        return rows, base, bad, len(E), {
            "Eraw": Eraw2, "P": P, "V": V, "D": D, "p0": p0,
            "pred_drop": pred_drop, "pred_days": pred_days,
        }

    val_rows_all = []
    sel_all = {}
    val_mats = {}

    FAMILY_KEYS = [
        ("A1+价格回归辅助收", "价格回归辅助"),
        ("A1+价格回归辅助出", "价格回归辅助"),
        ("A1+日期回归按预测日出手", "日期回归辅助"),
        ("A1+日期回归晚触达提前出", "日期回归辅助"),
        ("A1+价格过滤+日期按预测日", "两者都用"),
        ("A1+价格过滤+日期晚触达提前出", "两者都用"),
    ]

    for pool_name, pool in pools.items():
        thr = pool["阈值"]
        Eraw = va.loc[va["降分"] >= thr].reset_index(drop=True)
        rows, base, bad, n_e, mats = eval_schemes_on(Eraw, pool_name)
        val_rows_all += rows
        val_mats[pool_name] = mats
        log("VAL基线核对", pool_name, "入场", n_e, "不一致", bad)

        # 各族选优
        family_sel = {}
        # 价格回归辅助：辅助收与辅助出两族里一起比，再标支持与否
        price_cands = [r for r in rows if r["方案族"] in ("A1+价格回归辅助收", "A1+价格回归辅助出")]
        best_p, sup_p, chosen_p = pick_best(price_cands, base)
        family_sel["价格回归辅助"] = {
            "VAL最优候选": best_p, "VAL是否支持(R3)": "支持" if sup_p else "不支持",
            "选定方案": chosen_p["方案"] if chosen_p else "纯A1阈值出手_不提前出",
            "选定明细": chosen_p,
        }
        date_cands = [r for r in rows if r["方案族"] in ("A1+日期回归按预测日出手", "A1+日期回归晚触达提前出")]
        best_d, sup_d, chosen_d = pick_best(date_cands, base)
        family_sel["日期回归辅助"] = {
            "VAL最优候选": best_d, "VAL是否支持(R3)": "支持" if sup_d else "不支持",
            "选定方案": chosen_d["方案"] if chosen_d else "纯A1阈值出手_不提前出",
            "选定明细": chosen_d,
        }
        both_cands = [r for r in rows if r["方案族"] in ("A1+价格过滤+日期按预测日", "A1+价格过滤+日期晚触达提前出")]
        best_b, sup_b, chosen_b = pick_best(both_cands, base)
        family_sel["两者都用"] = {
            "VAL最优候选": best_b, "VAL是否支持(R3)": "支持" if both_cands and sup_b else "不支持",
            "选定方案": chosen_b["方案"] if chosen_b else "纯A1阈值出手_不提前出",
            "选定明细": chosen_b,
        }
        # 总选定：在三族选定（若支持）与基线中取 VAL k2 最大；若都无支持则基线
        supported = []
        for fk, fs in family_sel.items():
            if fs["VAL是否支持(R3)"] == "支持":
                supported.append(fs["选定明细"])
        if supported:
            overall = max(supported, key=lambda r: r["每单净省钱_k2"])
            overall_name = overall["方案"]
        else:
            overall = base
            overall_name = "纯A1阈值出手_不提前出"

        sel_all[pool_name] = {
            "入场池": pool_name,
            "降分阈值": float(thr),
            "出手比例": pool["出手比例"],
            "VAL入场单数": int(n_e),
            "基线核对不一致数": bad,
            "VAL基线": base,
            "各族选择": family_sel,
            "总选定方案": overall_name,
            "总选定明细": overall,
            "R3说明": "相对基线 Δ每单k2>=5 且 每单k1不降 → 该族支持；总选定=支持族中VAL k2最大者，否则基线",
        }
        log("VAL选择", pool_name, "总选定", overall_name,
            "价格", family_sel["价格回归辅助"]["VAL是否支持(R3)"],
            "日期", family_sel["日期回归辅助"]["VAL是否支持(R3)"],
            "两者", family_sel["两者都用"]["VAL是否支持(R3)"])

    pd.DataFrame(val_rows_all).round(4).to_csv(
        os.path.join(OUT, "结果/VAL选门槛过程表.csv"), index=False, encoding="utf-8-sig")
    val_sel_path = os.path.join(OUT, "结果/VAL选择_先于测试集读取.json")
    json.dump({
        "R规则": [
            "R1 方案网格：价格辅助收/出、日期按预测日/晚触达提前出、两者交叉；无模型第N日必出对照",
            "R2 各族内 VAL 每单k2最大；k2差<1元并列→更高门槛数值（更保守）",
            "R3 族支持：相对纯A1基线 Δk2>=5 且 k1不降；否则该族选定=基线",
            "R4 本文件落盘并记md5后才读测试结算",
            "R5 入场行可预载；结算严格后算；路径价可跨 exclude-from，只作路径",
            "R6 通程 len(航班号)==7 已剔除；读法B dtd>=14；结算dtd>=7",
        ],
        "标签定义": {
            "价格回归": reg_report["价格回归"].get("标签定义"),
            "日期回归": reg_report["日期回归"].get("标签定义"),
            "未触达处理": f"触达降到线天数=哨兵{SENTINEL_DAYS}（截尾保留样本）",
        },
        "入场池": pools,
        "VAL选择": sel_all,
        "回归模型": {
            "价格": {"路径": price_model, "md5": md5(price_model),
                   "验证": reg_report["价格回归"].get("验证指标")},
            "日期": {"路径": date_model, "md5": md5(date_model),
                   "验证": reg_report["日期回归"].get("验证指标")},
        },
        "降模型": {"路径": DROP_MODEL, "md5": drop_md5},
    }, open(val_sel_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    val_md5 = {
        "VAL选门槛过程表.csv": md5(os.path.join(OUT, "结果/VAL选门槛过程表.csv")),
        "VAL选择_先于测试集读取.json": md5(val_sel_path),
    }
    log("VAL选择已落盘（尚未读测试结算）", val_md5)

    # ---------- 7) 测试结算 ----------
    def replay_scheme(mats, scheme_row):
        """按选定明细的方案族/门槛在 mats 上重放。"""
        P, V, D, p0 = mats["P"], mats["V"], mats["D"], mats["p0"]
        pred_drop, pred_days = mats["pred_drop"], mats["pred_days"]
        fam = scheme_row.get("方案族", "纯A1基线")
        th = scheme_row.get("门槛数值", 0)
        lab = str(scheme_row.get("门槛标签", "—"))
        if fam == "纯A1基线" or scheme_row.get("方案", "").startswith("纯A1"):
            g, l, e, h, sx = outcome_baseline(P, V, p0)
            return g, l, e, h, sx, None
        if fam == "A1+价格回归辅助收":
            msk = pred_drop >= float(th)
            g, l, e, h, sx = outcome_baseline(P, V, p0)
            return g, l, e, h, sx, msk
        if fam == "A1+价格回归辅助出":
            # 门槛标签形如 <100
            thr = float(th)
            do = pred_drop < thr
            g, l, e, h, sx = outcome_cond_early(P, V, p0, do)
            return g, l, e, h, sx, None
        if fam == "A1+日期回归按预测日出手":
            g, l, e, h, sx = outcome_date_pred_settle(P, V, p0, pred_days)
            return g, l, e, h, sx, None
        if fam == "A1+日期回归晚触达提前出":
            thr = float(th)
            do = pred_days > thr
            g, l, e, h, sx = outcome_cond_early(P, V, p0, do)
            return g, l, e, h, sx, None
        if fam == "A1+价格过滤+日期按预测日":
            msk = pred_drop >= float(th)
            g, l, e, h, sx = outcome_date_pred_settle(P, V, p0, pred_days)
            return g, l, e, h, sx, msk
        if fam == "A1+价格过滤+日期晚触达提前出":
            # 门槛数值 = th_p + th_d/100；从方案名解析
            import re
            m = re.search(r"价格>=(\d+(?:\.\d+)?)且预测天数>(\d+)", scheme_row.get("方案", ""))
            if m:
                th_p, th_d = float(m.group(1)), float(m.group(2))
            else:
                th_p, th_d = float(int(th)), 4.0
            msk = pred_drop >= th_p
            do = pred_days > th_d
            g, l, e, h, sx = outcome_cond_early(P, V, p0, do)
            return g, l, e, h, sx, msk
        # 默认基线
        g, l, e, h, sx = outcome_baseline(P, V, p0)
        return g, l, e, h, sx, None

    test_rows_all = []
    delta_rows = []
    family_test_rows = []

    for pool_name, pool in pools.items():
        thr = pool["阈值"]
        Eraw = te.loc[te["降分"] >= thr].reset_index(drop=True)
        rows, base_row, bad, n_e, mats = eval_schemes_on(Eraw, pool_name)
        test_rows_all += rows
        log("TEST基线核对", pool_name, "入场", n_e, "不一致", bad)

        # 基线指标
        g0, l0, e0, h0, sx0 = outcome_baseline(mats["P"], mats["V"], mats["p0"])
        bmet = mets(g0, l0, e0)

        # 各族测试
        fam_results = {}
        for fam_key in ("价格回归辅助", "日期回归辅助", "两者都用"):
            fs = sel_all[pool_name]["各族选择"][fam_key]
            detail = fs["选定明细"]
            # 若选定就是基线方案名
            if fs["选定方案"].startswith("纯A1") or detail.get("方案族") == "纯A1基线":
                g, l, e, h, sx, msk = g0, l0, e0, h0, sx0, None
                smet = bmet
                used = "纯A1阈值出手_不提前出"
            else:
                g, l, e, h, sx, msk = replay_scheme(mats, detail)
                smet = mets(g, l, e, msk)
                used = detail.get("方案")
            fam_results[fam_key] = {
                "VAL判定": fs["VAL是否支持(R3)"],
                "选定方案": used,
                "测试": smet,
                "Δ每单k1": smet["每单净省钱_k1"] - bmet["每单净省钱_k1"],
                "Δ每单k2": smet["每单净省钱_k2"] - bmet["每单净省钱_k2"],
                "Δ合计k1": smet["净省钱合计_k1"] - bmet["净省钱合计_k1"],
                "Δ合计k2": smet["净省钱合计_k2"] - bmet["净省钱合计_k2"],
            }
            family_test_rows.append({
                "入场池": pool_name,
                "降分阈值": thr,
                "出手比例": pool["出手比例"],
                "测试入场单数": bmet["条数"] if msk is None else int(msk.sum()),
                "方案族": fam_key,
                "VAL判定": fs["VAL是否支持(R3)"],
                "选定方案": used,
                "是否族支持后的选定": fs["VAL是否支持(R3)"] == "支持",
                "基线_每单k1": bmet["每单净省钱_k1"],
                "基线_每单k2": bmet["每单净省钱_k2"],
                "基线_合计k1": bmet["净省钱合计_k1"],
                "基线_合计k2": bmet["净省钱合计_k2"],
                "方案_每单k1": smet["每单净省钱_k1"],
                "方案_每单k2": smet["每单净省钱_k2"],
                "方案_合计k1": smet["净省钱合计_k1"],
                "方案_合计k2": smet["净省钱合计_k2"],
                "Δ每单k1": smet["每单净省钱_k1"] - bmet["每单净省钱_k1"],
                "Δ每单k2": smet["每单净省钱_k2"] - bmet["每单净省钱_k2"],
                "Δ合计k1": smet["净省钱合计_k1"] - bmet["净省钱合计_k1"],
                "Δ合计k2": smet["净省钱合计_k2"] - bmet["净省钱合计_k2"],
                "提前出占比": smet.get("提前出占比", 0),
            })

        # 总选定
        overall = sel_all[pool_name]["总选定明细"]
        if sel_all[pool_name]["总选定方案"].startswith("纯A1"):
            g, l, e, h, sx, msk = g0, l0, e0, h0, sx0, None
            smet = bmet
            used = "纯A1阈值出手_不提前出"
            is_sel_overlay = False
        else:
            g, l, e, h, sx, msk = replay_scheme(mats, overall)
            smet = mets(g, l, e, msk)
            used = overall.get("方案")
            is_sel_overlay = True

        delta_rows.append({
            "入场池": pool_name,
            "降分阈值": thr,
            "出手比例": pool["出手比例"],
            "测试入场单数_基线": bmet["条数"],
            "测试入场单数_总选定": bmet["条数"] if msk is None else int(msk.sum()),
            "总选定方案": used,
            "VAL总选定是否叠回归": "是" if is_sel_overlay else "否（保持纯A1）",
            "基线_每单k1": bmet["每单净省钱_k1"],
            "基线_每单k2": bmet["每单净省钱_k2"],
            "基线_合计k1": bmet["净省钱合计_k1"],
            "基线_合计k2": bmet["净省钱合计_k2"],
            "总选定_每单k1": smet["每单净省钱_k1"],
            "总选定_每单k2": smet["每单净省钱_k2"],
            "总选定_合计k1": smet["净省钱合计_k1"],
            "总选定_合计k2": smet["净省钱合计_k2"],
            "Δ每单k1": smet["每单净省钱_k1"] - bmet["每单净省钱_k1"],
            "Δ每单k2": smet["每单净省钱_k2"] - bmet["每单净省钱_k2"],
            "Δ合计k1": smet["净省钱合计_k1"] - bmet["净省钱合计_k1"],
            "Δ合计k2": smet["净省钱合计_k2"] - bmet["净省钱合计_k2"],
            "价格族_VAL": fam_results["价格回归辅助"]["VAL判定"],
            "价格族_选定": fam_results["价格回归辅助"]["选定方案"],
            "价格族_Δk1": fam_results["价格回归辅助"]["Δ每单k1"],
            "价格族_Δk2": fam_results["价格回归辅助"]["Δ每单k2"],
            "日期族_VAL": fam_results["日期回归辅助"]["VAL判定"],
            "日期族_选定": fam_results["日期回归辅助"]["选定方案"],
            "日期族_Δk1": fam_results["日期回归辅助"]["Δ每单k1"],
            "日期族_Δk2": fam_results["日期回归辅助"]["Δ每单k2"],
            "两者族_VAL": fam_results["两者都用"]["VAL判定"],
            "两者族_选定": fam_results["两者都用"]["选定方案"],
            "两者族_Δk1": fam_results["两者都用"]["Δ每单k1"],
            "两者族_Δk2": fam_results["两者都用"]["Δ每单k2"],
            "尖端基线对齐备注": (
                "应对齐S3 A1测试_阈值出手：条数1205、k1≈194.45、k2≈175.22"
                if pool_name.startswith("尖端") else ""
            ),
        })

    pd.DataFrame(test_rows_all).round(4).to_csv(
        os.path.join(OUT, "结果/测试结果表_读法B_dtd14_全方案.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(family_test_rows).round(4).to_csv(
        os.path.join(OUT, "结果/测试_各族选定_相对基线.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(delta_rows).round(4).to_csv(
        os.path.join(OUT, "结果/相对基线_Δk1Δk2_两档摘要.csv"), index=False, encoding="utf-8-sig")

    # 通程核对
    tongcheng = int(con.execute(f"""
      SELECT count(*) FROM read_parquet('{lab_path}')
      WHERE length(CAST(航班号 AS VARCHAR))=7
    """).fetchone()[0])

    rep = {
        "性质": "仅离线影子对照；不是过门结论，不上线，不动现网；通过≠上线≠真收≠改阈≠开启线上推理服务",
        "授权": "Yangyang t1389（经经理 Grok 转达）：价格回归+日期回归离线，叠A1看赚不赚",
        "红线": "未碰线上推理服务/未上线/未真收/未写服务器库/未重启[REDACTED_SERVICE]/未读ana_extract·pull8001/未打印store.json；seats不入模；未并入禁止42列",
        "降模型": {"路径": DROP_MODEL, "md5": drop_md5, "特征数": 46},
        "回归模型": reg_report,
        "切分": {"训练": TRAIN_DAYS, "验证": VAL_DAYS, "测试": TEST_DAYS,
                "gap要求": ">=7", "exclude_from": EXCLUDE_FROM, "cutoff": CUTOFF,
                "采集窗切换日": SWITCH, "主评估dtd": "0..30", "读法B": DTD_MIN, "窗": W,
                "结算最少距起飞天数": SETTLE_MIN},
        "通程剔除核对_标签中len7": tongcheng,
        "入场池": pools,
        "VAL选择文件md5(落盘后、读测试结算前)": val_md5,
        "VAL选择": {k: {"总选定方案": v["总选定方案"],
                      "价格": v["各族选择"]["价格回归辅助"]["VAL是否支持(R3)"],
                      "日期": v["各族选择"]["日期回归辅助"]["VAL是否支持(R3)"],
                      "两者": v["各族选择"]["两者都用"]["VAL是否支持(R3)"]}
                   for k, v in sel_all.items()},
        "测试摘要_相对基线": delta_rows,
        "耗时秒": round(time.time() - T0, 1),
    }
    json.dump(rep, open(os.path.join(OUT, "结果/价格日期回归对照报告_S3A1.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    log("完成主计算", round(time.time() - T0, 1), "s", "通程标签中len7", tongcheng)
    print("DONE")


if __name__ == "__main__":
    main()
