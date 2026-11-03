#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S0→S1 主表日更并入（幂等）。

默认目标日 = Asia/Shanghai 昨天；支持 --day YYYY-MM-DD。
完整日：incremental 中该日末批小时位 >= 19；否则 exit 10（可预期跳过）。
幂等：S0 与 S1 的 max(观察日) 均 >= 目标日 → exit 0。
红线：不碰线上推理服务 / 不写服务器库 / 不真收 / 不上线 / 只读 AF incremental。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zoneinfo
from pathlib import Path

import duckdb

TZ = zoneinfo.ZoneInfo("Asia/Shanghai")
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
HERE = ROOT / "S0S1_日更并入"
S0_DIR = ROOT / "S0_底表"
S1_DIR = ROOT / "S1_特征"
S0_DATA = S0_DIR / "数据"
S0_CHK = S0_DIR / "核对"
S1_DATA = S1_DIR / "数据"
S1_MID = S1_DIR / "中间"
S0_MAIN = S0_DATA / "主表_全航司日价_dtd0_30.parquet"
S0_CTRL = S0_DATA / "对照表_全航司日价_dtd31_50.parquet"
S0_BACKFILL = S0_DATA / "回补表_全航司日价_另存.parquet"
S1_MAT = S1_DATA / "特征矩阵_全航司_dtd0_30.parquet"
S0_BUILD = S0_DIR / "脚本" / "构建_全航司日价底表_S0.py"
S1_BUILD = S1_DIR / "脚本" / "构建特征_S1.py"
AF_INC = Path(os.environ["AF_INCREMENTAL_DIR"])
VENV_PY = Path(os.environ.get("VENV_PYTHON", "python3"))
LOG_DIR = HERE / "日志"
STATE_DIR = HERE / "状态"
TMP_DIR = HERE / "_tmp"
BACKUP_KEEP = 3  # 日更前小备份保留最近 N 份

# 退出码
EXIT_OK = 0
EXIT_INCOMPLETE = 10  # 日未走完，可预期跳过
EXIT_FAIL = 1

for p in (LOG_DIR, STATE_DIR, TMP_DIR, HERE / "脚本"):
    p.mkdir(parents=True, exist_ok=True)


def now_sh() -> dt.datetime:
    return dt.datetime.now(TZ)


def log(msg: str, fp=None):
    line = f"{now_sh().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if fp:
        fp.write(line + "\n")
        fp.flush()


def md5file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description="S0S1 日更并入（幂等）")
    ap.add_argument("--day", help="目标观察日 YYYY-MM-DD（默认上海昨天）")
    ap.add_argument("--force", action="store_true", help="忽略幂等跳过，强制重建（仍要求完整日）")
    return ap.parse_args()


def target_day(args) -> dt.date:
    if args.day:
        return dt.date.fromisoformat(args.day)
    today = now_sh().date()
    return today - dt.timedelta(days=1)


def list_batches_for_day(day: dt.date) -> list[str]:
    """只读 incremental 目录名；仅计有 _SUCCESS 的批。"""
    prefix = day.strftime("%Y%m%d")
    out = []
    if not AF_INC.is_dir():
        return out
    for d in sorted(AF_INC.glob(f"batch={prefix}*")):
        bid = d.name.replace("batch=", "")
        if not re.fullmatch(r"\d{12}", bid):
            continue
        if not (d / "_SUCCESS").exists():
            continue
        out.append(bid)
    return out


def day_complete(day: dt.date) -> tuple[bool, str | None, list[str]]:
    batches = list_batches_for_day(day)
    if not batches:
        return False, None, []
    last = max(batches)
    hour = int(last[8:10])
    return hour >= 19, last, batches


def max_obs_day(parquet: Path) -> dt.date | None:
    if not parquet.exists():
        return None
    con = duckdb.connect()
    try:
        v = con.execute(f"SELECT max(CAST(观察日 AS DATE)) FROM read_parquet('{parquet}')").fetchone()[0]
        if v is None:
            return None
        if isinstance(v, dt.datetime):
            return v.date()
        return v
    finally:
        con.close()


def row_count(parquet: Path) -> int:
    con = duckdb.connect()
    try:
        return int(con.execute(f"SELECT count(*) FROM read_parquet('{parquet}')").fetchone()[0])
    finally:
        con.close()


def per_day_counts(parquet: Path, day_lt: dt.date | None = None) -> dict[str, int]:
    con = duckdb.connect()
    try:
        where = f"WHERE CAST(观察日 AS DATE) < DATE '{day_lt.isoformat()}'" if day_lt else ""
        rows = con.execute(
            f"SELECT CAST(观察日 AS VARCHAR), count(*) FROM read_parquet('{parquet}') {where} GROUP BY 1 ORDER BY 1"
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows}
    finally:
        con.close()


def patch_s0_obs_to(target: dt.date, logfp) -> str:
    """把构建脚本 OBS_TO 改成目标日；返回旧值。"""
    text = S0_BUILD.read_text(encoding="utf-8")
    m = re.search(r'OBS_FROM,\s*OBS_TO,\s*SWITCH\s*=\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("无法解析 S0 构建脚本 OBS_TO")
    old = m.group(2)
    new_to = target.isoformat()
    if old == new_to:
        log(f"S0 OBS_TO 已是 {new_to}", logfp)
        return old
    new_text, n = re.subn(
        r'(OBS_FROM,\s*OBS_TO,\s*SWITCH\s*=\s*"[^"]+",\s*")[^"]+(")',
        rf"\g<1>{new_to}\2",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("替换 OBS_TO 失败")
    # 同步注释里的「未走完」类措辞不强改；只改常量
    S0_BUILD.write_text(new_text, encoding="utf-8")
    log(f"S0 OBS_TO: {old} → {new_to}", logfp)
    return old


def run_s0_build(out_dir: Path, logfp) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"S0构建_{now_sh().strftime('%Y%m%d_%H%M%S')}.log"
    log(f"跑 S0 构建 → {out_dir}", logfp)
    with open(log_path, "w", encoding="utf-8") as lf:
        r = subprocess.run(
            [str(VENV_PY), "-u", str(S0_BUILD), str(out_dir)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if r.returncode != 0:
        raise RuntimeError(f"S0 构建失败 rc={r.returncode}，日志 {log_path}")
    log(f"S0 构建完成，日志 {log_path.name}", logfp)


def validate_s0(new_main: Path, old_main: Path | None, target: dt.date, logfp) -> dict:
    con = duckdb.connect()
    try:
        n = con.execute(f"SELECT count(*) FROM read_parquet('{new_main}')").fetchone()[0]
        mx = con.execute(f"SELECT max(CAST(观察日 AS DATE)) FROM read_parquet('{new_main}')").fetchone()[0]
        if isinstance(mx, dt.datetime):
            mx = mx.date()
        if mx != target:
            raise RuntimeError(f"新主表 max(观察日)={mx} ≠ 目标 {target}")
        tong = con.execute(
            f"SELECT count(*) FROM read_parquet('{new_main}') WHERE length(航班号)=7"
        ).fetchone()[0]
        if tong:
            raise RuntimeError(f"新主表有通程 {tong} 行")
        dup = con.execute(
            f"""SELECT count(*) FROM (
                 SELECT 观察日,航班号,航线,起飞日 FROM read_parquet('{new_main}')
                 GROUP BY ALL HAVING count(*)>1)"""
        ).fetchone()[0]
        if dup:
            raise RuntimeError(f"新主表重复键 {dup}")
        hui = con.execute(
            f"SELECT count(*) FROM read_parquet('{new_main}') WHERE 是否回补键='是'"
        ).fetchone()[0]
        if hui:
            raise RuntimeError(f"新主表含回补键 {hui}")

        # 旧日行数不变
        if old_main is not None and old_main.exists():
            old_c = per_day_counts(old_main, day_lt=target)
            new_c = per_day_counts(new_main, day_lt=target)
            if old_c != new_c:
                # 找差异
                keys = sorted(set(old_c) | set(new_c))
                diffs = [(k, old_c.get(k), new_c.get(k)) for k in keys if old_c.get(k) != new_c.get(k)]
                raise RuntimeError(f"旧观察日行数变化：{diffs[:5]}")
            log(f"校验：旧日行数 diff=0（{len(old_c)} 个观察日）", logfp)

        n_target = con.execute(
            f"SELECT count(*) FROM read_parquet('{new_main}') WHERE CAST(观察日 AS DATE)=DATE '{target.isoformat()}'"
        ).fetchone()[0]
        log(f"校验通过：总行={n} 目标日行={n_target} 无通程/无重复/无回补", logfp)
        return {"rows": int(n), "target_rows": int(n_target), "max_day": target.isoformat()}
    finally:
        con.close()


def prune_backups(parent: Path, prefix: str, keep: int, logfp):
    cands = sorted(
        [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
    )
    while len(cands) > keep:
        old = cands.pop(0)
        shutil.rmtree(old)
        log(f"删旧备份 {old}", logfp)


def backup_and_replace_s0(rebuild_dir: Path, target: dt.date, logfp) -> dict:
    day_tag = target.strftime("%Y%m%d")
    bak = S0_DATA / f"_备份_日更前_{day_tag}"
    bak.mkdir(parents=True, exist_ok=True)
    chk_bak = S0_CHK / f"_备份_日更前_{day_tag}"
    chk_bak.mkdir(parents=True, exist_ok=True)

    files = [
        ("主表_全航司日价_dtd0_30.parquet", S0_MAIN, rebuild_dir / "数据" / "主表_全航司日价_dtd0_30.parquet"),
        ("对照表_全航司日价_dtd31_50.parquet", S0_CTRL, rebuild_dir / "数据" / "对照表_全航司日价_dtd31_50.parquet"),
        ("回补表_全航司日价_另存.parquet", S0_BACKFILL, rebuild_dir / "数据" / "回补表_全航司日价_另存.parquet"),
    ]
    md5s = {}
    for name, dst, src in files:
        if not src.exists():
            raise RuntimeError(f"重建产物缺失 {src}")
        if dst.exists():
            shutil.copy2(dst, bak / name)
            md5s[f"旧_{name}"] = md5file(dst)
        # 原子替换：先拷到同目录临时名再 replace
        tmp = dst.with_suffix(dst.suffix + f".tmp_{day_tag}")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        md5s[f"新_{name}"] = md5file(dst)
        log(f"替换 S0 {name} md5={md5s[f'新_{name}']}", logfp)

    # 核对 csv/json（有则替换）
    new_chk = rebuild_dir / "核对"
    if new_chk.is_dir():
        for f in new_chk.iterdir():
            if f.is_file():
                if (S0_CHK / f.name).exists():
                    shutil.copy2(S0_CHK / f.name, chk_bak / f.name)
                shutil.copy2(f, S0_CHK / f.name)

    prune_backups(S0_DATA, "_备份_日更前_", BACKUP_KEEP, logfp)
    prune_backups(S0_CHK, "_备份_日更前_", BACKUP_KEEP, logfp)
    return md5s


def clear_s1_done(logfp):
    """清 _DONE_01..06，保留 _DONE_00_清单；顺带清块 parquet 以免脏读。"""
    for p in S1_MID.glob("_DONE_*"):
        if p.name == "_DONE_00_清单":
            continue
        p.unlink()
        log(f"删 {p.name}", logfp)
    for pat in ("块*.parquet", "tmp_*.parquet"):
        for p in S1_MID.glob(pat):
            p.unlink()
            log(f"删中间 {p.name}", logfp)


def run_s1_build(logfp) -> None:
    log_path = LOG_DIR / f"S1构建_{now_sh().strftime('%Y%m%d_%H%M%S')}.log"
    log(f"跑 S1 全量重跑 → 日志 {log_path.name}", logfp)
    # 在 S1 自己的构建.log 也留痕迹
    marker = S1_DIR / "日志" / "构建.log"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "a", encoding="utf-8") as mf:
        mf.write(f"---- 日更并入 {now_sh().isoformat()} ----\n")
    with open(log_path, "w", encoding="utf-8") as lf:
        r = subprocess.run(
            [str(VENV_PY), "-u", str(S1_BUILD)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(S1_DIR),
            check=False,
        )
    if r.returncode != 0:
        raise RuntimeError(f"S1 构建失败 rc={r.returncode}，日志 {log_path}")
    log(f"S1 构建完成，日志 {log_path.name}", logfp)


def validate_s1(old_mat: Path | None, target: dt.date, logfp) -> dict:
    if not S1_MAT.exists():
        raise RuntimeError("S1 矩阵不存在")
    con = duckdb.connect()
    try:
        n, mx = con.execute(
            f"SELECT count(*), max(CAST(观察日 AS DATE)) FROM read_parquet('{S1_MAT}')"
        ).fetchone()
        if isinstance(mx, dt.datetime):
            mx = mx.date()
        if mx != target:
            raise RuntimeError(f"S1 max(观察日)={mx} ≠ 目标 {target}")
        tong = con.execute(
            f"SELECT count(*) FROM read_parquet('{S1_MAT}') WHERE length(航班号)=7"
        ).fetchone()[0]
        if tong:
            raise RuntimeError(f"S1 有通程 {tong}")
        dup = con.execute(
            f"""SELECT count(*) FROM (
                 SELECT 观察日,航班号,航线,起飞日 FROM read_parquet('{S1_MAT}')
                 GROUP BY ALL HAVING count(*)>1)"""
        ).fetchone()[0]
        if dup:
            raise RuntimeError(f"S1 重复键 {dup}")
        # 与 S0 行数一致
        n0 = row_count(S0_MAIN)
        if int(n) != n0:
            raise RuntimeError(f"S1 行数 {n} ≠ S0 行数 {n0}")
        if old_mat and old_mat.exists():
            old_c = per_day_counts(old_mat, day_lt=target)
            new_c = per_day_counts(S1_MAT, day_lt=target)
            if old_c != new_c:
                keys = sorted(set(old_c) | set(new_c))
                diffs = [(k, old_c.get(k), new_c.get(k)) for k in keys if old_c.get(k) != new_c.get(k)]
                raise RuntimeError(f"S1 旧日行数变化：{diffs[:5]}")
            log(f"S1 校验：旧日行数 diff=0", logfp)
        log(f"S1 校验通过：总行={n} max={mx}", logfp)
        return {"rows": int(n), "max_day": target.isoformat()}
    finally:
        con.close()


def backup_s1(target: dt.date, logfp) -> Path | None:
    if not S1_MAT.exists():
        return None
    day_tag = target.strftime("%Y%m%d")
    bak_dir = S1_DATA / f"_备份_日更前_{day_tag}"
    bak_dir.mkdir(parents=True, exist_ok=True)
    dest = bak_dir / "特征矩阵_全航司_dtd0_30.parquet"
    shutil.copy2(S1_MAT, dest)
    log(f"S1 备份 → {dest} md5={md5file(dest)}", logfp)
    prune_backups(S1_DATA, "_备份_日更前_", BACKUP_KEEP, logfp)
    return dest


def write_status(payload: dict):
    path = STATE_DIR / "上次成功.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_说明(info: dict):
    path = HERE / "增量说明_最新.md"
    lines = [
        f"# S0S1 日更并入 · 最新",
        "",
        f"- 更新时间：{info.get('finished_at')}（Asia/Shanghai）",
        f"- 目标观察日：{info.get('target_day')}",
        f"- 结果：{info.get('result')}",
        f"- 末批：{info.get('last_batch')}",
        f"- 当日批次：{info.get('batches')}",
        "",
        "## md5 / 行数",
        "",
        f"- S0 主表：`{info.get('s0_md5')}` 行数 {info.get('s0_rows')}",
        f"- S1 矩阵：`{info.get('s1_md5')}` 行数 {info.get('s1_rows')}",
        "",
        "## 红线",
        "",
        "- 未碰线上推理服务；未写服务器库；未真收；未上线；AF incremental 只读。",
        "",
    ]
    if info.get("note"):
        lines.extend(["## 备注", "", info["note"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    target = target_day(args)
    run_id = now_sh().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"日更_{run_id}.log"
    t0 = time.time()

    with open(log_path, "w", encoding="utf-8") as fp:
        log(f"==== S0S1 日更开始 target={target.isoformat()} ====", fp)
        log(f"日志 {log_path}", fp)

        complete, last_batch, batches = day_complete(target)
        log(f"完整日判定：batches={batches} last={last_batch} complete={complete}", fp)

        if not complete:
            msg = (
                f"日未走完，跳过：目标日 {target.isoformat()} 现有批 {batches or '无'}，"
                f"末批={last_batch or '无'}（需小时位>=19）"
            )
            log(msg, fp)
            # 不改 增量说明_最新.md / 上次成功.json（保留上次成功口径）；仅写日志
            log(f"exit {EXIT_INCOMPLETE}（可预期跳过）", fp)
            return EXIT_INCOMPLETE

        s0_max = max_obs_day(S0_MAIN)
        s1_max = max_obs_day(S1_MAT)
        log(f"当前 max：S0={s0_max} S1={s1_max}", fp)

        need_s0 = args.force or s0_max is None or s0_max < target
        need_s1 = args.force or s1_max is None or s1_max < target

        if not need_s0 and not need_s1:
            msg = f"已是最新，无操作：S0/S1 max(观察日) 均 >= {target.isoformat()}"
            log(msg, fp)
            info = {
                "finished_at": now_sh().strftime("%F %T"),
                "target_day": target.isoformat(),
                "result": "幂等跳过_已是最新",
                "last_batch": last_batch,
                "batches": " ".join(batches),
                "s0_md5": md5file(S0_MAIN),
                "s1_md5": md5file(S1_MAT),
                "s0_rows": row_count(S0_MAIN),
                "s1_rows": row_count(S1_MAT),
                "note": msg,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            write_说明(info)
            write_status(
                {
                    "target_day": target.isoformat(),
                    "finished_at": info["finished_at"],
                    "result": info["result"],
                    "s0_md5": info["s0_md5"],
                    "s1_md5": info["s1_md5"],
                    "last_batch": last_batch,
                }
            )
            log(f"exit {EXIT_OK}", fp)
            return EXIT_OK

        # --- S0 ---
        old_s0_md5 = md5file(S0_MAIN) if S0_MAIN.exists() else None
        old_s0_rows = row_count(S0_MAIN) if S0_MAIN.exists() else None
        if need_s0:
            log("S0 落后，开始重建…", fp)
            rebuild = TMP_DIR / f"rebuild_s0_{target.strftime('%Y%m%d')}_{run_id}"
            if rebuild.exists():
                shutil.rmtree(rebuild)
            patch_s0_obs_to(target, fp)
            try:
                run_s0_build(rebuild, fp)
                new_main = rebuild / "数据" / "主表_全航司日价_dtd0_30.parquet"
                validate_s0(new_main, S0_MAIN if S0_MAIN.exists() else None, target, fp)
                backup_and_replace_s0(rebuild, target, fp)
            finally:
                # 清理临时重建目录（大数据）
                if rebuild.exists():
                    shutil.rmtree(rebuild, ignore_errors=True)
                    log(f"清理临时目录 {rebuild.name}", fp)
        else:
            log("S0 已覆盖目标日，跳过 S0", fp)

        # --- S1 ---
        old_s1_md5 = md5file(S1_MAT) if S1_MAT.exists() else None
        old_s1_rows = row_count(S1_MAT) if S1_MAT.exists() else None
        # 若刚更了 S0，或 S1 落后，都要滚 S1
        if need_s0 or need_s1:
            log("滚 S1 特征矩阵…", fp)
            old_bak = backup_s1(target, fp)
            clear_s1_done(fp)
            run_s1_build(fp)
            validate_s1(old_bak, target, fp)
        else:
            log("S1 已覆盖目标日，跳过 S1", fp)

        new_s0_md5 = md5file(S0_MAIN)
        new_s1_md5 = md5file(S1_MAT)
        new_s0_rows = row_count(S0_MAIN)
        new_s1_rows = row_count(S1_MAT)

        # 轻量 md5 清单追加
        (S0_DIR / "md5清单_日更最新.txt").write_text(
            f"# S0 日更 {now_sh().strftime('%F %T')} Asia/Shanghai target={target}\n"
            f"{new_s0_md5}  数据/主表_全航司日价_dtd0_30.parquet\n"
            f"{md5file(S0_CTRL)}  数据/对照表_全航司日价_dtd31_50.parquet\n"
            f"{md5file(S0_BACKFILL)}  数据/回补表_全航司日价_另存.parquet\n",
            encoding="utf-8",
        )

        info = {
            "finished_at": now_sh().strftime("%F %T"),
            "target_day": target.isoformat(),
            "result": "成功并入",
            "last_batch": last_batch,
            "batches": " ".join(batches),
            "s0_md5": new_s0_md5,
            "s1_md5": new_s1_md5,
            "s0_rows": new_s0_rows,
            "s1_rows": new_s1_rows,
            "note": (
                f"旧 S0 md5={old_s0_md5} rows={old_s0_rows} → 新 {new_s0_md5} rows={new_s0_rows}；"
                f"旧 S1 md5={old_s1_md5} rows={old_s1_rows} → 新 {new_s1_md5} rows={new_s1_rows}；"
                f"耗时 {round(time.time()-t0,1)}s"
            ),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        write_说明(info)
        write_status(
            {
                "target_day": target.isoformat(),
                "finished_at": info["finished_at"],
                "result": info["result"],
                "s0_md5": new_s0_md5,
                "s1_md5": new_s1_md5,
                "s0_rows": new_s0_rows,
                "s1_rows": new_s1_rows,
                "last_batch": last_batch,
                "elapsed_sec": info["elapsed_sec"],
            }
        )
        log(f"==== 完成 {info['note']} ====", fp)
        log(f"exit {EXIT_OK}", fp)
        return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # 真实事故
        err_log = LOG_DIR / f"日更失败_{now_sh().strftime('%Y%m%d_%H%M%S')}.log"
        msg = f"FAIL: {type(e).__name__}: {e}"
        print(msg, flush=True)
        err_log.write_text(msg + "\n", encoding="utf-8")
        sys.exit(EXIT_FAIL)
