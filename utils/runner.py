"""多 GPU seed-parallel 实验调度器。

设计原则：
- 每个 seed 由一个独立子进程承担，独占一张 GPU；
- 通过 CUDA_VISIBLE_DEVICES 限定可见设备，避免 PyTorch 在子进程内做多卡复制；
- 进程数 = min(len(seeds), len(gpus))；若无可用 GPU 则退化为顺序 CPU 执行；
- 子进程失败不影响其他 seed，错误写入 logs/。

实验调用方约定：
    run_seeds_parallel(seeds, gpus, worker_fn, config_template)
    worker_fn(seed, gpu_id, config) -> dict（必须可 pickle）
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger("runner")


def detect_gpus() -> List[int]:
    """返回可见 GPU id 列表（不依赖 torch 已 import）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _worker_entry(seed: int, gpu_id: Optional[int], worker_fn: Callable, config: Dict[str, Any],
                  result_path: str, log_path: str) -> None:
    """子进程入口：绑卡、调用 worker_fn、写结果。"""
    if gpu_id is not None and gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 重新 import torch 以读取 CUDA_VISIBLE_DEVICES
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES 后唯一卡号为 0
    except Exception:
        pass

    try:
        cfg = dict(config)
        cfg["seed"] = int(seed)
        cfg["gpu_id"] = int(gpu_id) if gpu_id is not None else -1
        result = worker_fn(seed, gpu_id, cfg)
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as fp:
            json.dump({"seed": seed, "gpu_id": gpu_id, "ok": True, "result": result},
                      fp, ensure_ascii=False, default=str)
    except Exception as e:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fp:
            fp.write(f"seed={seed} gpu={gpu_id}\n")
            fp.write(traceback.format_exc())
        with open(result_path, "w", encoding="utf-8") as fp:
            json.dump({"seed": seed, "gpu_id": gpu_id, "ok": False, "error": str(e)},
                      fp, ensure_ascii=False, default=str)


def run_seeds_parallel(
    seeds: List[int],
    worker_fn: Callable,
    config_template: Dict[str, Any],
    output_dir: Path,
    gpus: Optional[List[int]] = None,
    max_parallel: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """并行跑多个 seed。每个 seed 独占一卡（轮转分配）。

    Args:
        seeds: 要跑的 seed 列表。
        worker_fn: (seed, gpu_id, config) -> dict 的可 pickle 训练函数。
        config_template: 公共配置（每个 seed 共享，仅 seed/gpu_id 不同）。
        output_dir: 写 per-seed 的 result/log 到 output_dir/_runner/。
        gpus: 可用 GPU id 列表；None=自动检测；[]=强制 CPU。
        max_parallel: 同时并行进程上限；默认 = len(gpus) 或 1。

    Returns:
        每个 seed 的结果 dict 列表（含 ok/error 字段）。
    """
    output_dir = Path(output_dir)
    runner_dir = output_dir / "_runner"
    runner_dir.mkdir(parents=True, exist_ok=True)

    if gpus is None:
        gpus = detect_gpus()
    if not gpus:
        LOGGER.warning("No GPU detected, falling back to CPU-only sequential mode.")
        gpus = [-1]

    if max_parallel is None:
        max_parallel = max(1, len(gpus))

    LOGGER.info(f"run_seeds_parallel: seeds={len(seeds)} gpus={gpus} max_parallel={max_parallel}")

    ctx = mp.get_context("spawn")
    pending = list(enumerate(seeds))
    active: List[Any] = []  # list of (proc, seed, result_path)

    def _launch(idx: int, seed: int) -> None:
        gpu_id = gpus[idx % len(gpus)] if gpus[0] != -1 else -1
        result_path = runner_dir / f"seed_{seed}_result.json"
        log_path = runner_dir / f"seed_{seed}_error.log"
        proc = ctx.Process(
            target=_worker_entry,
            args=(int(seed), gpu_id, worker_fn, config_template, str(result_path), str(log_path)),
        )
        proc.start()
        active.append((proc, seed, result_path))
        LOGGER.info(f"  launched seed={seed} on gpu={gpu_id} (pid={proc.pid})")

    # 启动初始批
    while pending and len(active) < max_parallel:
        idx, seed = pending.pop(0)
        _launch(idx, seed)

    results: List[Dict[str, Any]] = []
    while active:
        # 轮询完成的进程
        still: List[Any] = []
        for proc, seed, result_path in active:
            if proc.is_alive():
                still.append((proc, seed, result_path))
                continue
            proc.join()
            try:
                with open(result_path, "r", encoding="utf-8") as fp:
                    results.append(json.load(fp))
            except Exception as e:
                results.append({"seed": seed, "ok": False, "error": f"missing result: {e}"})
            LOGGER.info(f"  finished seed={seed} (exit={proc.exitcode})")
        active = still
        # 启动新进程填补空位
        while pending and len(active) < max_parallel:
            idx, seed = pending.pop(0)
            _launch(idx, seed)
        if active:
            for proc, _, _ in active:
                proc.join(timeout=1.0)
                break  # 只等一个，回到外循环

    # 按 seed 排序保持稳定输出
    results.sort(key=lambda r: int(r.get("seed", 0)))

    summary_path = runner_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump({"seeds": seeds, "gpus": gpus, "results": results},
                  fp, ensure_ascii=False, indent=2, default=str)
    LOGGER.info(f"runner summary: {summary_path}")
    return results


def run_seeds_sequential(
    seeds: List[int],
    worker_fn: Callable,
    config_template: Dict[str, Any],
    output_dir: Path,
    gpu_id: int = -1,
) -> List[Dict[str, Any]]:
    """单进程顺序版本（调试用，不 spawn 子进程）。"""
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        except Exception:
            pass
    output_dir = Path(output_dir)
    runner_dir = output_dir / "_runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg = dict(config_template)
        cfg["seed"] = int(seed)
        cfg["gpu_id"] = int(gpu_id)
        try:
            res = worker_fn(seed, gpu_id if gpu_id >= 0 else None, cfg)
            results.append({"seed": seed, "ok": True, "result": res})
        except Exception as e:
            traceback.print_exc()
            results.append({"seed": seed, "ok": False, "error": str(e)})
    with (runner_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump({"seeds": seeds, "results": results}, fp, ensure_ascii=False, indent=2, default=str)
    return results
