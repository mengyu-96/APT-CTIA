"""时间与资源统计工具。

提供：
- TimeLogger：聚合所有模块时间，最终写 time_log.json
- Timer：单次 wall-clock 测量
- count_params / count_flops / peak_vram / peak_ram

设计：调用方在训练管道各阶段调用 logger.section('xxx') 上下文，
最后 logger.save(path) 一次性落盘。
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _local_time_metadata() -> dict[str, str]:
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    if offset is None:
        offset_text = ""
    else:
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        total_minutes = abs(total_minutes)
        hours, minutes = divmod(total_minutes, 60)
        offset_text = f"UTC{sign}{hours:02d}:{minutes:02d}"
    tz_name = now.tzname() or ""
    return {
        "experiment_date": now.date().isoformat(),
        "timezone": offset_text or tz_name,
        "timezone_name": tz_name,
    }


def peak_vram_mb(device: Optional[Any] = None) -> float:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    except Exception:
        return 0.0


def reset_vram_peak(device: Optional[Any] = None) -> None:
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def peak_ram_mb() -> float:
    if not _HAS_PSUTIL:
        return 0.0
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        return 0.0


def count_params(model) -> Dict[str, int]:
    if not _HAS_TORCH:
        return {"total_params": 0, "trainable_params": 0}
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def count_flops_safe(model, sample_input_fn) -> Optional[int]:
    """尝试统计单次前向 FLOPs；失败返回 None。

    优先用 PyTorch 内置 FlopCounterMode（2.0+），否则尝试 thop，否则返回 None。
    sample_input_fn() 应返回一个能直接传入 model(*inputs) 的元组。
    """
    if not _HAS_TORCH:
        return None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        inputs = sample_input_fn()
        with FlopCounterMode(display=False) as flop_counter:
            with torch.no_grad():
                model(*inputs)
        return int(flop_counter.get_total_flops())
    except Exception:
        pass
    try:
        from thop import profile  # type: ignore
        model.eval()
        inputs = sample_input_fn()
        flops, _ = profile(model, inputs=inputs, verbose=False)
        return int(flops)
    except Exception:
        return None


@dataclass
class TimeLogger:
    """实验过程时间与资源采集器。

    用法：
        logger = TimeLogger(experiment_id="RQ1_RGAT_AADM_seed42",
                            dataset="AADM", model="RGAT", seed=42)
        with logger.section("preprocess"):
            ...
        logger.add_epoch_time(train=2.3, val=0.5, data_load=0.1)
        logger.add("inference_per_sample_ms", 4.2)
        logger.finalize()
        logger.save(path)
    """
    experiment_id: str = ""
    dataset: str = ""
    model: str = ""
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    sections: Dict[str, float] = field(default_factory=dict)
    epoch_train_s: List[float] = field(default_factory=list)
    epoch_val_s: List[float] = field(default_factory=list)
    epoch_data_load_s: List[float] = field(default_factory=list)
    forward_ms_samples: List[float] = field(default_factory=list)
    backward_ms_samples: List[float] = field(default_factory=list)

    start_ts: str = ""
    end_ts: str = ""

    _wall_start: float = 0.0
    _wall_end: float = 0.0

    def __post_init__(self):
        self.start_ts = now_ts()
        self._wall_start = time.perf_counter()
        self.extra.update(_local_time_metadata())
        reset_vram_peak()

    @contextmanager
    def section(self, name: str):
        """上下文管理：累计 sections[name] 的耗时（秒，wall-clock）。"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.sections[name] = self.sections.get(name, 0.0) + dt

    def add_epoch_time(self, train: float, val: float = 0.0, data_load: float = 0.0) -> None:
        self.epoch_train_s.append(float(train))
        self.epoch_val_s.append(float(val))
        self.epoch_data_load_s.append(float(data_load))

    def add_forward_ms(self, ms: float) -> None:
        self.forward_ms_samples.append(float(ms))

    def add_backward_ms(self, ms: float) -> None:
        self.backward_ms_samples.append(float(ms))

    def add(self, key: str, value: Any) -> None:
        self.extra[key] = value

    def update(self, **kwargs) -> None:
        self.extra.update(kwargs)

    def finalize(self) -> None:
        self._wall_end = time.perf_counter()
        self.end_ts = now_ts()

    def _mean(self, xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "wall_clock_total_s": self._wall_end - self._wall_start if self._wall_end else 0.0,
        }
        # Module/section times
        for k, v in self.sections.items():
            out[f"t_{k}_total_s"] = round(float(v), 4)
        # Per-epoch arrays + means
        out["t_train_per_epoch_s"] = [round(x, 4) for x in self.epoch_train_s]
        out["t_val_per_epoch_s"] = [round(x, 4) for x in self.epoch_val_s]
        out["t_data_load_per_epoch_s"] = [round(x, 4) for x in self.epoch_data_load_s]
        out["t_train_per_epoch_mean_s"] = round(self._mean(self.epoch_train_s), 4)
        out["t_val_per_epoch_mean_s"] = round(self._mean(self.epoch_val_s), 4)
        out["t_train_total_s"] = round(sum(self.epoch_train_s), 4)
        out["t_val_total_s"] = round(sum(self.epoch_val_s), 4)
        # Forward/backward sample means
        out["t_forward_ms_mean"] = round(self._mean(self.forward_ms_samples), 4)
        out["t_backward_ms_mean"] = round(self._mean(self.backward_ms_samples), 4)
        out["t_forward_ms_samples"] = [round(x, 4) for x in self.forward_ms_samples[:200]]
        out["t_backward_ms_samples"] = [round(x, 4) for x in self.backward_ms_samples[:200]]
        # Resource peaks
        out["peak_vram_mb"] = round(peak_vram_mb(), 2)
        out["peak_ram_mb"] = round(peak_ram_mb(), 2)
        # Extras (best_epoch, inference_per_sample_ms, params, flops, etc.)
        out.update(self.extra)
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, ensure_ascii=False, indent=2)


class CudaTimer:
    """CUDA 事件级精确计时；CPU 上回退到 perf_counter。"""

    def __init__(self, device: Optional[Any] = None):
        self.use_cuda = _HAS_TORCH and torch.cuda.is_available()
        self.device = device
        self._start_evt = None
        self._end_evt = None
        self._cpu_start = 0.0
        self._cpu_end = 0.0

    def start(self) -> None:
        if self.use_cuda:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
        else:
            self._cpu_start = time.perf_counter()

    def stop(self) -> float:
        if self.use_cuda and self._end_evt is not None:
            self._end_evt.record()
            torch.cuda.synchronize(self.device)
            return float(self._start_evt.elapsed_time(self._end_evt))  # ms
        else:
            self._cpu_end = time.perf_counter()
            return (self._cpu_end - self._cpu_start) * 1000.0
