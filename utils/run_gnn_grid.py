from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple


LOGGER = logging.getLogger("grid_runner")


def run_preprocess(
    pdf_root: Path,
    txt_root: Path,
    output_dir: Path,
    feature_dim: int,
    embed_text: bool,
    embed_context: bool,
    context_window: int,
    use_tfidf: bool,
    tfidf_dim: int,
) -> bool:
    if (output_dir / "graphs.pt").exists() and (output_dir / "label_mapping.json").exists():
        LOGGER.info("跳过预处理，已存在图数据: %s", output_dir)
        return True
    cmd: List[str] = [
        sys.executable,
        "preprocess_apt_dataset.py",
        "--pdf-root",
        str(pdf_root),
        "--txt-root",
        str(txt_root),
        "--output-dir",
        str(output_dir),
        "--only-txt",
        "--min-entities",
        "3",
        "--feature-dim",
        str(int(feature_dim)),
    ]
    if embed_text:
        cmd.append("--embed-text")
    if embed_context:
        cmd.append("--embed-context")
        cmd.extend(["--context-window", str(int(context_window))])
    if use_tfidf:
        cmd.append("--use-tfidf")
        cmd.extend(["--tfidf-dim", str(int(tfidf_dim))])
    LOGGER.info("开始预处理: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error("预处理失败: %s", e)
        return False


def run_training(
    graphs_path: Path,
    label_mapping_path: Path,
    output_dir: Path,
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    focal: bool,
    focal_gamma: float,
    label_smoothing: float,
    logit_adjustment: float,
    edge_dropout: float,
    feature_noise: float,
    oversample: bool,
    epochs: int,
    cv_folds: int,
) -> bool:
    if cv_folds and cv_folds > 1:
        if (output_dir / "cv_results.json").exists():
            LOGGER.info("跳过训练，已存在交叉验证结果: %s", output_dir)
            return True
    else:
        if (output_dir / "test_results.json").exists():
            LOGGER.info("跳过训练，已存在测试结果: %s", output_dir)
            return True
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        "train_gnn.py",
        "--graphs-path",
        str(graphs_path),
        "--label-mapping-path",
        str(label_mapping_path),
        "--output-dir",
        str(output_dir),
        "--model-type",
        model_type,
        "--hidden-dim",
        str(int(hidden_dim)),
        "--num-layers",
        str(int(num_layers)),
        "--dropout",
        str(float(dropout)),
        "--batch-size",
        "32",
        "--epochs",
        str(int(epochs)),
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0005",
        "--patience",
        "20",
        "--optimizer",
        "adamw",
        "--scheduler",
        "cosine",
        "--train-ratio",
        "0.7",
        "--val-ratio",
        "0.15",
        "--test-ratio",
        "0.15",
        "--device",
        "auto",
        "--seed",
        "42",
        "--stratified",
        "--class-weights",
        "--edge-dropout",
        str(float(edge_dropout)),
        "--feature-noise",
        str(float(feature_noise)),
        "--label-smoothing",
        str(float(label_smoothing)),
        "--topk",
        "3",
        "--log-level",
        "INFO",
        "--cv-folds",
        str(int(cv_folds)),
        "--logit-adjustment",
        str(float(logit_adjustment)),
    ]
    if focal:
        cmd.append("--focal")
        cmd.extend(["--focal-gamma", str(float(focal_gamma))])
    if oversample:
        cmd.append("--oversample")
    LOGGER.info("开始训练: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error("训练失败: %s", e)
        return False


def load_metrics(output_dir: Path, cv_folds: int) -> Dict[str, Any]:
    if cv_folds and cv_folds > 1:
        path = output_dir / "cv_results.json"
    else:
        path = output_dir / "test_results.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        LOGGER.error("读取结果失败: %s", e)
        return {}


def build_grid() -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    feature_space: List[Dict[str, Any]] = [
        {
            "name": "hash64",
            "feature_dim": 64,
            "embed_text": False,
            "embed_context": False,
            "context_window": 0,
            "use_tfidf": False,
            "tfidf_dim": 0,
        },
        {
            "name": "hash64_tfidf128",
            "feature_dim": 64,
            "embed_text": False,
            "embed_context": False,
            "context_window": 0,
            "use_tfidf": True,
            "tfidf_dim": 128,
        },
        {
            "name": "hash64_embed",
            "feature_dim": 64,
            "embed_text": True,
            "embed_context": False,
            "context_window": 0,
            "use_tfidf": False,
            "tfidf_dim": 0,
        },
        {
            "name": "hash64_embed_ctx",
            "feature_dim": 64,
            "embed_text": True,
            "embed_context": True,
            "context_window": 64,
            "use_tfidf": False,
            "tfidf_dim": 0,
        },
    ]
    model_space: List[Dict[str, Any]] = [
        {"name": "gat", "model_type": "GAT", "hidden_dim": 128, "num_layers": 2, "dropout": 0.6},
        {"name": "sage", "model_type": "GraphSAGE", "hidden_dim": 128, "num_layers": 3, "dropout": 0.5},
        {"name": "hybrid", "model_type": "Hybrid", "hidden_dim": 128, "num_layers": 2, "dropout": 0.6},
    ]
    loss_space: List[Dict[str, Any]] = [
        {
            "name": "ce_ls01",
            "focal": False,
            "focal_gamma": 2.0,
            "label_smoothing": 0.1,
            "logit_adjustment": 0.0,
        },
        {
            "name": "focal_g2",
            "focal": True,
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "logit_adjustment": 0.0,
        },
        {
            "name": "focal_g2_logit1",
            "focal": True,
            "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "logit_adjustment": 1.0,
        },
    ]
    reg_space: List[Dict[str, Any]] = [
        {
            "name": "reg0",
            "edge_dropout": 0.0,
            "feature_noise": 0.0,
            "oversample": False,
        },
        {
            "name": "reg1",
            "edge_dropout": 0.2,
            "feature_noise": 0.01,
            "oversample": True,
        },
    ]
    grid: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for f_cfg, m_cfg, l_cfg, r_cfg in product(feature_space, model_space, loss_space, reg_space):
        grid.append((f_cfg, m_cfg, l_cfg, r_cfg))
    return grid


def write_summary(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "feature_name",
        "model_name",
        "loss_name",
        "reg_name",
        "model_type",
        "hidden_dim",
        "num_layers",
        "dropout",
        "focal",
        "label_smoothing",
        "logit_adjustment",
        "edge_dropout",
        "feature_noise",
        "oversample",
        "cv_folds",
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
        "f1_macro",
        "topk_accuracy",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行GNN网格搜索实验并汇总结果")
    parser.add_argument("--pdf-root", type=Path, default=Path("dataset_PDF"))
    parser.add_argument("--txt-root", type=Path, default=Path("dataset_TXT"))
    parser.add_argument("--preprocess-root", type=Path, default=Path("grid_processed"))
    parser.add_argument("--model-root", type=Path, default=Path("models") / "grid")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="最多运行的实验数量，0表示全部运行")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    grid = build_grid()
    rows: List[Dict[str, Any]] = []
    total = len(grid)
    LOGGER.info("实验组合总数: %d", total)
    count = 0
    for f_cfg, m_cfg, l_cfg, r_cfg in grid:
        if args.limit and count >= args.limit:
            break
        exp_id = f"{f_cfg['name']}__{m_cfg['name']}__{l_cfg['name']}__{r_cfg['name']}"
        LOGGER.info("开始实验: %s", exp_id)
        processed_dir = args.preprocess_root / f_cfg["name"]
        ok_pre = run_preprocess(
            pdf_root=args.pdf_root,
            txt_root=args.txt_root,
            output_dir=processed_dir,
            feature_dim=int(f_cfg["feature_dim"]),
            embed_text=bool(f_cfg["embed_text"]),
            embed_context=bool(f_cfg["embed_context"]),
            context_window=int(f_cfg["context_window"]),
            use_tfidf=bool(f_cfg["use_tfidf"]),
            tfidf_dim=int(f_cfg["tfidf_dim"]) if int(f_cfg["tfidf_dim"]) > 0 else 0,
        )
        if not ok_pre:
            LOGGER.error("实验预处理失败，跳过: %s", exp_id)
            count += 1
            continue
        graphs_path = processed_dir / "graphs.pt"
        label_mapping_path = processed_dir / "label_mapping.json"
        if not graphs_path.exists() or not label_mapping_path.exists():
            LOGGER.error("缺少图数据或标签映射，跳过: %s", exp_id)
            count += 1
            continue
        model_dir = args.model_root / exp_id
        ok_train = run_training(
            graphs_path=graphs_path,
            label_mapping_path=label_mapping_path,
            output_dir=model_dir,
            model_type=str(m_cfg["model_type"]),
            hidden_dim=int(m_cfg["hidden_dim"]),
            num_layers=int(m_cfg["num_layers"]),
            dropout=float(m_cfg["dropout"]),
            focal=bool(l_cfg["focal"]),
            focal_gamma=float(l_cfg["focal_gamma"]),
            label_smoothing=float(l_cfg["label_smoothing"]),
            logit_adjustment=float(l_cfg["logit_adjustment"]),
            edge_dropout=float(r_cfg["edge_dropout"]),
            feature_noise=float(r_cfg["feature_noise"]),
            oversample=bool(r_cfg["oversample"]),
            epochs=int(args.epochs),
            cv_folds=int(args.cv_folds),
        )
        if not ok_train:
            LOGGER.error("实验训练失败，跳过结果读取: %s", exp_id)
            count += 1
            continue
        metrics = load_metrics(model_dir, cv_folds=int(args.cv_folds))
        if not metrics:
            LOGGER.error("未找到结果文件: %s", exp_id)
            count += 1
            continue
        row: Dict[str, Any] = {
            "id": exp_id,
            "feature_name": f_cfg["name"],
            "model_name": m_cfg["name"],
            "loss_name": l_cfg["name"],
            "reg_name": r_cfg["name"],
            "model_type": m_cfg["model_type"],
            "hidden_dim": int(m_cfg["hidden_dim"]),
            "num_layers": int(m_cfg["num_layers"]),
            "dropout": float(m_cfg["dropout"]),
            "focal": bool(l_cfg["focal"]),
            "label_smoothing": float(l_cfg["label_smoothing"]),
            "logit_adjustment": float(l_cfg["logit_adjustment"]),
            "edge_dropout": float(r_cfg["edge_dropout"]),
            "feature_noise": float(r_cfg["feature_noise"]),
            "oversample": bool(r_cfg["oversample"]),
            "cv_folds": int(args.cv_folds),
            "accuracy": float(metrics.get("accuracy_mean", metrics.get("accuracy", 0.0))),
            "balanced_accuracy": float(metrics.get("balanced_accuracy_mean", metrics.get("balanced_accuracy", 0.0))),
            "f1_weighted": float(metrics.get("f1_weighted_mean", metrics.get("f1_weighted", 0.0))),
            "f1_macro": float(metrics.get("f1_macro_mean", metrics.get("f1_macro", 0.0))),
            "topk_accuracy": float(metrics.get("topk_accuracy_mean", metrics.get("topk_accuracy", 0.0))),
        }
        rows.append(row)
        count += 1
    summary_path = args.model_root / "summary.csv"
    write_summary(rows, summary_path)
    LOGGER.info("实验完成，结果汇总文件: %s", summary_path)


if __name__ == "__main__":
    main()

