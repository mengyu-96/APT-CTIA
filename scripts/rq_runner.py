from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rq_presets import DEFAULT_SEEDS_SPEC, get_preset
from scripts.run_experiment_suite import build_arg_parser, parse_seeds, run_suite, validate_args


DEFAULT_POST_RQ1_SEED_LIMIT = 8


def _resolve_variants(args_variant: List[str], variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_label = {str(v["label"]): v for v in variants}
    if args_variant == ["implemented"]:
        return [
            v for v in variants
            if bool(v.get("implemented", False))
            and bool(v.get("default_enabled", True))
            and str(v.get("reproduction_level", "verified")) == "verified"
        ]
    if args_variant == ["all"]:
        return list(variants)

    selected: List[Dict[str, Any]] = []
    for label in args_variant:
        if label not in by_label:
            raise ValueError(f"Unknown variant '{label}'. Available: {sorted(by_label)}")
        selected.append(by_label[label])
    return selected


def _ensure_variant_allowed(variant: Dict[str, Any], allow_approximate: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if not bool(variant.get("runnable", True)):
        reason = str(variant.get("gap_reason", "No runnable implementation is available."))
        required = variant.get("required_artifacts", [])
        required_msg = ""
        if required:
            required_msg = f" Required artifacts: {', '.join(str(item) for item in required)}."
        raise ValueError(
            f"Variant '{variant['label']}' is required by the paper but is not runnable as a strict reproduction. "
            f"{reason}{required_msg}"
        )
    level = str(variant.get("reproduction_level", "verified"))
    if allow_approximate:
        return
    if level != "verified":
        raise ValueError(
            f"Variant '{variant['label']}' is marked {level!r} and is blocked in strict reproduction mode. "
            "Re-run with --allow-approximate only if you explicitly want a non-verified or not-yet-complete experiment."
        )


def _limit_post_rq1_seeds(rq_name: str, seed_spec: str) -> str:
    if rq_name == "RQ1":
        return seed_spec

    raw_limit = os.getenv("APT_DSHG_POST_RQ1_SEED_LIMIT", str(DEFAULT_POST_RQ1_SEED_LIMIT)).strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError(f"Invalid APT_DSHG_POST_RQ1_SEED_LIMIT={raw_limit!r}; expected an integer.") from exc

    if limit <= 0:
        return seed_spec

    seeds = parse_seeds(seed_spec)
    if len(seeds) <= limit:
        return seed_spec
    limited = seeds[:limit]
    print(
        json.dumps(
            {
                "rq_name": rq_name,
                "seed_policy": "post_rq1_limit",
                "requested_seed_count": len(seeds),
                "effective_seed_count": len(limited),
                "effective_seeds": limited,
                "override_env": "APT_DSHG_POST_RQ1_SEED_LIMIT",
            },
            ensure_ascii=False,
        )
    )
    return ",".join(str(seed) for seed in limited)


def run_rq_entry(rq_name: str) -> None:
    preset = get_preset(rq_name)
    parser = build_arg_parser()
    parser.description = f"{rq_name} experiment entrypoint."
    parser.set_defaults(
        rq=preset["rq"],
        suite_name=preset["suite_name"],
        datasets=preset["datasets"],
        models=preset["default_models"],
        seeds=DEFAULT_SEEDS_SPEC,
        base_output_dir=str(ROOT / "results_archive" / "experiments" / "APT-DSHG"),
        dry_run=True,
    )
    parser.add_argument("--variant", nargs="+", default=["implemented"], help="Variant labels, or 'implemented', or 'all'.")
    parser.add_argument("--execute", action="store_true", help="Run the selected variants instead of only generating the dry-run plan.")
    args = parser.parse_args()
    if bool(args.execute):
        args.dry_run = False

    chosen_variants = _resolve_variants(args.variant, preset["variants"])
    summaries: List[Dict[str, Any]] = []

    for variant in chosen_variants:
        _ensure_variant_allowed(variant, allow_approximate=bool(args.allow_approximate), dry_run=bool(args.dry_run))
        variant_args = argparse.Namespace(**vars(deepcopy(args)))
        variant_args.variant_label = variant["label"]
        variant_args.notes = variant.get("description")
        variant_args.models = list(variant.get("models", preset["default_models"]))
        variant_args.datasets = list(variant.get("datasets", args.datasets))
        variant_args.seeds = _limit_post_rq1_seeds(rq_name, str(variant_args.seeds))
        for key, value in variant.get("overrides", {}).items():
            setattr(variant_args, key, value)
        if not bool(args.dry_run):
            validate_args(variant_args)
        summary = run_suite(variant_args)
        summaries.append(
            {
                "variant_label": variant["label"],
                "implemented": bool(variant.get("implemented", False)),
                "reproduction_level": variant.get("reproduction_level", "verified"),
                "description": variant.get("description"),
                "models": variant_args.models,
                "datasets": variant_args.datasets,
                "result": summary,
            }
        )

    out = {
        "rq_name": rq_name,
        "rq": preset["rq"],
        "dry_run": bool(args.dry_run),
        "selected_variants": [item["variant_label"] for item in summaries],
        "summaries": summaries,
    }
    out_path = Path(args.base_output_dir) / f"{preset['rq']}_selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selection_path": str(out_path), "variant_count": len(summaries), "dry_run": bool(args.dry_run)}, ensure_ascii=False, indent=2))
