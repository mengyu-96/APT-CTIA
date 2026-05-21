import json
from pathlib import Path
import csv

def collect_results(base_dir: Path) -> list[dict]:
    rows = []
    for path in base_dir.rglob("test_results.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append({
                "model_dir": str(path.parent),
                "accuracy": data.get("accuracy", None),
                "f1_weighted": data.get("f1_weighted", None),
                "f1_macro": data.get("f1_macro", None),
            })
        except Exception:
            continue
    return rows

def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["model_dir", "accuracy", "f1_weighted", "f1_macro"]) 
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    base = Path("models")
    rows = collect_results(base)
    write_csv(rows, base / "cmp" / "summary.csv")

if __name__ == "__main__":
    main()
