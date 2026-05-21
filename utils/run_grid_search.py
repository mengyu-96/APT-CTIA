import argparse
import itertools
import subprocess
import sys
import pandas as pd
from pathlib import Path
import json
import re

def run_experiment(
    data_dir, 
    output_dir, 
    model_type, 
    hidden_dim, 
    oversample, 
    focal, 
    class_weights, 
    dropout,
    lr,
    epochs=200
):
    cmd = [
        sys.executable, "train_gnn.py",
        "--graphs-path", f"{data_dir}/graphs.pt",
        "--label-mapping-path", f"{data_dir}/label_mapping.json",
        "--output-dir", str(output_dir),
        "--model-type", model_type,
        "--hidden-dim", str(hidden_dim),
        "--epochs", str(epochs),
        "--cv-folds", "5",
        "--dropout", str(dropout),
        "--lr", str(lr),
        "--patience", "30"
    ]
    
    if oversample:
        cmd.append("--oversample")
    if focal:
        cmd.append("--focal")
        cmd.append("--focal-gamma")
        cmd.append("2.0")
    if class_weights:
        cmd.append("--class-weights")
        
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8' if sys.platform == 'win32' else None
        )
        
        # Parse output for metrics
        output = result.stderr + result.stdout
        
        # Look for: 交叉验证完成: Acc=0.5724±0.0679, ...
        match = re.search(r"交叉验证完成: Acc=([0-9.]+)±", output)
        acc = float(match.group(1)) if match else 0.0
        
        match = re.search(r"BalAcc=([0-9.]+)±", output)
        bal_acc = float(match.group(1)) if match else 0.0
        
        match = re.search(r"F1_w=([0-9.]+)±", output)
        f1_w = float(match.group(1)) if match else 0.0
        
        match = re.search(r"Top-3=([0-9.]+)±", output)
        top3 = float(match.group(1)) if match else 0.0
        
        return {
            "model": model_type,
            "hidden": hidden_dim,
            "oversample": oversample,
            "focal": focal,
            "class_weights": class_weights,
            "dropout": dropout,
            "lr": lr,
            "acc": acc,
            "bal_acc": bal_acc,
            "f1_w": f1_w,
            "top3": top3
        }
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="grid_search_results.csv")
    args = parser.parse_args()
    
    # Grid Search Space
    # Reduce space to run fast for user
    models = ["GraphSAGE", "GAT"]
    hidden_dims = [128, 256]
    oversamples = [True, False]
    focals = [True, False]
    # Fixed for now
    class_weights = [True] 
    dropouts = [0.3, 0.5]
    lrs = [0.001]
    
    results = []
    
    combinations = list(itertools.product(models, hidden_dims, oversamples, focals, class_weights, dropouts, lrs))
    print(f"Total combinations: {len(combinations)}")
    
    for i, (model, hidden, over, foc, cw, drop, lr) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: {model}, Hidden={hidden}, Oversample={over}, Focal={foc}, CW={cw}, Drop={drop}")
        
        out_dir = Path(f"grid_results/exp_{i}")
        
        res = run_experiment(
            args.data_dir, 
            out_dir, 
            model, 
            hidden, 
            over, 
            foc, 
            cw, 
            drop,
            lr
        )
        
        if res:
            results.append(res)
            print(f"Result: Acc={res['acc']:.4f}, BalAcc={res['bal_acc']:.4f}")
            
            # Save intermediate
            df = pd.DataFrame(results)
            df.to_csv(args.output_csv, index=False)
            
    print("\nTop 5 Configs:")
    df = pd.DataFrame(results)
    print(df.sort_values(by="acc", ascending=False).head(5))

if __name__ == "__main__":
    main()
