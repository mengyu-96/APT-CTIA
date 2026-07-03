from argparse import ArgumentParser
from copy import deepcopy
import math
import os
import pickle
from types import SimpleNamespace

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch
from torch.optim import Adam
from tqdm import tqdm

from models.gnn import SageClassifier


torch.set_num_threads(16)
HYPERPARAMS = SimpleNamespace(
    icefog_fixed=True,
    ioc_enc=64, hidden=512, layers=3, aggr='max', autoencoder=True, variational=False,
    lr=1e-4, wd=1e-5, epochs=100, bs=16, patience=10, sample_size=512
)
BEST_HP = SimpleNamespace(
    ioc_enc=64, hidden=512, layers=2, aggr='max', autoencoder=True,
    variational=False, lr=0.0001, wd=1e-05, epochs=100, bs=16, patience=10,
    sample_size=32, heads=0
)

def train(hp, model, g, batch_tr, y_tr, batch_va, y_va, batch_te, y_te):
    opt = Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)
    n_batches = math.ceil(batch_tr.size(0)/hp.bs)

    best = {'e': -1, 'va': 0, 'te-acc': 0, 'te-bacc':0, 'sd': None}
    logs = []

    original_y = deepcopy((g.event_ids, g.y))

    # Make sure model can only see classes of nodes it's not doing inference on
    tr_mask = ~((g.event_ids == batch_tr.unsqueeze(-1)).sum(dim=0).bool())
    g_tr_y = g.y[tr_mask]
    g_tr_ids = g.event_ids[tr_mask]

    va_mask = ~((g.event_ids == batch_va.unsqueeze(-1)).sum(dim=0).bool())
    g_va_y = g.y[va_mask]
    g_va_ids = g.event_ids[va_mask]

    te_mask = ~((g.event_ids == batch_te.unsqueeze(-1)).sum(dim=0).bool())
    g_te_y = g.y[te_mask]
    g_te_ids = g.event_ids[te_mask]

    for e in range(hp.epochs):
        perm = torch.randperm(batch_tr.size(0))
        batch_tr = batch_tr[perm]
        y_tr = y_tr[perm]

        prog = tqdm(range(n_batches))

        g.y = g_tr_y
        g.event_ids = g_tr_ids
        for b in prog:
            idx = batch_tr[b*hp.bs : (b+1)*hp.bs]
            y = y_tr[b*hp.bs : (b+1)*hp.bs]

            model.train()
            opt.zero_grad()
            loss = model(g, idx, y)
            loss.backward()
            opt.step()

            prog.desc = f"[{e}] {loss.item():0.2E}"

        model.eval()
        tr_acc, tr_bacc = eval(model, g, batch_tr, y_tr)
        print(f"\tTr   acc: {tr_acc:0.3f}\tb-acc: {tr_bacc:0.3f}")

        g.y = g_va_y
        g.event_ids = g_va_ids
        va_acc, va_bacc = eval(model, g, batch_va, y_va)
        print(f"\tVal  acc: {va_acc:0.3f}\tb-acc: {va_bacc:0.3f}")

        g.y = g_te_y
        g.event_ids = g_te_ids
        te_acc, te_bacc = eval(model, g, batch_te, y_te)
        print(f"\tTest acc: {te_acc:0.3f}\tb-acc: {te_bacc:0.3f}")

        log = dict(
            epoch=e,
            tr_acc=tr_acc, tr_bacc=tr_bacc,
            va_acc=va_acc, va_bacc=va_bacc,
            te_acc=te_acc, te_bacc=te_bacc,
        )
        logs.append(log)

        if va_bacc > best['va']:
            best['e'] = e
            best['va'] = va_bacc
            best['te-acc'] = te_acc
            best['te-bacc'] = te_bacc
            best['sd'] = deepcopy(model.state_dict())

        if e-best['e'] >= hp.patience:
            print("Early stopping")
            break

    g.event_ids = original_y[0]
    g.y = original_y[1]
    return best, logs

@torch.no_grad()
def eval(model, g, idx, y, print_mat=False):
    model.eval()
    preds = model.inference(g, idx)
    y_hat = preds.argmax(dim=1)
    y_true_cpu = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    y_hat_cpu = y_hat.detach().cpu().numpy()

    b_acc = balanced_accuracy_score(y_true_cpu, y_hat_cpu)
    acc = accuracy_score(y_true_cpu, y_hat_cpu)

    if print_mat:
        print(confusion_matrix(y_true_cpu, y_hat_cpu))

    return acc, b_acc

def get_and_split(dataset, folds=5, val=True):
    try:
        g = torch.load(f'{dataset}/full_graph_csr.pt', weights_only=False)
    except:
        g = torch.load(f'{dataset}/full_graph_csr.pt')
    kf = StratifiedKFold(n_splits=folds)

    # For local dataset, use 'local' source or just all non-empty events
    if 'OTX' in g.src_map:
        valid_mask = (g.sources == g.src_map['OTX']).logical_and(g.y != -1)
    else:
        # Use all labeled events if OTX is not present (local mode)
        valid_mask = (g.y != -1)

    # Taken care of in graph construction now
    #num_neighbors = torch.tensor([len(n) for n in g.edge_csr[g.event_ids]])
    #valid_mask = num_neighbors != 0

    valid_ids = g.event_ids[valid_mask]
    valid_ys = g.y[valid_mask]

    other_ids = g.event_ids[~valid_mask]
    other_ys = g.y[~valid_mask]

    for tr_idx_all, te_idx in kf.split(valid_ids, valid_ys):
        if val:
            # Split tr_idx_all into 70% train, 10% val (from total 80% training fold)
            # Calculate 12.5% of tr_idx_all as validation (10% of total)
            val_size = int(len(tr_idx_all) * 0.125)
            
            # Ensure each class is present in both train and val
            classes = torch.unique(valid_ys[tr_idx_all])
            tr_idx = []
            va_idx = []
            
            # First, ensure at least one sample per class in both splits
            for cls in classes:
                cls_mask = (valid_ys[tr_idx_all] == cls)
                cls_indices = tr_idx_all[cls_mask.nonzero().squeeze()]
                if len(cls_indices) >= 2:
                    va_idx.append(cls_indices[0:1])
                    tr_idx.append(cls_indices[1:])
                else:
                    tr_idx.append(cls_indices)
            
            tr_idx = torch.cat(tr_idx)
            va_idx = torch.cat(va_idx) if len(va_idx) > 0 else torch.tensor([], dtype=torch.long)
            
            # Then fill remaining to reach approximate 7:1 ratio
            remaining = tr_idx_all[~torch.isin(tr_idx_all, torch.cat([tr_idx, va_idx]))]
            if len(remaining) > 0:
                perm = torch.randperm(len(remaining))
                split_point = len(remaining) - max(0, val_size - len(va_idx))
                tr_idx = torch.cat([tr_idx, remaining[perm[:split_point]]])
                va_idx = torch.cat([va_idx, remaining[perm[split_point:]]])
            
            # Final splits
            final_tr_idx = torch.cat([valid_ids[tr_idx], other_ids])
            final_tr_ys = torch.cat([valid_ys[tr_idx], other_ys])
            
            final_va_idx = valid_ids[va_idx]
            final_te_idx = valid_ids[te_idx]

            yield g, \
                (final_tr_idx, final_tr_ys), \
                (final_va_idx, valid_ys[va_idx]), \
                (final_te_idx, valid_ys[te_idx])
        else:
            tr_idx = torch.cat([valid_ids[tr_idx_all], other_ids])
            tr_ys = torch.cat([valid_ys[tr_idx_all], other_ys])

            te_idx_final = valid_ids[te_idx]
            yield g, \
                (tr_idx, tr_ys), \
                (te_idx_final, valid_ys[te_idx])

def to_onehot(y, num_classes=None):
    if num_classes is None:
        num_classes = y.max().item() + 1
    y_onehot = torch.zeros(y.size(0), num_classes)
    y_onehot[torch.arange(y.size(0)), y.long()] = 1.
    return y_onehot


@torch.no_grad()
def get_final_preds(model, g, te_idx, te_y):
    original_y = deepcopy((g.event_ids, g.y))

    # Mask out labels we're testing on
    te_mask = ~((g.event_ids == te_idx.unsqueeze(-1)).sum(dim=0).bool())
    g_te_y = g.y[te_mask]
    g_te_ids = g.event_ids[te_mask]
    g.y = g_te_y
    g.event_ids = g_te_ids

    preds = model.inference(g, te_idx)
    g.event_ids = original_y[0]
    g.y = original_y[1]

    return preds, te_y

def main(hp):
    stats = []
    generator = get_and_split(hp.dataset)
    i = 0

    has_ae = '+ae' if hp.autoencoder else ''
    has_ae += '+var' if hp.variational else ''
    has_ae += f'+attn-{hp.heads}' if hp.heads else ''
    for g, tr,va,te in generator:
        y = tr[1]
        num_classes = (g.y.max()+1).item()
        
        # Calculate class weights
        per_class = torch.bincount(y, minlength=num_classes).float()
        weight = y.size(0) / (num_classes * per_class + 1e-6)

        model = SageClassifier(
            hp.dataset, hp.ioc_enc, hp.hidden, (g.y.max()+1).item(), class_weights=weight,
            layers=hp.layers, aggr=hp.aggr, autoencoder=hp.autoencoder,
            sample_size=hp.sample_size, variational=hp.variational,
            heads=hp.heads
        )

        # Ensure directories exist
        os.makedirs(f'weights/{hp.layers}-layer', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs(f'predictions/{hp.layers}-layer', exist_ok=True)
        os.makedirs('results/lprop+feats', exist_ok=True)

        best,log = train(hp, model, g, *tr, *va, *te)
        sd = best.pop('sd')
        torch.save((sd, model.args, model.kwargs), f'weights/{hp.layers}-layer/gnn_train-{best["va"]:0.3f}_{hp.aggr}_lprop+feats{has_ae}-new-data.pt')

        with open(f'logs/lprop_feats{has_ae}.pkl', 'wb') as f:
            pickle.dump(log, f)

        stats.append(best)

        model.load_state_dict(sd)
        model.eval()
        preds, ys = get_final_preds(model, g, *te)
        torch.save((preds,ys), f'predictions/{hp.layers}-layer/lprop_feats_gnn-{has_ae}-{i}.pt')

        i+=1

    df = pd.DataFrame(stats)
    print(df)
    with open(f'results/lprop+feats/gnn_{hp.layers}-layers{has_ae}.csv', 'a') as f:
        f.write(str(hp) + '\n\n')
        df.to_csv(f)
        df.mean().to_csv(f)
        df.sem().to_csv(f)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--no-ae', action='store_false')
    ap.add_argument('--variational', action='store_true')
    ap.add_argument('--layers', type=int, default=2)
    # Paper-style setting uses larger neighborhood sampling.
    ap.add_argument('-s', '--samples', type=int, default=512)
    ap.add_argument('--aggr', default='max')
    ap.add_argument('--attention', type=int, default=0)
    ap.add_argument('--dataset', default='otx_dataset-d2e')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--ioc_enc', type=int, default=64)
    args = ap.parse_args()

    HYPERPARAMS.autoencoder = args.no_ae
    HYPERPARAMS.variational = args.variational
    HYPERPARAMS.layers = args.layers
    HYPERPARAMS.sample_size = args.samples
    HYPERPARAMS.aggr = args.aggr
    HYPERPARAMS.heads = args.attention
    HYPERPARAMS.dataset = args.dataset
    HYPERPARAMS.hidden = args.hidden
    HYPERPARAMS.ioc_enc = args.ioc_enc
    HYPERPARAMS.dataset = args.dataset
    main(HYPERPARAMS)
