import os, time, torch
from model import Attribution
from model_utils import log_data, args_parse, EarlyStopping, score, evaluate, write_log
from data_loader import load_cti_kg

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def model_run(args):

    start = time.time()

    labels,num_classes,train_mask,val_mask,test_mask,attribute_type_feat,nlt_feat,topo_relation_feat,node_type_vec,heterG_adj,report_node,homoG_adj_MPs = load_cti_kg(seed=args['seed'])
    print('dataset loaded.')

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    heterG_adj = heterG_adj.to(args['device'])
    report_node = report_node.to(args['device'])
    attribute_type_feat = attribute_type_feat.to(args['device'])
    nlt_feat = nlt_feat.to(args['device'])
    topo_relation_feat = topo_relation_feat.to(args['device'])
    node_type_vec = node_type_vec.to(args['device'])

    homoG_adj_MPs = [graph.to(args['device']) for graph in homoG_adj_MPs]
    inputs = heterG_adj, report_node, attribute_type_feat, nlt_feat, topo_relation_feat, node_type_vec

    model = Attribution(
                        nlt_in_size=nlt_feat.shape[1],
                        ft_out_dim=args['ft_out_dim'],
                        emb_dim=args['emb_dim'],
                        type_dim=node_type_vec.shape[1],
                        dropout_ioc=args['dropout_ioc'],
                        num_heads=args['num_heads'],
                        num_meta_paths=len(homoG_adj_MPs),
                        hidden_size=args['hidden_units'],
                        out_size=num_classes,
                        dropout_mpneigh=args['dropout_mpneigh'],
                        cuda=args['cuda']).to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    train_labels = labels[train_mask]
    class_count = torch.bincount(train_labels, minlength=num_classes).float().to(args['device'])
    class_weight = torch.pow(torch.clamp(class_count.sum() / torch.clamp(class_count, min=1.0), max=20.0), 0.5)
    class_weight = class_weight / class_weight.mean()
    loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args['lr_factor'],
        patience=args['lr_patience'],
        min_lr=args['min_lr']
    )

    log_strs = [str(args)]

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(homoG_adj_MPs, inputs)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += p.grad.detach().pow(2).sum().item()
        grad_norm = grad_norm_sq ** 0.5
        if args['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, homoG_adj_MPs, inputs, val_mask, labels, loss_fcn)
        early_stop = stopper.step(val_macro_f1, val_acc, model)
        scheduler.step(val_macro_f1)
        current_lr = optimizer.param_groups[0]['lr']

        log_str = 'Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | ' \
                'Val Loss {:.4f} | Val Acc {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | GradNorm {:.4f} | LR {:.6f}' \
                .format(epoch+1, loss.item(), train_acc, train_micro_f1, train_macro_f1, val_loss.item(), val_acc, val_micro_f1, val_macro_f1, grad_norm, current_lr)
        log_strs.append(log_str)
        print(log_str)

        if early_stop:
            break

    stopper.load_checkpoint(model)
    if args.get('run_test_eval', True):
        test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, homoG_adj_MPs, inputs, test_mask, labels, loss_fcn)
        log_str = 'Test loss {:.4f} | Test Acc {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_loss.item(), test_acc, test_micro_f1, test_macro_f1)
        log_strs.append(log_str)
        print(log_str)
    else:
        log_str = 'Test evaluation skipped (--no_test_eval); tuning should rely on validation metrics only.'
        log_strs.append(log_str)
        print(log_str)

    end = time.time()
    log_str = 'Total seconds for train and test: {}s'.format(int(end-start))
    log_strs.append(log_str)
    print(log_str)

    write_log(log_strs, os.path.join(log_data, stopper.time+'_log.txt'))
    return None


if __name__ == '__main__':

    args = args_parse()
    model_run(args)
