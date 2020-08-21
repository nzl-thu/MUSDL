import os
import sys

sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader
from dataset import VideoDataset

from utils import *
from opts import *
from config import *
from models.i3d import InceptionI3d
from models.evaluator import Evaluator


def proc_label(label, args):
    if args.type == 'USDL':
        # Scores of JIGSAWS dataset ranges from 6 to 30
        soft_label = [stats.norm.pdf(np.arange(6, 31), loc=i, scale=args.std).astype(np.float32)
                      for i in label.sum(dim=1)]  # N, 25
    else:
        # Sub-scores of JIGSAWS dataset ranges from 1 to 5, which can be seen as judge scores. See the paper of
        # JIGSAWS dataset for more details
        soft_label = [np.stack([stats.norm.pdf(np.arange(1, 6), loc=j, scale=args.std).astype(np.float32)
                                for j in i]) for i in label]  # N, 6x5

    soft_label = np.stack(soft_label)
    soft_label = soft_label / soft_label.sum(axis=-1, keepdims=True)
    soft_label = torch.from_numpy(soft_label)
    return soft_label


def get_models(args):
    """
    Get the i3d i3d and the evaluator with parameters moved to GPU.
    Use ModuleList to perform 4-fold training.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3ds = nn.ModuleList([InceptionI3d() for _ in range(4)]).cuda()
    for i3d in i3ds:
        i3d.load_state_dict(torch.load(i3d_pretrained_path))

    if args.type == 'USDL':
        evaluators = nn.ModuleList([Evaluator(output_dim=output_dim['USDL'], model_type='USDL')
                                    for _ in range(4)]).cuda()
    else:
        evaluators = nn.ModuleList([Evaluator(output_dim=output_dim['MUSDL'], model_type='MUSDL', num_judges=num_judges)
                                    for _ in range(4)]).cuda()

    if len(args.gpu.split(',')) > 1:
        i3ds = nn.ModuleList([nn.DataParallel(i3d) for i3d in i3ds])
        evaluators = nn.ModuleList([nn.DataParallel(evaluator) for evaluator in evaluators])

    return i3ds, evaluators


def get_dataloaders(args):
    dataloaders = []
    for fold in range(4):
        loader = {}
        loader['train'] = torch.utils.data.DataLoader(VideoDataset(fold, 'train', args.cls),
                                                      batch_size=args.train_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
        loader['test'] = torch.utils.data.DataLoader(VideoDataset(fold, 'test', args.cls),
                                                     batch_size=args.test_batch_size,
                                                     num_workers=args.num_workers,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     worker_init_fn=worker_init_fn)
        dataloaders.append(loader)
    return dataloaders


def compute_score(model_type, probs):
    if model_type == 'USDL':
        pred = probs.argmax(dim=-1) + label_min
    else:
        judge_scores_pred = torch.stack([prob.argmax(dim=-1) + judge_min for prob in probs], dim=1)  # N, 6
        pred = torch.sum(judge_scores_pred, dim=-1)
    return pred


def compute_loss(model_type, criterion, probs, soft_label):
    if model_type == 'USDL':
        loss = criterion(torch.log(probs), soft_label.cuda())
    else:
        loss = sum([criterion(torch.log(probs[i]), soft_label[:, i].cuda())
                    for i in range(num_judges)])
    return loss


def main(dataloaders, i3ds, evaluators, avg_logger, loggers, args):
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    criterion = nn.KLDivLoss()
    optimizers = [torch.optim.Adam([*i3d.parameters()] + [*evaluator.parameters()],
                                   lr=args.lr, weight_decay=args.weight_decay)
                  for i3d, evaluator in zip(i3ds, evaluators)]

    rho_best = 0
    epoch_best = 0
    rho_test_final = 0
    for epoch in range(args.num_epochs):

        log_and_print(avg_logger, f'Epoch: {epoch}  Current Best: {rho_best} at epoch {epoch_best}')
        rho_total = {split: 0. for split in ['train', 'test']}  # 4-fold sum

        for dataloader, i3d, evaluator, optimizer, logger in \
                zip(dataloaders, i3ds, evaluators, optimizers, loggers):

            logger.info(f'Epoch: {epoch}')

            for split in ['train', 'test']:
                true_scores = []
                pred_scores = []

                if split == 'train':
                    i3d.train()
                    evaluator.train()
                    torch.set_grad_enabled(True)
                else:
                    i3d.eval()
                    evaluator.eval()
                    torch.set_grad_enabled(False)

                for videos, label in dataloader[split]:
                    videos.transpose_(1, 2)

                    # batch_size, C, frames, H, W = videos.shape
                    # clip_feats = torch.empty(batch_size, 10, feature_dim).cuda()
                    # for i in range(10):
                    #     clip_feats[:, i] = i3d(videos[:, :, 16 * i:16 * i + 16, :, :].cuda()).squeeze(2)

                    data_pack = torch.cat(
                        [videos[:, :, i:i + 16] for i in range(0, num_frames, 16)])  # 10xN, c, 16, h, w
                    outputs = i3d(data_pack).reshape(10, len(videos), feature_dim).transpose(0, 1)  # N, 10, featdim

                    probs = evaluator(outputs.mean(dim=1))
                    preds = compute_score(args.type, probs)

                    pred_scores.extend([i.item() for i in preds])
                    true_scores.extend(label.numpy().sum(axis=-1))

                    soft_label = proc_label(label, args).cuda()

                    if split == 'train':
                        loss = compute_loss(args.type, criterion, probs, soft_label)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                rho, p = stats.spearmanr(pred_scores, true_scores)
                rho_total[split] += rho

                logger.info(f'{split} correlation: {rho}')

        for split in ['train', 'test']:
            avg_rho = rho_total[split] / 4
            log_and_print(avg_logger, f'{split} correlation: {avg_rho}')

        if epoch >= args.num_epochs - 10:
            rho_test_final += avg_rho  # sum up the last 10 epochs' test rhos

        if rho_total['test'] / 4 > rho_best:
            rho_best = rho_total['test'] / 4
            epoch_best = epoch
            log_and_print(avg_logger, '-----New best found!-----')
            if args.save:
                torch.save({'epoch': epoch,
                            'i3ds': i3ds.state_dict(),
                            'evaluators': evaluators.state_dict(),
                            'optimizer': [optimizer.state_dict() for optimizer in optimizers],
                            'rho_best': rho_best}, os.path.join('ckpts', f'{args.cls}_{args.type}.pt'))

    log_and_print(avg_logger, '-' * 50)
    log_and_print(avg_logger, f'{args.type} {args.cls} final test correlation: {rho_test_final / 10}')


if __name__ == '__main__':

    args = get_parser().parse_args()

    init_seed(args)

    log_dir = f'exp/{args.type}/{args.cls}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    avg_logger = get_logger(os.path.join(log_dir, 'avg.log'), args.log_info)
    loggers = [get_logger(os.path.join(log_dir, f'fold_{i}.log'), args.log_info)
               for i in range(4)]

    i3ds, evaluators = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3ds, evaluators, avg_logger, loggers, args)
