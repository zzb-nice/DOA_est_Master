import numpy as np
import argparse
import os
import json

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import copy

from data_creater.signal_datasets import ULA_dataset, array_Dataloader
from data_creater.Create_k_source_dataset import Create_random_k_input_theta, \
    Create_datasets

from models.dl_model.vision_transformer.vit_model import VisionTransformer
from models.dl_model.vision_transformer.embeding_layer import scm_embeding
from vit_tranfer_learning.theta_creater.theta_creater import same_data_Creater

from utils.early_stop import EarlyStopping


def transfer_learning(transfer_model, base_model, data_loader, data_creater, optimizer, loss_f, device, epoch, snap,
                      snr, cal_Gram_matrix=False):
    transfer_model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data
        # pred: batch,n_dims
        pred = transfer_model(input.to(device), logits=True)

        # data to fit
        source_domain_data = data_creater(labels, snap, snr).to(device)
        fit_logits = base_model(source_domain_data, logits=True)

        loss = loss_f(pred, fit_logits)
        if cal_Gram_matrix:
            align_value = 5
            # p * p
            Gram_target = pred.transpose(-1, -2) @ pred
            gram_inv = torch.linalg.pinv(1/2*(Gram_target+Gram_target.transpose(-1, -2)))
            gram_inv = 1/2*(gram_inv+gram_inv.transpose(-1, -2))

            Gram_source = fit_logits.transpose(-1, -2) @ fit_logits
            s_gram_inv = torch.linalg.pinv(1/2*(Gram_source+Gram_source.transpose(-1, -2)))
            s_gram_inv = 1/2*(s_gram_inv+s_gram_inv.transpose(-1, -2))
            loss_inv = torch.mean((gram_inv-s_gram_inv)**2)

            loss = loss + 1 * loss_inv
            # loss_scale = torch.mean((S1[:align_value]-S2[:align_value])**2)
            # loss = loss+1*loss_cos+0.1*loss_scale

        # loss = loss_f(score, labels)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] RMSE_loss: {:.3f}".format(epoch,
                                                                       np.sqrt(accu_loss.item() / (step + 1))
                                                                       )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device, epoch, grid_to_theta=True, k=3):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        pred = model(input.to(device))
        if grid_to_theta:
            _, pred = model.sp_to_doa(pred, k)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] RMSE_loss: {:.3f}".format(epoch,
                                                                       np.sqrt(accu_loss.item() / (step + 1))
                                                                       )

    return accu_loss.item() / (step + 1)


def main(args):
    train_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                  args.signal_range[1], 500, min_delta_theta=3)
    val_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                args.signal_range[1], 20000, min_delta_theta=3)
    # theta_set = Create_determined_sep_doas(args.k, args.signal_range[0], args.signal_range[1], None, 10, True, 0.1)

    save_path = args.root + f'_transfer_learning_rho_{args.rho}_plus'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the parameter setting
    save_args(args, os.path.join(save_path, 'laboratory_set.json'))
    # save the run_file
    tb_writer = SummaryWriter(logdir=os.path.join(save_path, 'run'))
    tags = ["train_loss", "val_loss", "learning_rate"]

    # loss function
    # loss_function_1 = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()

    id1 = f'_snr_{args.snr}_snap_{args.snap}'
    embeding_dim = 768
    base_model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                                   out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
    model_name = 'DOA-ViT'

    load_name = f'weight_snr_{args.snr}.pth'
    state_dict = torch.load(os.path.join(args.root, load_name), map_location=args.device)
    base_model.load_state_dict(state_dict, strict=False)
    # base_model.load_state_dict(state_dict, strict=True)
    base_model.to(args.device)

    transfer_model = copy.deepcopy(base_model)
    # 冻结 base model 权重
    for param in base_model.parameters():
        param.requires_grad = False

    # dataset 有阵列误差, contrast 无阵列误差
    dataset, val_dataset = ULA_dataset(args.M, -60, 60, 1, args.rho), ULA_dataset(args.M, -60, 60, 1, args.rho)
    base_dataset = ULA_dataset(args.M, -60, 60, 1, 0)
    Create_datasets(dataset, args.k, train_theta_set, batch_size=50, snap=args.snap, snr=args.snr)
    Create_datasets(val_dataset, args.k, val_theta_set, batch_size=512, snap=args.snap, snr=args.snr)
    data_creater = same_data_Creater(base_dataset, 'scm')

    train_dataloader = array_Dataloader(dataset, 32, load_style='torch', input_type='scm', output_type='doa')
    val_dataloader = array_Dataloader(val_dataset, 32, shuffle=False, load_style='torch',
                                      input_type='scm', output_type='doa')

    # 优化器初始化
    parm = [p for p in transfer_model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(parm,lr=0.0001,momentum=0.9,weight_decay=0)
    optimizer = optim.Adam(parm, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
    # optimizer = optim.Adam(parm, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, 0.0001, 'rel', 3, 0, 1e-8,
                                                       False)
    # lr_schedule = optim.lr_scheduler.StepLR(optimizer, 10, 0.5, -1)

    # stop training earlier if validation loss doesn't decrease
    early_stopping = EarlyStopping(100, 0)

    # 对所有的信噪比情况进行训练
    min_val_loss = 100
    for epoch in range(args.epochs):
        val_loss = evaluate(transfer_model, val_dataloader, loss_function, args.device, epoch + 1, False, args.k)
        train_loss = transfer_learning(transfer_model, base_model, train_dataloader, data_creater, optimizer,
                                       loss_function, args.device, epoch + 1, args.snap, args.snr, cal_Gram_matrix=False)

        val_loss = np.sqrt(val_loss)

        # 根据val_loss,学习率调度器
        lr_schedule.step(val_loss)

        # 保存数据
        tb_writer.add_scalar(tags[0] + id1, train_loss, epoch)
        tb_writer.add_scalar(tags[1] + id1, val_loss, epoch)
        tb_writer.add_scalar(tags[2] + id1, optimizer.param_groups[0]["lr"], epoch)

        # 提早停止训练
        early_stopping(val_loss)
        # 若满足 early stopping 要求,结束模型训练
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(transfer_model.state_dict(), os.path.join(save_path, f'weight' + id1 + '.pth'))
            print(f'model saved, minimun val_loss:{min_val_loss}')

        # 保存最后一个epoch的模型参数
        if epoch == args.epochs - 1:
            torch.save(transfer_model.state_dict(), os.path.join(save_path, f'weight_end' + id1 + '.pth'))

    vloss_total = min_val_loss
    print(f'min validation loss is {vloss_total}')


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # scenario parameters
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--snr', type=list, default=-10)
    parser.add_argument('--snap', type=int, default=10)
    parser.add_argument('--signal_range', type=tuple, default=(-60, 60))
    parser.add_argument('--rho', type=float, default=2)

    root_path = os.path.abspath('../')
    root = os.path.join(root_path, 'results', 'vit', f'vit_M_8_k_3')
    parser.add_argument('--root', type=str, default=root)

    # training parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=int, default=0.0001)

    # model parameters
    # ...
    parser.add_argument('--grid_to_theta', type=bool, default=False)

    # transfer_learning parameter
    parser.add_argument('--same_num', type=int, default=4)

    args = parser.parse_args()

    main(args)
