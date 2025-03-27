import numpy as np
import argparse
import os
import json

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys

from data_creater.signal_datasets import ULA_dataset, array_Dataloader
from data_creater.Create_k_source_dataset import Create_random_k_input_theta, \
    Create_datasets
from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot

from models.dl_model.vision_transformer.vit_model import VisionTransformer
from models.dl_model.vision_transformer.embeding_layer import scm_embeding
from vit_tranfer_learning.theta_creater.theta_creater import same_data_Creater

from utils.early_stop import EarlyStopping


def train_one_epoch(model, data_loader, loss_function, optimizer, device, epoch, grid_to_theta=True, k=3):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        pred = model(input.to(device))
        if grid_to_theta:
            _, pred = model.sp_to_doa(pred, k)

        loss = loss_function(pred, labels.to(device))
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
    save_path = args.root + f'_fine_tune_various_num_rho_{args.rho}(2)'
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
    vloss_total = np.zeros((len(args.train_sample_nums)))
    for step_1, train_sample_num in enumerate(args.train_sample_nums):
        id1 = f'_train_sample_num_{train_sample_num}'
        train_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                      args.signal_range[1], train_sample_num, min_delta_theta=3)
        val_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                    args.signal_range[1], 20000, min_delta_theta=3)

        embeding_dim = 768
        model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                                  out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
        model_name = 'vision_transformer'

        load_name = f'weight_snr_{args.snr}.pth'
        state_dict = torch.load(os.path.join(args.root, load_name), map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        # base_model.load_state_dict(state_dict, strict=True)
        model.to(args.device)

        # # 冻结 base model 权重
        # for param in model.parameters():
        #     param.requires_grad = False

        # dataset 有阵列误差, contrast 无阵列误差
        dataset, val_dataset = ULA_dataset(args.M, -60, 60, 1, args.rho), ULA_dataset(args.M, -60, 60, 1, args.rho)
        base_dataset = ULA_dataset(args.M, -60, 60, 1, args.ori_rho)
        Create_datasets(dataset, args.k, train_theta_set, batch_size=10, snap=args.snap, snr=args.snr)
        Create_datasets(val_dataset, args.k, val_theta_set, batch_size=512, snap=args.snap, snr=args.snr)
        data_creater = same_data_Creater(base_dataset, 'scm')

        train_dataloader = array_Dataloader(dataset, 32, load_style='torch', input_type='scm', output_type='doa')
        val_dataloader = array_Dataloader(val_dataset, 32, shuffle=False, load_style='torch',
                                          input_type='scm', output_type='doa')

        # 优化器初始化
        parm = [p for p in model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(parm,lr=0.0001,momentum=0.9,weight_decay=0)
        optimizer = optim.Adam(parm, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        # optimizer = optim.Adam(parm, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, 0.0001, 'rel', 3, 0, 1e-8,
                                                           False)
        # lr_schedule = optim.lr_scheduler.StepLR(optimizer, 10, 0.5, -1)

        # stop training earlier if validation loss doesn't decrease
        early_stopping = EarlyStopping(100, 0)

        min_val_loss = 100
        for epoch in range(args.epochs):
            val_loss = evaluate(model, val_dataloader, loss_function, args.device, epoch + 1, False, args.k)
            train_loss = train_one_epoch(model, train_dataloader, loss_function, optimizer, args.device, epoch + 1,
                                         False, args.k)

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
                torch.save(model.state_dict(), os.path.join(save_path, f'weight' + id1 + '.pth'))
                print(f'model saved, minimun val_loss:{min_val_loss}')

            # 保存最后一个epoch的模型参数
            if epoch == args.epochs - 1:
                torch.save(model.state_dict(), os.path.join(save_path, f'weight_end' + id1 + '.pth'))

        vloss_total[step_1] = min_val_loss
        print(f'min validation loss is {vloss_total}')

    # save loss
    random_name = str(np.random.rand(1))
    save_array(vloss_total, os.path.join(save_path, 'validation_loss_' + random_name + '.csv'),
               index=['snap_' + str(args.snap)+'_snr_'+str(args.snr)],
               header=['train_sample_num_' + str(i) for i in args.train_sample_nums])
    # plot loss
    loss_1d_plot(vloss_total, model_name, args.train_sample_nums, 'train_sample_nums', False,
                 os.path.join(save_path, f'validation_loss_{random_name}.png'))


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # scenario parameters
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--snr', type=list, default=-5)

    parser.add_argument('--snap', type=int, default=10)
    parser.add_argument('--signal_range', type=tuple, default=(-60, 60))
    parser.add_argument('--ori_rho', type=float, default=0)
    parser.add_argument('--rho', type=float, default=1)
    # parser.add_argument('--train_sample_nums', type=list, default=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    # parser.add_argument('--train_sample_nums', type=list, default=[350, 400, 450, 500, 550, 600, 650])
    parser.add_argument('--train_sample_nums', type=list, default=[20, 50, 100, 200, 300, 500, 800, 1000])

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

    args = parser.parse_args()

    main(args)
