import numpy as np
import argparse
import os
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import inspect

from data_creater.signal_datasets import ULA_dataset, array_Dataloader
from data_creater.Create_k_source_dataset import Create_determined_sep_doas, Create_random_k_input_theta, \
    Create_datasets
from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot
from data_save.plot.plot_pre_result import plot_predict_result, plot_error

from literature_CNN import std_CNN,modified_CNN
from utils.loss_function import multi_classification_loss

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
            pred = F.sigmoid(pred)
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
            pred = F.sigmoid(pred)
            _, pred = model.sp_to_doa(pred, k)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] RMSE_loss: {:.3f}".format(epoch,
                                                                       np.sqrt(accu_loss.item() / (step + 1))
                                                                       )

    return accu_loss.item() / (step + 1)


def main(args):
    # save_path
    save_path = args.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                  args.signal_range[1], 50000, min_delta_theta=2)
    val_theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                                args.signal_range[1], 20000, min_delta_theta=2)
    # theta_set = Create_determined_sep_doas(args.k, args.signal_range[0], args.signal_range[1], None, 10, True, 0.1)

    # save the parameter setting
    save_args(args, os.path.join(save_path, 'laboratory_set.json'))
    # save the run_file
    tb_writer = SummaryWriter(logdir=os.path.join(save_path, 'run'))
    tags = ["train_loss", "val_loss", "learning_rate"]

    # loss function
    # loss_function = torch.nn.MSELoss()
    loss_function = multi_classification_loss()

    vloss_total = np.zeros((len(args.snaps)))
    for step_1, snap in enumerate(args.snaps):
        id1 = f'_snap_{snap}'
        model = std_CNN(3, args.M, 121, sp_mode=True)
        # model = modified_CNN(2, args.M, 121, sp_mode=True)
        model_name = 'std_CNN'
        model.to(args.device)

        dataset, val_dataset = ULA_dataset(args.M, -60, 60, 1, args.rho), ULA_dataset(args.M, -60, 60, 1, args.rho)
        Create_datasets(dataset, args.k, train_theta_set, batch_size=512, snap=snap, snr=args.snr)
        Create_datasets(val_dataset, args.k, val_theta_set, batch_size=512, snap=snap, snr=args.snr)

        train_dataloader = array_Dataloader(dataset, 256, load_style='torch', input_type='enhance_scm', output_type='spatial_sp')
        val_dataloader = array_Dataloader(val_dataset, 256, shuffle=False, load_style='torch',
                                          input_type='enhance_scm', output_type='spatial_sp')

        # save initial model weight,
        if not os.path.exists(os.path.join(save_path, '_init_weight')):
            os.makedirs(os.path.join(save_path, '_init_weight'))
        torch.save(model.state_dict(), os.path.join(save_path, '_init_weight', f'_init_weight' + id1 + '.pth'))

        # save the model structure，知道自己跑了啥代码
        model_class_file = os.path.join(save_path, 'model.py')
        with open(model_class_file, 'w') as f:
            f.write(inspect.getsource(model.__class__))

        # 优化器初始化
        parm = [p for p in model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(parm,lr=0.0001,momentum=0.9,weight_decay=0)
        optimizer = optim.Adam(parm, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        # optimizer = optim.Adam(parm, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5, 0.0001, 'rel', 3, 0, 1e-8,
                                                           False)
        # lr_schedule = optim.lr_scheduler.StepLR(optimizer, 10, 0.5, -1)

        # stop training earlier if validation loss doesn't decrease
        early_stopping = EarlyStopping(30, 0)

        # 对所有的信噪比情况进行训练
        min_val_loss = 100
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_dataloader, loss_function, optimizer, args.device, epoch + 1,
                                         False, args.k)
            val_loss = evaluate(model, val_dataloader, loss_function, args.device, epoch + 1, False, args.k)
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

    # save loss
    random_name = str(np.random.rand(1))
    save_array(vloss_total, os.path.join(save_path, 'validation_loss_' + random_name + '.csv'),
               index=['snr_' + str(args.snr)],
               header=['snap_' + str(i) for i in args.snaps])
    # plot loss
    loss_1d_plot(vloss_total, model_name, args.snaps, 'snaps', False,
                 os.path.join(save_path, f'validation_loss_{random_name}.png'))


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # scenario parameters
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--snr', type=int, default=-10)
    parser.add_argument('--snaps', type=list, default=[1, 5, 10, 30, 50, 100])
    parser.add_argument('--signal_range', type=tuple, default=(-60, 60))
    parser.add_argument('--rho', type=float, default=0)

    root_path = os.path.abspath('../../')
    save_root = os.path.join(root_path, 'results', 'CNN_load_path', f'std_CNN_snr_-10_v_snap(2)')
    parser.add_argument('--save_root', type=str, default=save_root)

    # training parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=500)

    # model parameters
    # ...
    parser.add_argument('--grid_to_theta', type=bool, default=False)

    args = parser.parse_args()

    main(args)
