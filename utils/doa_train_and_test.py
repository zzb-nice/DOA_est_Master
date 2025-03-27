import torch
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm


# TODO: failure 控制
# Use function:train_one_epoch while confirm that the model does not produce abnormal outputs (such as torch.nan)
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


#  train and test the model while grid_to_theta is set to True, there must be failure samples in training samples
def train_with_flase(model, data_loader, loss_function, optimizer, device, epoch, grid_to_theta=True, k=3):
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
def evaluate_with_flase(model, data_loader, loss_function, device, epoch, grid_to_theta=True, k=3):
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


# test process should record predict results, and record spectrum while grid_to_theta is set to False
@torch.no_grad()
def test(model, data_loader, loss_function, device, grid_to_theta=True, k=3, return_sp=False):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    pred_result = []
    sp_result = []
    succ_vec = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        pred = model(input.to(device))
        if grid_to_theta:
            pred = F.sigmoid(pred)
            sp_result.append(pred)
            succ_i, pred = model.sp_to_doa(pred, k)
            succ_vec.append(succ_i)

            # calculate success loss
            loss = loss_function(pred[succ_i], labels.to(device)[succ_i])
        else:
            loss = loss_function(pred, labels.to(device))

        pred_result.append(pred)
        accu_loss += loss

    print("test_RMSE_loss: {:.3f}".format(np.sqrt(accu_loss.item() / (step + 1))))
    pred_result = torch.concatenate(pred_result, 0).cpu().numpy()

    if grid_to_theta:
        succ_vec = torch.concatenate(succ_vec, 0).cpu().numpy()
        succ_ratio = np.sum(succ_vec)/len(succ_vec)
        print(f"success ratio = {succ_ratio}")
        if return_sp:
            sp_result = torch.concatenate(sp_result, 0).cpu().numpy()
            return succ_ratio, np.sqrt(accu_loss.item() / (step + 1)), pred_result, sp_result
    else:
        succ_ratio = 1
    return succ_ratio, np.sqrt(accu_loss.item() / (step + 1)), pred_result


# 太稀疏了,不太可能回归完整空间谱
@torch.no_grad()
def test_2d(model, data_loader, loss_function, device, grid_to_theta=True, k=3):
    model.eval()

    accu_loss_theta = torch.zeros(1).to(device)  # 累计损失
    accu_loss_phi = torch.zeros(1).to(device)
    pred_result = []
    sp_result = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, labels = data

        pred = model(input.to(device))
        if grid_to_theta:  # 稀疏性，可以利用嘛
            pred = F.sigmoid(pred)
            sp_result.append(pred)
            _, pred = model.sp_to_doa(pred, k)
        else:
            sig_num = pred.shape[-1] // 2
            pred = torch.stack([pred[..., :sig_num], pred[..., sig_num:]], dim=-2)
        pred_result.append(pred)

        loss_theta = loss_function(pred[:, 0, :], labels.to(device)[:, 0, :])
        loss_phi = loss_function(pred[:, 1, :], labels.to(device)[:, 1, :])
        accu_loss_theta += loss_theta
        accu_loss_phi += loss_phi

    print(f"test_theta_RMSE_loss: {np.sqrt(accu_loss_theta.item() / (step + 1)):.3f}")
    print(f"test_phi_RMSE_loss: {np.sqrt(accu_loss_phi.item() / (step + 1)):.3f}")
    pred_result = torch.concatenate(pred_result, 0).cpu().numpy()

    # if grid_to_theta:
    #     sp_result = torch.concatenate(sp_result, 0).cpu().numpy()
    #     return accu_loss.item() / (step + 1), pred_result, sp_result
    # else:
    return accu_loss_theta.item() / (step + 1), accu_loss_phi.item() / (step + 1), pred_result
