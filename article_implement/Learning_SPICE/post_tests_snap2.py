import numpy as np
import argparse
import os
import json

import torch.nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_creater.file_dataloader import file_array_Dataloader, DictionaryToAttributes
from data_creater.extend_R_dataset import extend_R
from data_creater.signal_datasets import ULA_dataset

from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot, loss_1d_v_plot
from data_save.plot.plot_pre_result import plot_predict_result, plot_error, plot_v_predict_result, plot_v_error, \
    plot_doa_error
from data_save.plot.plot_CDF import plot_v_cdf, calculate_cdf_and_quantiles

from models.subspace_model.unity_esprit import Unity_ESPRIT
from models.subspace_model.esprit import ESPRIT
from models.subspace_model.music import Root_Music, Music
from models.dl_model.CNN.literature_CNN import std_CNN
from models.compress_sensing.invoke_matlab.l1_svd import matlab_l1_svd
from scipy.io import savemat

from model import MLP

from utils.doa_train_and_test import test
from utils.batch_matrix_operator import vec_2_matrix
from utils.util import read_csv_results


@torch.no_grad()
def test_R_fit(model, data_loader, device):
    model.eval()
    pred_result = []

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        input, labels = data
        pred = model(input.to(device))
        pred_result.append(pred)

    pred_result = torch.concatenate(pred_result, 0).cpu().numpy()
    return pred_result


def main(args):
    test_type = "monte_carlo[5.1,30.3,54.5]"  # random_input,monte_carlo[-9.9,15.3,39.5],monte_carlo[5.1,30.3,54.5]
    dataset_type = f'file_test1_{test_type}'
    load_file = os.path.join(args.dataset_path, f'test_{test_type}_snap_1.npz')
    dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch', input_type='scm', output_type='doa')
    num_samples = dataloader.num_data
    dataset = ULA_dataset(args.M, -60, 60, 1, args.rho)
    target_dataset = extend_R(args.M, args.M_ext, args.rho)

    save_path = os.path.join(args.save_root, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_args(args, os.path.join(save_path, 'laboratory_set.json'))

    # 初始化对比模型
    contrast_model_name = ['MUSIC', 'Root-Music', 'ESPRIT', 'Unity-ESPRIT', r'$\ell_1$-SVD', 'SPE-CNN']
    record_loss = [True, False, False, True, True, True]
    record_pred = [True, False, False, True, True, True]

    post_music = Music(target_dataset.get_ext_A)
    music = Music(dataset.get_A)
    root_music = Root_Music(dataset.get_theta_fromz)
    esprit = ESPRIT(dataset.get_theta_fromz, args.M)
    unity_esprit = Unity_ESPRIT(dataset.get_theta_fromz, args.M)
    l1_svd_path = os.path.join(root_path, 'models', 'compress_sensing', 'invoke_matlab')
    l1_svd = matlab_l1_svd(l1_svd_path, -60, 60, 1)
    CNN = std_CNN(3, args.M, 121, sp_mode=True)
    cnn_path = os.path.join(root_path, 'results', f'CNN_load_path')
    cnn_file = "std_CNN_snr_-10_v_snap"
    CNN_load_path = os.path.join(cnn_path, cnn_file)

    if not os.path.exists(os.path.join(save_path, 'l1_svd_file')):
        os.makedirs(os.path.join(save_path, 'l1_svd_file'))

    # 我们的模型
    model = MLP([args.M * (args.M + 1), 128, 128, 128, args.M_ext * (args.M_ext + 1)])
    model_name = 'R_ext_MUSIC'

    # 损失函数
    loss_function = torch.nn.MSELoss()

    RMSE = np.zeros((len(args.snaps), args.repeats))
    succ_ratio = np.zeros(len(args.snaps))
    music_succ_ratio = np.zeros(len(args.snaps))
    l1_svd_succ_ratio = np.zeros(len(args.snaps))
    predict_results = np.zeros((len(args.snaps), num_samples, args.k))

    contrast_loss = {key: np.zeros((len(args.snaps), args.repeats))
                     for key, include in zip(contrast_model_name, record_loss) if include}

    contrast_model_result = {key: np.zeros((len(args.snaps), num_samples, args.k))
                             for key, include in zip(contrast_model_name, record_pred) if include}

    for repeat_i in range(args.repeats):
        for step_1, snap in enumerate(args.snaps):
            id1 = f'_snap_{snap}'
            id2 = f'_snr_{args.snr}_snap_{snap}'
            load_file = os.path.join(args.dataset_path, f'test_{test_type}_snap_{snap}.npz')
            dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch',
                                               input_type='scm_vec_include_diag', output_type='doa')

            weight_path = os.path.join(args.load_root, 'weight' + id1 + '.pth')
            state_dict = torch.load(weight_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=True)
            model.to(args.device)

            predict_result = test_R_fit(model, dataloader, args.device)

            dataset = ULA_dataset(args.M, -60, 60, 1, args.rho)
            DictionaryToAttributes(dataset, dataloader.all_data)

            # 评估我们的模型
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            R_exds = vec_2_matrix(predict_result, [predict_result.shape[0], args.M_ext, args.M_ext])
            for i, R_exd in enumerate(R_exds):
                succ, doa = post_music.estimate(R_exd, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            test_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            RMSE[step_1, repeat_i] = test_loss
            print(id1, '.', 'predict_RMSE:', test_loss)

            if repeat_i == 0:
                predict_results[step_1] = doa_hat
                succ_ratio[step_1] = np.sum(succ_idx) / num_samples

            # 保存扩展协方差矩阵
            savemat(os.path.join(save_path, f'R_ext_{snap}_snr_{args.snr}.mat'),
                    {'R_ext': R_exds, 'ext_M': args.M_ext,
                     'true_doa': np.stack(dataset.doa),
                     'grid': dataset.signal_grid, 'steer_vec': target_dataset.get_ext_A(dataset.signal_grid, None)})

            # 评估对比算法
            # MUSIC
            music_index = contrast_model_name.index('MUSIC')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, R in enumerate(dataset.ori_scm):
                succ, doa = music.estimate(R, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            music_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'music_RMSE:', music_loss)

            if record_loss[music_index]:
                contrast_loss['MUSIC'][step_1, repeat_i] = music_loss

            if repeat_i == 0 and record_pred[music_index]:
                contrast_model_result['MUSIC'][step_1] = doa_hat
                music_succ_ratio[step_1] = np.sum(succ_idx) / num_samples

            # Root-MUSIC
            root_music_index = contrast_model_name.index('Root-Music')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, R in enumerate(dataset.ori_scm):
                succ, doa = root_music.estimate(R, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            root_music_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'root_music_RMSE:', root_music_loss)

            if record_loss[root_music_index]:
                contrast_loss['Root-Music'][step_1, repeat_i] = root_music_loss

            if repeat_i == 0 and record_pred[root_music_index]:
                contrast_model_result['Root-Music'][step_1] = doa_hat

            # ESPRIT
            esprit_index = contrast_model_name.index('ESPRIT')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, R in enumerate(dataset.ori_scm):
                succ, doa = esprit.tls_estimate(R, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            esprit_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'esprit_RMSE:', esprit_loss)

            if record_loss[esprit_index]:
                contrast_loss['ESPRIT'][step_1, repeat_i] = esprit_loss

            if repeat_i == 0 and record_pred[esprit_index]:
                contrast_model_result['ESPRIT'][step_1] = doa_hat

            # Unity-ESPRIT
            unity_esprit_index = contrast_model_name.index('Unity-ESPRIT')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, y in enumerate(dataset.y_t):
                succ, doa = unity_esprit.tls_estimate(y, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            unity_esprit_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'unity_esprit_RMSE:', unity_esprit_loss)

            if record_loss[unity_esprit_index]:
                contrast_loss['Unity-ESPRIT'][step_1, repeat_i] = unity_esprit_loss

            if repeat_i == 0 and record_pred[unity_esprit_index]:
                contrast_model_result['Unity-ESPRIT'][step_1] = doa_hat

            # l1-SVD
            l1_svd_index = contrast_model_name.index(r'$\ell_1$-SVD')

            save_file = os.path.join(save_path, 'l1_svd_file', f'{id1}.mat')
            l1_svd.save_used_mat(dataset, save_file)
            succ_idx, doa_hat = l1_svd.predict(save_file,
                                               os.path.join(save_path, 'l1_svd_file', f'predict_snap_{id1}.mat'),
                                               args.k,
                                               args.snr, args.M, snap)

            l1_svd_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'l1_svd_RMSE:', l1_svd_loss)

            if record_loss[l1_svd_index]:
                contrast_loss[r'$\ell_1$-SVD'][step_1, repeat_i] = l1_svd_loss

            if repeat_i == 0 and record_pred[l1_svd_index]:
                contrast_model_result[r'$\ell_1$-SVD'][step_1] = doa_hat
                l1_svd_succ_ratio[step_1] = np.sum(succ_idx) / num_samples

            # SPE_CNN for M=8,k=3
            cnn_index = contrast_model_name.index(r'SPE-CNN')

            cnn_dataloader = file_array_Dataloader(load_file, 256, False, 'torch', 'enhance_scm', 'doa')
            weight_path = os.path.join(CNN_load_path, 'weight' + id1 + '.pth')

            state_dict = torch.load(weight_path, map_location=args.device)
            CNN.load_state_dict(state_dict, strict=True)
            CNN.to(args.device)

            suc_ratio, cnn_test_loss, cnn_predict_result = test(CNN, cnn_dataloader, loss_function, args.device, True,
                                                                args.k)
            print(f'CNN test loss: {cnn_test_loss}')

            if record_loss[cnn_index]:
                contrast_loss['SPE-CNN'][step_1, repeat_i] = cnn_test_loss

            if repeat_i == 0 and record_pred[cnn_index]:
                contrast_model_result['SPE-CNN'][step_1] = cnn_predict_result

        # 计算平均RMSE
        RMSE_mean = np.mean(RMSE, axis=-1)
        RMSE_std = np.std(RMSE, axis=-1)

        print(RMSE_mean)
        print(f'success ratio of algorithm is {succ_ratio}')
        print(f'MUSIC success ratio is {music_succ_ratio}')
        print(f'$\ell_1$-SVD success ratio is {l1_svd_succ_ratio}')

        # 保存结果
        random_name = str(np.random.rand(1))
        save_array(RMSE_mean, os.path.join(save_path, 'RMSE_mean_' + random_name + '.csv'),
                   index=['snr_' + str(args.snr)],
                   header=['snap_' + str(i) for i in args.snaps])
        save_array(RMSE_std, os.path.join(save_path, 'RMSE_std_' + random_name + '.csv'),
                   index=['snr_' + str(args.snr)],
                   header=['snap_' + str(i) for i in args.snaps])

        # 绘制结果
        loss_1d_plot(RMSE_mean, model_name, args.snaps, 'Snapshots', False,
                     os.path.join(save_path, f'RMSE_{args.snaps}_{random_name}.png'))

        if not os.path.exists(os.path.join(save_path, 'contrast_model')):
            os.makedirs(os.path.join(save_path, 'contrast_model'))
        for name, rmse in contrast_loss.items():
            rmse_mean = np.mean(rmse, axis=-1)
            save_array(rmse_mean, os.path.join(save_path, 'contrast_model', name + '_rmse_' + random_name + '.csv'),
                       index=['snr_' + str(args.snr)],
                       header=['snap_' + str(i) for i in args.snaps])

        contrast_loss_mean = {key: np.mean(value, axis=-1) for key, value in contrast_loss.items()}
        loss_1d_v_plot(RMSE_mean, model_name, args.snaps, 'Snapshots', contrast_loss_mean, False,
                       os.path.join(save_path, f'contrast_RMSE_{random_name}.png'))

        # plot predict results
        plot_predict = False
        if plot_predict:
            for i, snap in enumerate(args.snaps):
                contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
                plot_v_predict_result(dataset.doa, predict_results[i], contrast_plot,
                                      os.path.join(save_path, f'pre_result_snap_{snap}_{random_name}.png'))
                # 绘制预测误差
                plot_v_error(dataset.doa, predict_results[i], contrast_plot,
                             os.path.join(save_path, f'pre_error_snap_{snap}_{random_name}.png'))
                # 绘制DOA误差分布
                plot_doa_error(dataset.doa, predict_results[i], contrast_plot,
                               os.path.join(save_path, f'doa_error_snap_{snap}_{random_name}.png'))

        # plot CDF results
        plot_cdf = False
        if plot_cdf:
            for i, snap in enumerate(args.snaps):
                contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
                # 绘制CDF曲线
                plot_v_cdf(dataset.doa, predict_results[i], contrast_plot,
                           os.path.join(save_path, f'pre_cdf_snap_{snap}_{random_name}.png'))

                # 计算CDF的分位数和百分比
                model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent = \
                    calculate_cdf_and_quantiles(dataset.doa, predict_results[i], contrast_plot,
                                                quantiles=[80], cal_percent=[10])

                print(f'\nSnapshot {snap}:')
                print('Model quantiles:', model_quantile)
                print('Contrast model quantiles:', contrast_quantile)
                print('Model percentages:', model_cal_percent)
                print('Contrast model percentages:', contrast_cal_percent)

                # 保存CDF统计结果
                cdf_stats = {
                    'model_quantile': model_quantile,
                    'contrast_quantile': contrast_quantile,
                    'model_cal_percent': model_cal_percent,
                    'contrast_cal_percent': contrast_cal_percent
                }
                np.save(os.path.join(save_path, f'cdf_stats_snap_{snap}_{random_name}.npy'), cdf_stats)

        # 保存预测结果
        save_pred_result = True
        if save_pred_result:
            print("\nSaving model predict results:")
            np.save(os.path.join(save_path, f'model_predict_results.npy'), predict_results)
            for name, pred in contrast_model_result.items():
                np.save(os.path.join(save_path, f'{name}_predict_results.npy'), pred)
                print(f"Saved {name} results")


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 场景参数
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--M_ext', type=int, default=32)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--snr', type=int, default=-10)
    parser.add_argument('--snaps', type=list, default=[1, 5, 10, 30, 50, 100])
    parser.add_argument('--signal_range', type=tuple, default=(-60, 60))
    parser.add_argument('--rho', type=float, default=0)
    parser.add_argument('--repeats', type=int, default=1)

    root_path = os.path.abspath('../../')
    dataset_path = os.path.join(root_path, 'data', 'ULA_data', 'test', f'M_8_k_3_v_snap_test1')
    load_root = os.path.join(root_path, 'article_implement', 'Learning_SPICE', 'results', f'R_fit_snap_M_8_k_3')
    parser.add_argument('--dataset_path', type=str, default=dataset_path)
    parser.add_argument('--load_root', type=str, default=load_root)
    parser.add_argument('--save_root', type=str, default=load_root)

    # test device
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)
