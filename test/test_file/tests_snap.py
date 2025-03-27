import numpy as np
import argparse
import os
import json

import torch.nn
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_creater.file_dataloader import file_array_Dataloader, DictionaryToAttributes
from data_creater.signal_datasets import ULA_dataset

from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot, loss_1d_v_plot, succ_1d_v_plot
from data_save.plot.plot_pre_result import plot_predict_result, plot_error, plot_v_predict_result, plot_v_error, \
    plot_doa_error
from data_save.plot.plot_CDF import plot_v_cdf, calculate_cdf_and_quantiles

from models.subspace_model.unity_esprit import Unity_ESPRIT
from models.subspace_model.esprit import ESPRIT
from models.subspace_model.music import Root_Music, Music
from models.dl_model.CNN.literature_CNN import std_CNN
from models.dl_model.vision_transformer.vit_model import VisionTransformer
from models.dl_model.vision_transformer.embeding_layer import scm_embeding
from models.compress_sensing.invoke_matlab.l1_svd import matlab_l1_svd

from utils.doa_train_and_test import test
from utils.util import read_csv_results


def main(args):
    test_type = "random_input"  # random_input,monte_carlo[-9.9,15.3,39.5],monte_carlo[5.1,30.3,54.5]
    # filename
    dataset_type = f'file_test1_{test_type}'  # save path
    load_file = os.path.join(args.dataset_path, f'test_{test_type}_snap_10.npz')  # init the dataloader for num_samples
    dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch', input_type='scm', output_type='doa')
    num_samples = dataloader.num_data
    dataset = ULA_dataset(args.M, -60, 60, 1, args.rho)  # init the dataset for algorithm used

    save_path = os.path.join(args.save_root, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the parameter setting
    save_args(args, os.path.join(save_path, 'laboratory_set.json'))

    # initialize contrast model
    contrast_model_name = ['MUSIC', 'Root-Music', 'ESPRIT', 'Unity-ESPRIT', r'$\ell_1$-SVD', 'SPE-CNN']
    record_results = [True, False, False, True, True, True]

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

    # our model
    embeding_dim = 768
    model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
    # model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
    #                           out_dims=121, drop_ratio=0, attn_drop_ratio=0, sp_mode=args.grid_to_theta,
    #                           start_angle=-60, end_angle=60, step=1)
    model_name = 'DOA_ViT'

    # loss
    loss_function = torch.nn.MSELoss()

    RMSE = np.zeros((len(args.snaps), args.repeats))
    succ_ratio = np.zeros((len(args.snaps), args.repeats))
    predict_results = np.zeros((len(args.snaps), num_samples, args.k))

    # record loss of contrast model
    contrast_loss = {key: np.zeros((len(args.snaps), args.repeats))
                     for key, include in zip(contrast_model_name, record_results) if include}

    contrast_model_result = {key: np.zeros((len(args.snaps), num_samples, args.k))
                             for key, include in zip(contrast_model_name, record_results) if include}

    contrast_succ_ratio = {key: np.zeros((len(args.snaps), args.repeats))
                           for key, include in zip(contrast_model_name, record_results) if include}
    for repeat_i in range(args.repeats):
        for step_1, snap in enumerate(args.snaps):
            id1 = f'_snap_{snap}'
            id2 = f'_snap_{(args.snaps[0], args.snaps[-1])}_snr_{args.snr}'
            load_file = os.path.join(args.dataset_path, f'test_{test_type}_snap_{snap}.npz')
            dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch', input_type='scm',
                                               output_type='doa')

            weight_path = os.path.join(args.load_root, 'weight' + id1 + '.pth')

            state_dict = torch.load(weight_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=True)
            model.to(args.device)

            suc_ratio, test_loss, predict_result = test(model, dataloader, loss_function, args.device, args.grid_to_theta, args.k)
            RMSE[step_1, repeat_i] = test_loss

            succ_ratio[step_1, repeat_i] = suc_ratio
            if repeat_i == 0:
                predict_results[step_1] = predict_result

            # initial the dataset
            dataset = ULA_dataset(args.M, -60, 60, 1, args.rho)  # init the dataset for algorithm used
            DictionaryToAttributes(dataset, dataloader.all_data)  # load the data to dataset

            # music algorithm
            # directly iterate the data in dataset
            music_index = contrast_model_name.index('MUSIC')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            # remove the results from calculate RMSE if model return False
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, R in enumerate(dataset.ori_scm):
                succ, doa = music.estimate(R, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            music_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'music_RMSE:', music_loss)

            if record_results[music_index]:  # 保存loss和成功率
                contrast_loss['MUSIC'][step_1, repeat_i] = music_loss
                contrast_succ_ratio['MUSIC'][step_1, repeat_i] = np.sum(succ_idx) / num_samples
                if repeat_i == 0:  # 保存预测结果,用于画图
                    contrast_model_result['MUSIC'][step_1] = doa_hat

            # root_music algorithm
            root_music_index = contrast_model_name.index('Root-Music')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, R in enumerate(dataset.ori_scm):
                succ, doa = root_music.estimate(R, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            root_music_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'root_music_RMSE:', root_music_loss)

            if record_results[root_music_index]:
                contrast_loss['Root-Music'][step_1, repeat_i] = root_music_loss
                contrast_succ_ratio['Root-Music'][step_1, repeat_i] = np.sum(succ_idx) / num_samples
                if repeat_i == 0:
                    contrast_model_result['Root-Music'][step_1] = doa_hat

            # ESPRIT algorithm
            esprit_index = contrast_model_name.index('ESPRIT')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, R in enumerate(dataset.ori_scm):
                succ, doa = esprit.tls_estimate(R, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            esprit_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'esprit_RMSE:', esprit_loss)

            if record_results[esprit_index]:
                contrast_loss['ESPRIT'][step_1, repeat_i] = esprit_loss
                contrast_succ_ratio['ESPRIT'][step_1, repeat_i] = np.sum(succ_idx) / num_samples
                if repeat_i == 0:
                    contrast_model_result['ESPRIT'][step_1] = doa_hat

            # Unity-ESPRIT algorithm
            unity_esprit_index = contrast_model_name.index('Unity-ESPRIT')
            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            for i, y in enumerate(dataset.y_t):
                succ, doa = unity_esprit.tls_estimate(y, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            unity_esprit_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'unity_esprit_RMSE:', unity_esprit_loss)

            if record_results[unity_esprit_index]:
                contrast_loss['Unity-ESPRIT'][step_1, repeat_i] = unity_esprit_loss
                contrast_succ_ratio['Unity-ESPRIT'][step_1, repeat_i] = np.sum(succ_idx) / num_samples
                if repeat_i == 0:
                    contrast_model_result['Unity-ESPRIT'][step_1] = doa_hat

            # l1-svd
            l1_svd_index = contrast_model_name.index(r'$\ell_1$-SVD')

            save_file = os.path.join(save_path, 'l1_svd_file', f'{id1}.mat')
            l1_svd.save_used_mat(dataset, save_file)
            succ_idx, doa_hat = l1_svd.predict(save_file,
                                               os.path.join(save_path, 'l1_svd_file', f'predict_snap_{id1}.mat'),
                                               args.k,
                                               args.snr, args.M, snap)

            l1_svd_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'l1_svd_RMSE:', l1_svd_loss)

            if record_results[l1_svd_index]:
                contrast_loss[r'$\ell_1$-SVD'][step_1, repeat_i] = l1_svd_loss
                contrast_succ_ratio[r'$\ell_1$-SVD'][step_1, repeat_i] = np.sum(succ_idx) / num_samples
                if repeat_i == 0:
                    contrast_model_result[r'$\ell_1$-SVD'][step_1] = doa_hat

            # SPE_CNN for M=8,k=3
            cnn_index = contrast_model_name.index(r'SPE-CNN')

            cnn_dataloader = file_array_Dataloader(load_file, 256, False, 'torch', 'enhance_scm', 'doa')
            weight_path = os.path.join(CNN_load_path, 'weight' + id1 + '.pth')

            state_dict = torch.load(weight_path, map_location=args.device)
            CNN.load_state_dict(state_dict, strict=True)
            CNN.to(args.device)

            suc_ratio, cnn_test_loss, cnn_predict_result = test(CNN, cnn_dataloader, loss_function, args.device, True, args.k)
            print(f'CNN test loss: {cnn_test_loss}')

            if record_results[cnn_index]:
                contrast_loss['SPE-CNN'][step_1, repeat_i] = cnn_test_loss
                contrast_succ_ratio[r'SPE-CNN'][step_1, repeat_i] = suc_ratio
                if repeat_i == 0:
                    contrast_model_result['SPE-CNN'][step_1] = cnn_predict_result

        RMSE_mean = np.mean(RMSE, axis=-1)
        RMSE_std = np.std(RMSE, axis=-1)
        succ_ratio_mean = np.mean(succ_ratio, axis=-1)

        print(RMSE_mean)
        print(f'success ratio of algorithm is {succ_ratio_mean}')
        print(f'MUSIC success ratio: {np.mean(contrast_succ_ratio["MUSIC"],axis=-1)}')

        # save loss
        random_name = str(np.random.rand(1))
        save_array(RMSE_mean, os.path.join(save_path, 'RMSE_mean_' + random_name + '.csv'),
                   index=['snr_' + str(args.snr)],
                   header=['snap_' + str(i) for i in args.snaps])
        save_array(RMSE_std, os.path.join(save_path, 'RMSE_std_' + random_name + '.csv'),
                   index=['snr_' + str(args.snr)],
                   header=['snap_' + str(i) for i in args.snaps])
        save_array(succ_ratio_mean, os.path.join(save_path, 'succ_ratio_mean_' + random_name + '.csv'),
                   index=['snr_' + str(args.snr)],
                   header=['snap_' + str(i) for i in args.snaps])
        # plot loss
        loss_1d_plot(RMSE_mean, model_name, args.snaps, 'snap', False,
                     os.path.join(save_path, f'result_{args.snaps}_{random_name}.png'))
        # save model loss
        if not os.path.exists(os.path.join(save_path, 'contrast_model')):
            os.makedirs(os.path.join(save_path, 'contrast_model'))
        for name, rmse in contrast_loss.items():
            rmse_mean = np.mean(rmse, axis=-1)
            save_array(rmse_mean, os.path.join(save_path, 'contrast_model', name + '_rmse_' + random_name + '.csv'),
                       index=['snr_' + str(args.snr)],
                       header=['snap_' + str(i) for i in args.snaps])

        contrast_loss_mean = {key: np.mean(value, axis=-1) for key, value in contrast_loss.items()}
        loss_1d_v_plot(RMSE_mean, model_name, args.snaps, 'snap', contrast_loss_mean, False,
                       os.path.join(save_path, f'contrast_RMSE_{random_name}.png'))
        contrast_succ_mean = {key: np.mean(value, axis=-1) for key, value in contrast_succ_ratio.items()}
        succ_1d_v_plot(succ_ratio_mean, model_name, args.snaps, 'snap', contrast_succ_mean, False,
                       os.path.join(save_path, f'contrast_succ_{random_name}.png'))

        # plot predict
        plot_predict = False
        if plot_predict:
            for i, snap in enumerate(args.snaps):
                contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
                plot_doa_error(dataset.doa, predict_results[i], contrast_plot,
                               os.path.join(save_path, f'pre_result_snap_{snap}_{random_name}.png'))

        plot_cdf = False
        if plot_cdf:
            for i, snap in enumerate(args.snaps):
                contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
                plot_v_cdf(dataset.doa, predict_results[i], contrast_plot,
                           os.path.join(save_path, f'pre_cdf_snap_{snap}_{random_name}.png'))
                model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent = \
                    calculate_cdf_and_quantiles(dataset.doa, predict_results[i], contrast_plot, quantiles=[80],
                                                cal_percent=[10])
                print(f'condition {i}', model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent,
                      sep='\n')

        save_pred_result = True
        if save_pred_result:
            print("save model predict results:")
            np.save(os.path.join(save_path, f'model_predict_results.npy'), predict_results)
            for name,pred in contrast_model_result.items():
                np.save(os.path.join(save_path, f'{name}_predict_results.npy'), pred)


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
    # parser.add_argument('--signal_range', type=tuple, default=((-5, 0.5, 6), (-0.5, 5, 10.5)))
    parser.add_argument('--rho', type=float, default=0)
    parser.add_argument('--repeats', type=int, default=1)

    root_path = os.path.abspath('../../')
    dataset_path = os.path.join(root_path, 'data', 'ULA_data', 'test', f'M_8_k_3_v_snap_test1')
    load_root = os.path.join(root_path, 'results', 'vit', f'vit_v_snap')
    parser.add_argument('--dataset_path', type=str, default=dataset_path)
    parser.add_argument('--load_root', type=str, default=load_root)
    parser.add_argument('--save_root', type=str, default=load_root)

    # test device
    parser.add_argument('--device', type=str, default='cuda')

    # model parameters
    # ...
    parser.add_argument('--grid_to_theta', type=bool, default=False)

    args = parser.parse_args()

    main(args)

