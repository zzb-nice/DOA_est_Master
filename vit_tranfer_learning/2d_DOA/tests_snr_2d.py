import numpy as np
import argparse
import os
import json

import torch.nn
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_creater.UCA_datasets.UCA_datasets import UCA_dataset, array_Dataloader
from data_creater.UCA_datasets.Create_2d_k_source_dataset import Create_random_k_input_theta, \
    Create_datasets
from data_creater.UCA_datasets.Create_2d_classic_test_dataset import Create_monte_carlo_theta
from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot, loss_1d_v_plot
from data_save.plot.plot_pre_result import plot_predict_result, plot_error, plot_v_predict_result, plot_v_error, \
    plot_doa_error
from data_save.plot.plot_CDF import plot_v_cdf, calculate_cdf_and_quantiles
from data_save.plot.plot_radar import plot_radar_fig

from models.subspace_model.unity_esprit import Unity_ESPRIT
from models.subspace_model.esprit import ESPRIT
from models.subspace_model.music import Root_Music, Music, Music_2D
from models.dl_model.vision_transformer.vit_model import VisionTransformer
from models.dl_model.vision_transformer.embeding_layer import scm_embeding

from utils.doa_train_and_test import test_2d
from utils.util import read_csv_results


def main(args):
    # filename
    dataset_type = 'test_monte_carlo_[[-135, -120, -60, 0, 60], [35, 30, 10, 30, 10]]' + '_' + str(
        args.theta_range) + '_' + str(
        args.phi_range)  # random_input_sep_3,monte_carlo_[10, 13, 16],equal_sep_2,several_sep,plot
    dataset = UCA_dataset(args.M, -90, 90, 1, args.rho)
    # theta_set = Create_random_k_input_theta(args.k, args.theta_range[0], args.theta_range[1], args.phi_range[0],
    #                                         args.phi_range[1], 1000, min_delta_theta=3)
    # theta_set = Create_monte_carlo_theta(all_theta=[np.array([[-45, 0, 45], [-45, 0, 45]])], repeat_num=1000)

    theta_set = Create_monte_carlo_theta(all_theta=[np.array([[-135, -120, -60, 0, 60], [35, 30, 10, 30, 10]])],
                                         repeat_num=100)
    # [[-60, -30, 0, 30, 60], [35, 15, 35, 15, 40]]
    # [[-60, -30, 0, 30, 60], [50, 30, 50, 30, 55]]
    # [-60, -30, 0, 30, 60], [40, 25, 5, 50, 12]

    save_path = os.path.join(args.save_root, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the parameter setting
    save_args(args, os.path.join(save_path, 'laboratory_set.json'))

    # initialize contrast model
    contrast_model_name = ['MUSIC']
    record_loss = [True]
    record_pred = [True]

    music = Music_2D(dataset.get_A)

    # our model
    embeding_dim = 768
    model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=2 * args.k, drop_ratio=0, attn_drop_ratio=0)
    # model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
    #                           out_dims=121, drop_ratio=0, attn_drop_ratio=0, sp_mode=args.grid_to_theta,
    #                           start_angle=-60, end_angle=60, step=1)
    model_name = 'DOA-ViT'
    # loss
    loss_function = torch.nn.MSELoss()

    RMSE_theta = np.zeros((len(args.snrs), args.repeats))
    RMSE_phi = np.zeros((len(args.snrs), args.repeats))
    succ_ratio = np.zeros(len(args.snrs))
    music_succ_ratio = np.zeros(len(args.snrs))
    predict_results = np.zeros((len(args.snrs), theta_set.shape[0], 2, args.k))

    # record loss of contrast model
    contrast_loss_theta = {key: np.zeros((len(args.snrs), args.repeats))
                           for key, include in zip(contrast_model_name, record_loss) if include}
    contrast_loss_phi = {key: np.zeros((len(args.snrs), args.repeats))
                         for key, include in zip(contrast_model_name, record_loss) if include}

    contrast_model_result = {key: np.zeros((len(args.snrs), theta_set.shape[0], 2, args.k))
                             for key, include in zip(contrast_model_name, record_pred) if include}
    # TODO:succ ratio of the model
    for repeat_i in range(args.repeats):
        for step_1, snr in enumerate(args.snrs):
            id1 = f'_snr_{snr}'
            dataset = UCA_dataset(args.M, -90, 90, 1, args.rho)
            # bachsize should be divisible by total number, set to 1 while num of theta is not sure
            Create_datasets(dataset, args.k, theta_set, batch_size=50, snap=args.snap, snr=snr)
            dataloader = array_Dataloader(dataset, 256, False, 'torch', 'scm', 'doa')
            # dataloader = array_Dataloader(dataset, 256, False, 'torch', 'scm_vec', 'doa')

            weight_path = os.path.join(args.load_root, 'weight' + id1 + '.pth')

            state_dict = torch.load(weight_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=True)
            model.to(args.device)

            test_loss_theta, test_loss_phi, predict_result = test_2d(model, dataloader, loss_function, args.device,
                                                                     args.grid_to_theta, args.k)
            RMSE_theta[step_1, repeat_i] = np.sqrt(test_loss_theta)
            RMSE_phi[step_1, repeat_i] = np.sqrt(test_loss_phi)

            if repeat_i == 0:
                predict_results[step_1] = predict_result
                succ_ratio[step_1] = 1  # TODO

            # music algorithm
            # directly iterate the data in dataset
            music_index = contrast_model_name.index('MUSIC')
            doa_hat = np.zeros((theta_set.shape[0], 2, args.k), dtype=np.float32)
            # remove the results from calculate RMSE if model return False
            succ_idx = np.zeros(theta_set.shape[0], dtype=bool)
            for i, R in enumerate(dataset.ori_scm):
                succ, doa = music.estimate(R, args.k)
                # doa[[0, 1], :] = doa[[1, 0], :]  # 交换theta和phi,和数据保持一致
                doa_hat[i] = doa
                succ_idx[i] = succ

            music_loss_theta = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, 0, :] - doa_hat[succ_idx, 0, :]) ** 2))
            music_loss_phi = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, 1, :] - doa_hat[succ_idx, 1, :]) ** 2))
            print(id1, '.', 'music_RMSE_theta:', music_loss_theta)
            print(id1, '.', 'music_RMSE_phi:', music_loss_phi)

            if record_loss[music_index]:
                contrast_loss_theta['MUSIC'][step_1, repeat_i] = music_loss_theta
                contrast_loss_phi['MUSIC'][step_1, repeat_i] = music_loss_phi

            if repeat_i == 0 and record_pred[music_index]:
                contrast_model_result['MUSIC'][step_1] = doa_hat
                music_succ_ratio[step_1] = np.sum(succ_idx) / theta_set.shape[0]

        RMSE_theta_mean = np.mean(RMSE_theta, axis=-1)
        RMSE_phi_mean = np.mean(RMSE_phi, axis=-1)
        RMSE_theta_std = np.std(RMSE_theta, axis=-1)
        RMSE_phi_std = np.std(RMSE_phi, axis=-1)

        print(RMSE_theta_mean)
        print(RMSE_phi_mean)
        print(f'success ratio of algorithm is {succ_ratio}')
        print(f'MUSIC success ratio is {music_succ_ratio}')

        # save loss
        random_name = str(np.random.rand(1))
        save_array(RMSE_theta_mean, os.path.join(save_path, 'RMSE_theta_mean_' + random_name + '.csv'),
                   index=['snap_' + str(args.snap)],
                   header=['snr_' + str(i) for i in args.snrs])
        save_array(RMSE_phi_mean, os.path.join(save_path, 'RMSE_phi_mean_' + random_name + '.csv'),
                   index=['snap_' + str(args.snap)],
                   header=['snr_' + str(i) for i in args.snrs])

        # plot loss
        loss_1d_plot(RMSE_theta_mean, model_name, args.snrs, 'SNR(db)', False,
                     os.path.join(save_path, f'RMSE_theta_{args.snrs}_{random_name}.png'))
        loss_1d_plot(RMSE_phi_mean, model_name, args.snrs, 'SNR(db)', False,
                     os.path.join(save_path, f'RMSE_phi_{args.snrs}_{random_name}.png'))
        # save model loss
        if not os.path.exists(os.path.join(save_path, 'contrast_model')):
            os.makedirs(os.path.join(save_path, 'contrast_model'))
        for name, rmse in contrast_loss_theta.items():
            rmse_mean = np.mean(rmse, axis=-1)
            save_array(rmse_mean,
                       os.path.join(save_path, 'contrast_model', name + '_rmse_theta_' + random_name + '.csv'),
                       index=['snap_' + str(args.snap)],
                       header=['snr_' + str(i) for i in args.snrs])

        contrast_loss_mean = {key: np.mean(value, axis=-1) for key, value in contrast_loss_theta.items()}

        loss_1d_v_plot(RMSE_theta_mean, model_name, args.snrs, 'SNR(db)', contrast_loss_mean, False,
                       os.path.join(save_path, f'contrast_RMSE_{random_name}_theta.png'))

        contrast_loss_mean = {key: np.mean(value, axis=-1) for key, value in contrast_loss_phi.items()}
        loss_1d_v_plot(RMSE_phi_mean, model_name, args.snrs, 'SNR(db)', contrast_loss_mean, False,
                       os.path.join(save_path, f'contrast_RMSE_{random_name}_phi.png'))
        # plot predict
        plot_predict = False
        if plot_predict:
            for i, snr in enumerate(args.snrs):
                # plot theta
                contrast_plot = {key: value[i, :, 0, :] for key, value in contrast_model_result.items()}
                plot_doa_error(theta_set[:, 0, :], predict_results[i, :, 0, :], contrast_plot,
                               os.path.join(save_path, f'pre_result_snr_{snr}_{random_name}_theta.png'))
                # plot phi
                contrast_plot = {key: value[i, :, 1, :] for key, value in contrast_model_result.items()}
                plot_doa_error(theta_set[:, 1, :], predict_results[i, :, 1, :], contrast_plot,
                               os.path.join(save_path, f'pre_result_snr_{snr}_{random_name}_phi.png'))

        plot_cdf = True
        if plot_cdf:
            for i, snr in enumerate(args.snrs):
                # plot theta
                contrast_plot = {key: value[i, :, 0, :] for key, value in contrast_model_result.items()}
                plot_v_cdf(theta_set[:, 0, :], predict_results[i, :, 0, :], contrast_plot,
                           os.path.join(save_path, f'pre_cdf_snr_{snr}_{random_name}_theta.png'))
                # plot phi
                contrast_plot = {key: value[i, :, 1, :] for key, value in contrast_model_result.items()}
                plot_v_cdf(theta_set[:, 1, :], predict_results[i, :, 1, :], contrast_plot,
                           os.path.join(save_path, f'pre_cdf_snr_{snr}_{random_name}_phi.png'))
                # 计算分位点
                # model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent = \
                #     calculate_cdf_and_quantiles(theta_set, predict_results[i], contrast_plot, quantiles=[80],
                #                                 cal_percent=[10])
                # print(f'condition {i}', model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent,
                #       sep='\n')

        plot_radar = True
        if plot_radar:
            for i, snr in enumerate(args.snrs):
                # plot theta
                contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
                plot_radar_fig(theta_set, predict_results[i], contrast_plot,
                               os.path.join(save_path, f'radar_snr_{snr}_{random_name}.png'))


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # scenario parameters
    parser.add_argument('--M', type=int, default=12)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--snrs', type=list, default=[-20, -15, -10, -5, 0, 5])
    # parser.add_argument('--snrs', type=list, default=[50, 60, 70, 80])
    parser.add_argument('--snap', type=int, default=50)
    parser.add_argument('--theta_range', type=tuple, default=(-180, 180))
    parser.add_argument('--phi_range', type=tuple, default=(0, 60))
    # parser.add_argument('--signal_range', type=tuple, default=((-5, 0.5, 6), (-0.5, 5, 10.5)))
    parser.add_argument('--rho', type=float, default=0)
    parser.add_argument('--repeats', type=int, default=1)

    root_path = os.path.abspath('../../')
    load_root = os.path.join(root_path, 'results', 'vit', f'vit_2D_DOA_origin')
    parser.add_argument('--load_root', type=str, default=load_root)
    parser.add_argument('--save_root', type=str, default=load_root)

    # test device
    parser.add_argument('--device', type=str, default='cuda')

    # model parameters
    # ...
    parser.add_argument('--grid_to_theta', type=bool, default=False)

    args = parser.parse_args()

    main(args)
