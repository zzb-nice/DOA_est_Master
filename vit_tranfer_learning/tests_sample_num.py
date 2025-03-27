import numpy as np
import argparse
import os
import json

import torch.nn
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_creater.signal_datasets import ULA_dataset, array_Dataloader
from data_creater.Create_k_source_dataset import Create_determined_sep_doas, Create_random_k_input_theta, \
    Create_datasets
from data_creater.Create_classic_test_dataset import Create_monte_carlo_theta, Create_equal_separate_theta
from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot, loss_1d_v_plot
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
    # filename
    dataset_type = 'test_random_input_sep_3_rho1' + '_' + str(
        args.signal_range)  # random_input_sep_3,monte_carlo_[10, 13, 16],equal_sep_2,several_sep,plot
    dataset = ULA_dataset(args.M, -90, 90, 1, args.rho)
    theta_set = Create_random_k_input_theta(args.k, args.signal_range[0],
                                            args.signal_range[1], 5000, min_delta_theta=3)
    # theta_set = Create_random_k_input_theta(args.k, (-5, 10.5, 26),
    #                                         (-0.5, 15, 30.5), 100, min_delta_theta=3)
    # theta_set = Create_equal_separate_theta(args.k, args.signal_range[0],
    #                                         args.signal_range[1], delta_theta_list=[[10, 10]], step=0.5)
    # theta_set = Create_monte_carlo_theta(all_theta=[np.array([10.1, 13.3])], repeat_num=1000)
    # theta_set = Create_monte_carlo_theta(all_theta=[np.array([10, 20, 30])], repeat_num=5000)

    save_path = os.path.join(args.root, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the parameter setting
    save_args(args, os.path.join(save_path, 'laboratory_set.json'))

    # initialize contrast model
    contrast_model_name = ['MUSIC', 'Root-Music', 'ESPRIT', 'Unity-ESPRIT']
    record_loss = [True, True, True, True]
    record_pred = [True, True, True, True]

    music = Music(dataset.get_A)
    root_music = Root_Music(dataset.get_theta_fromz)
    esprit = ESPRIT(dataset.get_theta_fromz, args.M)
    unity_esprit = Unity_ESPRIT(dataset.get_theta_fromz, args.M)

    # our model
    embeding_dim = 768
    model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
                              out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)
    # model = VisionTransformer(embed_layer=scm_embeding(args.M, embeding_dim), embed_dim=embeding_dim,
    #                           out_dims=121, drop_ratio=0, attn_drop_ratio=0, sp_mode=args.grid_to_theta,
    #                           start_angle=-60, end_angle=60, step=1)
    model_name = 'DOA-ViT'
    # model = KAN([56, 128, 128, 128, 121], 5, 3, device=args.device, sp_mode=args.grid_to_theta,
    #             start_angle=-60, end_angle=60, step=1)
    # model_name = 'KAN'
    # loss
    loss_function = torch.nn.MSELoss()

    RMSE = np.zeros((len(args.train_nums), args.repeats))
    succ_ratio = np.zeros(len(args.train_nums))
    music_succ_ratio = np.zeros(len(args.train_nums))
    predict_results = np.zeros((len(args.train_nums), theta_set.shape[0], args.k))

    # record loss of contrast model
    contrast_loss = {key: np.zeros((len(args.train_nums), args.repeats))
                     for key, include in zip(contrast_model_name, record_loss) if include}

    contrast_model_result = {key: np.zeros((len(args.train_nums), theta_set.shape[0], args.k))
                             for key, include in zip(contrast_model_name, record_pred) if include}
    # TODO:succ ratio of the model
    for repeat_i in range(args.repeats):
        for step_1, train_num in enumerate(args.train_nums):
            id1 = f'_train_sample_num_{train_num}'
            dataset = ULA_dataset(args.M, -90, 90, 1, args.rho)
            # bachsize should be divisible by total number, set to 1 while num of theta is not sure
            Create_datasets(dataset, args.k, theta_set, batch_size=50, snap=args.snap, snr=args.snr)
            dataloader = array_Dataloader(dataset, 256, False, 'torch', 'scm', 'doa')
            # dataloader = array_Dataloader(dataset, 256, False, 'torch', 'scm_vec', 'doa')

            weight_path = os.path.join(args.root, 'weight' + id1 + '.pth')

            state_dict = torch.load(weight_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=True)
            # model.load_state_dict(state_dict, strict=False)  # TODO： 后续重新train一下
            model.to(args.device)

            suc_ratio, test_loss, predict_result = test(model, dataloader, loss_function, args.device, args.grid_to_theta, args.k)
            RMSE[step_1, repeat_i] = test_loss

            if repeat_i == 0:
                predict_results[step_1] = predict_result
                succ_ratio[step_1] = 1  # TODO

            # music algorithm
            # directly iterate the data in dataset
            music_index = contrast_model_name.index('MUSIC')
            doa_hat = np.zeros((theta_set.shape[0], args.k), dtype=np.float32)
            # remove the results from calculate RMSE if model return False
            succ_idx = np.zeros(theta_set.shape[0], dtype=bool)
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
                music_succ_ratio[step_1] = np.sum(succ_idx)/theta_set.shape[0]

            # root_music algorithm
            root_music_index = contrast_model_name.index('Root-Music')
            doa_hat = np.zeros((theta_set.shape[0], args.k), dtype=np.float32)
            succ_idx = np.zeros(theta_set.shape[0], dtype=bool)
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

            # ESPRIT algorithm
            esprit_index = contrast_model_name.index('ESPRIT')
            doa_hat = np.zeros((theta_set.shape[0], args.k), dtype=np.float32)
            succ_idx = np.zeros(theta_set.shape[0], dtype=bool)
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

            # Unity-ESPRIT algorithm
            unity_esprit_index = contrast_model_name.index('Unity-ESPRIT')
            doa_hat = np.zeros((theta_set.shape[0], args.k), dtype=np.float32)
            succ_idx = np.zeros(theta_set.shape[0], dtype=bool)
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

    RMSE_mean = np.mean(RMSE, axis=-1)
    RMSE_std = np.std(RMSE, axis=-1)

    print(RMSE_mean)
    print(f'success ratio of algorithm is {succ_ratio}')
    print(f'MUSIC success ratio is {music_succ_ratio}')

    # save loss
    random_name = str(np.random.rand(1))
    save_array(RMSE_mean, os.path.join(save_path, 'RMSE_mean_' + random_name + '.csv'),
               index=['snr_' + str(args.snr) + '_snap_' + str(args.snap)],
               header=['train_num_' + str(i) for i in args.train_nums])
    save_array(RMSE_std, os.path.join(save_path, 'RMSE_std_' + random_name + '.csv'),
               index=['snr_' + str(args.snr) + '_snap_' + str(args.snap)],
               header=['train_num_' + str(i) for i in args.train_nums])
    # plot loss
    loss_1d_plot(RMSE_mean, model_name, args.train_nums, 'sample number', False,
                 os.path.join(save_path, f'RMSE_{args.train_nums}_{random_name}.png'))
    # save model loss
    if not os.path.exists(os.path.join(save_path, 'contrast_model')):
        os.makedirs(os.path.join(save_path, 'contrast_model'))
    for name, rmse in contrast_loss.items():
        rmse_mean = np.mean(rmse, axis=-1)
        save_array(rmse_mean, os.path.join(save_path, 'contrast_model', name + '_rmse_' + random_name + '.csv'),
                   index=['snr_' + str(args.snr) + '_snap_' + str(args.snap)],
                   header=['train_num_' + str(i) for i in args.train_nums])
    contrast_loss_mean = {key: np.mean(value, axis=-1) for key, value in contrast_loss.items()}

    loss_1d_v_plot(RMSE_mean, model_name, args.train_nums, 'sample number', contrast_loss_mean, False,
                   os.path.join(save_path, f'contrast_RMSE_{random_name}.png'))

    # plot predict
    plot_predict = False
    if plot_predict:
        for i, snr in enumerate(args.snrs):
            contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
            plot_v_predict_result(theta_set, predict_results[i], contrast_plot,
                           os.path.join(save_path, f'pre_result_snr_{snr}_{random_name}.png'))

    plot_cdf = False
    if plot_cdf:
        for i, snr in enumerate(args.snrs):
            contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
            plot_v_cdf(theta_set, predict_results[i], contrast_plot,
                       os.path.join(save_path, f'pre_cdf_snr_{snr}_{random_name}.png'))
            model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent = \
                calculate_cdf_and_quantiles(theta_set, predict_results[i], contrast_plot,quantiles=[80],cal_percent=[10])
            print(f'condition {i}', model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent, sep='\n')


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
    # parser.add_argument('--signal_range', type=tuple, default=((-5, 0.5, 6), (-0.5, 5, 10.5)))
    parser.add_argument('--rho', type=float, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--train_nums', type=list, default=[20, 50, 100, 200, 300, 500, 800, 1000])

    root_path = os.path.abspath('../')
    root = os.path.join(root_path, 'results', 'vit', f'vit_M_8_k_3_transfer_learning_various_num_rho_1(2)')
    parser.add_argument('--root', type=str, default=root)

    # test device
    parser.add_argument('--device', type=str, default='cuda')

    # model parameters
    # ...
    parser.add_argument('--grid_to_theta', type=bool, default=False)

    args = parser.parse_args()

    main(args)
