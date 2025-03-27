import numpy as np
import argparse
import os
import json

import torch.nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_creater.file_dataloader import file_array_Dataloader,DictionaryToAttributes
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
    test_type = "random_input"  # random_input,monte_carlo[10,20,30],monte_carlo[10,13,16]
    # filename
    dataset_type = f'file_test1_{test_type}_rho{rho}'  # save path
    load_file = os.path.join(args.dataset_path, f'test_{test_type}_snr_0.npz')  # init the dataloader for num_samples
    dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch', input_type='scm', output_type='doa')
    num_samples = dataloader.num_data
    dataset = ULA_dataset(args.M, -60, 60, 1, args.rho)  # init the dataset for algorithm used
    target_dataset = extend_R(args.M, args.M_ext, args.rho)  # init the ext_dataset for algorithm used

    save_path = os.path.join(args.save_root, dataset_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the parameter setting
    save_args(args, os.path.join(save_path, 'laboratory_set.json'))

    # initialize contrast model
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
    cnn_file = "std_CNN_M_8_k_3"
    CNN_load_path = os.path.join(cnn_path, cnn_file)

    if not os.path.exists(os.path.join(save_path, 'l1_svd_file')):
        os.makedirs(os.path.join(save_path, 'l1_svd_file'))

    # our model
    model = MLP([args.M * (args.M + 1), 128, 128, 128, args.M_ext * (args.M_ext + 1)])
    model_name = 'R_ext_MUSIC'

    # loss
    loss_function = torch.nn.MSELoss()

    RMSE = np.zeros((len(args.snrs), args.repeats))
    succ_ratio = np.zeros(len(args.snrs))
    music_succ_ratio = np.zeros(len(args.snrs))
    l1_svd_succ_ratio = np.zeros(len(args.snrs))
    predict_results = np.zeros((len(args.snrs), num_samples, args.k))

    # record loss of contrast model
    contrast_loss = {key: np.zeros((len(args.snrs), args.repeats))
                     for key, include in zip(contrast_model_name, record_loss) if include}

    contrast_model_result = {key: np.zeros((len(args.snrs), num_samples, args.k))
                             for key, include in zip(contrast_model_name, record_pred) if include}
    # TODO:succ ratio of the model
    for repeat_i in range(args.repeats):
        for step_1, snr in enumerate(args.snrs):
            id1 = f'_snr_{snr}'
            id2 = f'_snr_{(args.snrs[0], args.snrs[-1])}_snap_{args.snap}'
            load_file = os.path.join(args.dataset_path, f'test_{test_type}_snr_{snr}.npz')
            dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch', input_type='scm_vec_include_diag', output_type='doa')

            weight_path = os.path.join(args.load_root, 'weight' + id1 + '.pth')

            state_dict = torch.load(weight_path, map_location=args.device)
            model.load_state_dict(state_dict, strict=True)
            model.to(args.device)

            predict_result = test_R_fit(model, dataloader, args.device)

            # initial the dataset
            dataset = ULA_dataset(args.M, -60, 60, 1, args.rho)  # init the dataset for algorithm used
            DictionaryToAttributes(dataset, dataloader.all_data)  # load the data to dataset

            doa_hat = np.zeros((num_samples, args.k), dtype=np.float32)
            succ_idx = np.zeros(num_samples, dtype=bool)
            R_exds = vec_2_matrix(predict_result, [predict_result.shape[0], args.M_ext, args.M_ext])
            for i, R_exd in enumerate(R_exds):
                succ, doa = post_music.estimate(R_exd, args.k)
                doa_hat[i] = doa
                succ_idx[i] = succ

            # TODO: use rmse or test_loss
            test_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            RMSE[step_1, repeat_i] = test_loss
            print(id1, '.', 'predict_RMSE:', test_loss)

            if repeat_i == 0:
                predict_results[step_1] = doa_hat
                succ_ratio[step_1] = 1  # TODO

            # save .mat file for R construct
            savemat(os.path.join(save_path, f'R_ext_{args.snap}_snr_{snr}.mat'),
                    {'R_ext': R_exds, 'ext_M': args.M_ext,
                     'true_doa': np.stack(dataset.doa),
                     'grid': dataset.signal_grid, 'steer_vec': target_dataset.get_ext_A(dataset.signal_grid, None)})

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

            if record_loss[music_index]:
                contrast_loss['MUSIC'][step_1, repeat_i] = music_loss

            if repeat_i == 0 and record_pred[music_index]:
                contrast_model_result['MUSIC'][step_1] = doa_hat
                music_succ_ratio[step_1] = np.sum(succ_idx) / num_samples

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

            if record_loss[root_music_index]:
                contrast_loss['Root-Music'][step_1, repeat_i] = root_music_loss

            if repeat_i == 0 and record_pred[root_music_index]:
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

            if record_loss[esprit_index]:
                contrast_loss['ESPRIT'][step_1, repeat_i] = esprit_loss

            if repeat_i == 0 and record_pred[esprit_index]:
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

            if record_loss[unity_esprit_index]:
                contrast_loss['Unity-ESPRIT'][step_1, repeat_i] = unity_esprit_loss

            if repeat_i == 0 and record_pred[unity_esprit_index]:
                contrast_model_result['Unity-ESPRIT'][step_1] = doa_hat

            # l1-svd
            l1_svd_index = contrast_model_name.index(r'$\ell_1$-SVD')

            save_file = os.path.join(save_path, 'l1_svd_file', f'{id1}.mat')
            l1_svd.save_used_mat(dataset, save_file)
            succ_idx, doa_hat = l1_svd.predict(save_file,
                                               os.path.join(save_path, 'l1_svd_file', f'predict_snr_{id1}.mat'), args.k,
                                               snr, args.M, args.snap)

            l1_svd_loss = np.sqrt(np.mean((np.array(dataset.doa)[succ_idx, :] - doa_hat[succ_idx, :]) ** 2))
            print(id1, '.', 'l1_svd_RMSE:', l1_svd_loss)

            if record_loss[l1_svd_index]:
                contrast_loss[r'$\ell_1$-SVD'][step_1, repeat_i] = l1_svd_loss
                l1_svd_succ_ratio[step_1] = np.sum(succ_idx) / num_samples

            if repeat_i == 0 and record_pred[l1_svd_index]:
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

            if record_loss[cnn_index]:
                contrast_loss['SPE-CNN'][step_1, repeat_i] = cnn_test_loss

            if repeat_i == 0 and record_pred[cnn_index]:
                contrast_model_result['SPE-CNN'][step_1] = cnn_predict_result

        RMSE_mean = np.mean(RMSE, axis=-1)
        RMSE_std = np.std(RMSE, axis=-1)

        print(RMSE_mean)
        print(f'success ratio of algorithm is {succ_ratio}')
        print(f'MUSIC success ratio is {music_succ_ratio}')
        print(f'$\ell_1$-SVD success ratio is {l1_svd_succ_ratio}')

        # save loss
        random_name = str(np.random.rand(1))
        save_array(RMSE_mean, os.path.join(save_path, 'RMSE_mean_' + random_name + '.csv'),
                   index=['snap_' + str(args.snap)],
                   header=['snr_' + str(i) for i in args.snrs])
        save_array(RMSE_std, os.path.join(save_path, 'RMSE_std_' + random_name + '.csv'),
                   index=['snap_' + str(args.snap)],
                   header=['snr_' + str(i) for i in args.snrs])
        # plot loss
        loss_1d_plot(RMSE_mean, model_name, args.snrs, 'SNR(db)', False,
                     os.path.join(save_path, f'RMSE_{args.snrs}_{random_name}.png'))
        # save model loss
        if not os.path.exists(os.path.join(save_path, 'contrast_model')):
            os.makedirs(os.path.join(save_path, 'contrast_model'))
        for name, rmse in contrast_loss.items():
            rmse_mean = np.mean(rmse, axis=-1)
            save_array(rmse_mean, os.path.join(save_path, 'contrast_model', name + '_rmse_' + random_name + '.csv'),
                       index=['snap_' + str(args.snap)],
                       header=['snr_' + str(i) for i in args.snrs])
        contrast_loss_mean = {key: np.mean(value, axis=-1) for key, value in contrast_loss.items()}

        loss_1d_v_plot(RMSE_mean, model_name, args.snrs, 'SNR(db)', contrast_loss_mean, False,
                       os.path.join(save_path, f'contrast_RMSE_{random_name}.png'))

        # plot predict
        plot_predict = False
        if plot_predict:
            for i, snr in enumerate(args.snrs):
                contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
                plot_v_predict_result(dataset.doa, predict_results[i], contrast_plot,
                                      os.path.join(save_path, f'pre_result_snr_{snr}_{random_name}.png'))

        plot_cdf = False
        if plot_cdf:
            for i, snr in enumerate(args.snrs):
                contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
                plot_v_cdf(dataset.doa, predict_results[i], contrast_plot,
                           os.path.join(save_path, f'pre_cdf_snr_{snr}_{random_name}.png'))
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
    parser.add_argument('--M_ext', type=int, default=32)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--snrs', type=list, default=[-20, -15, -10, -5, 0, 5])
    parser.add_argument('--snap', type=int, default=10)
    parser.add_argument('--signal_range', type=tuple, default=(-60, 60))
    # parser.add_argument('--signal_range', type=tuple, default=((-5, 0.5, 6), (-0.5, 5, 10.5)))
    parser.add_argument('--repeats', type=int, default=1)

    root_path = os.path.abspath('../../')
    rho = 0.5
    parser.add_argument('--rho', type=float, default=rho)
    dataset_path = os.path.join(root_path, 'data', 'ULA_data', 'test', f'M_8_k_3_test1_rho{rho}')
    load_root = os.path.join(root_path, 'article_implement', 'Learning_SPICE', 'results', f'R_fit_M_8_k_3_origin')
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
