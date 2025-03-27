import numpy as np
import argparse
import os
import scipy.io as sio

from data_creater.file_dataloader import file_array_Dataloader
from data_save.save_csv.loss_save import save_array
from data_save.plot.plot_loss import loss_1d_plot, loss_1d_v_plot
from data_save.plot.plot_pre_result import plot_v_predict_result, plot_doa_error
from data_save.plot.plot_CDF import plot_v_cdf, calculate_cdf_and_quantiles


def main(args):
    # 加载标准数据
    load_file = os.path.join(args.dataset_path, f'test_sep_{args.intervals[0]}.npz')
    dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch', input_type='scm', output_type='doa')
    ground_truth = dataloader.all_data['doa']

    # 创建保存路径
    save_path = args.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_name = 'DOA-ViT'
    load_path = os.path.join(args.root_path,
                             f'results/vit/vit_M_8_k_3_combine/equal_sep_rho1_snr0_(-60, 60)/model_predict_results.npy')
    # 读取预测结果
    model_pred = np.load(load_path)
    assert model_pred.shape[0] == len(args.intervals), 'error input size'

    # When the interval is set differently, the length of dataset is different but must not exceed max_len
    max_len = 2000

    var_len = []
    step = 0.5
    for interval in args.intervals:
        var_len.append((args.signal_range[1] - args.signal_range[0] - np.sum(interval[0])) // step )
    var_len = np.array(var_len).astype(np.int32)

    # 读取各个模型的预测结果
    contrast_model_name = ['transfer learning DOA-ViT', 'MUSIC', 'Unity-ESPRIT', r'$\ell_1$-SVD', 'SPE-CNN', 'ASL-2', 'SubspaceNet', r'Learning $\ell_1$-SVD']
    load_root = [
        os.path.join(args.root_path,
                     f'results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample500/equal_sep_rho1_snr0_(-60, 60)/model_predict_results.npy'),
        os.path.join(args.root_path,
                     f'results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample500/equal_sep_rho1_snr0_(-60, 60)/MUSIC_predict_results.npy'),
        os.path.join(args.root_path,
                     f'results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample500/equal_sep_rho1_snr0_(-60, 60)/Unity-ESPRIT_predict_results.npy'),
        os.path.join(args.root_path,
                     f'results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample500/equal_sep_rho1_snr0_(-60, 60)/$\ell_1$-SVD_predict_results.npy'),
        os.path.join(args.root_path,
                     f'results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample500/equal_sep_rho1_snr0_(-60, 60)/SPE-CNN_predict_results.npy'),
        os.path.join(args.root_path,
                     f'article_implement/ASL/results/ASL_M_8_k_3/equal_sep_snr_0_rho1_(-60, 60)/pred_results_and_RMSE.mat'),
        os.path.join(args.root_path,
                     f'article_implement/SubspaceNet/results/SubspaceNet_M_8_k_3_snap_10/equal_sep_snr0_rho1_(-60, 60)/model_predict_results.npy'),
        os.path.join(args.root_path,
                     f'article_implement/Learning_SPICE/results/R_fit_M_8_k_3_origin/equal_sep_snr0_rho1_(-60, 60)/pred_results_and_RMSE.mat')
    ]
    contrast_model_result = {}
    contrast_model_loss = {}

    # 读取对比模型的预测结果
    for name, load_file in zip(contrast_model_name, load_root):
        if os.path.exists(load_file):
            if load_file.endswith('.npy'):
                contrast_model_result[name] = np.load(load_file)
            elif load_file.endswith('.mat'):
                mat_data = sio.loadmat(load_file)
                # 假设.mat文件中的预测结果存储在'pred_results'键下
                contrast_model_result[name] = mat_data['pred_result']
        else:
            print(f"Warning: File not found for {name}: {load_file}")

    # # 计算并保存模型的RMSE
    # model_loss = np.zeros(len(args.intervals))
    # model_succ_ratio = np.zeros(len(args.intervals))
    # for i in range(len(args.intervals)):
    #     valid_idx = ~np.isnan(model_pred[i, :var_len[i]]).any(axis=1)
    #     model_succ_ratio[i] = np.sum(valid_idx) / var_len[i]
    #     model_loss[i] = np.sqrt(np.mean((ground_truth[valid_idx] - model_pred[i][valid_idx]) ** 2))
    #
    # print(f'Model success ratio: {model_succ_ratio}')
    #
    # # 计算其他模型的RMSE（如果没有在.mat文件中提供）
    # contrast_succ_ratio = {}
    # for name, pred in contrast_model_result.items():
    #     if name not in contrast_model_loss:
    #         loss = np.zeros(len(args.snrs))
    #         succ_ratio = np.zeros(len(args.snrs))
    #         for i in range(len(args.snrs)):
    #             valid_idx = ~np.isnan(pred[i]).any(axis=1)
    #             succ_ratio[i] = np.sum(valid_idx) / len(ground_truth)
    #             loss[i] = np.sqrt(np.mean((ground_truth[valid_idx] - pred[i][valid_idx]) ** 2))
    #         contrast_model_loss[name] = loss
    #         contrast_succ_ratio[name] = succ_ratio
    #         print(f'{name} success ratio: {succ_ratio}')
    #
    # # 打印每个SNR下的成功率
    # print("\nSuccess ratios for each SNR:")
    # for i, snr in enumerate(args.snrs):
    #     print(f'\nSNR = {snr}:')
    #     print(f'Model: {model_succ_ratio[i]:.3f}')
    #     for name, ratio in contrast_succ_ratio.items():
    #         print(f'{name}: {ratio[i]:.3f}')
    #
    # # 保存loss
    # random_name = str(np.random.rand(1))
    # save_array(model_loss, os.path.join(save_path, 'model_rmse_' + random_name + '.csv'),
    #            index=[f'snap_{args.snap}'],
    #            header=['snr_' + str(i) for i in args.snrs])
    #
    # for name, loss in contrast_model_loss.items():
    #     save_array(loss, os.path.join(save_path, f'{name}_rmse_' + random_name + '.csv'),
    #                index=[f'snap_{args.snap}'],
    #                header=['snr_' + str(i) for i in args.snrs])
    #
    # # 绘制loss对比图
    # loss_1d_v_plot(model_loss, model_name, args.snrs, 'SNR(db)', contrast_model_loss, False,
    #                os.path.join(save_path, f'contrast_RMSE_{random_name}.png'))

    # 为每个sep画图
    for i, sep in enumerate(args.intervals):
        # 加载真实DOA数据
        load_file = os.path.join(args.dataset_path, f'test_sep_{args.intervals[i]}.npz')
        dataloader = file_array_Dataloader(load_file, 256, False, load_style='torch', input_type='scm',output_type='doa')
        ground_truth = dataloader.all_data['doa']

        contrast_plot = {key: value[i][:var_len[i]] for key, value in contrast_model_result.items()}
        plot_v_predict_result(ground_truth[:var_len[i]], model_pred[i][:var_len[i]], contrast_plot,
                              os.path.join(save_path, f'pre_result_sep_{sep}.png'))
        # plot_doa_error(ground_truth[start_idx:end_idx], model_pred[i][start_idx:end_idx], contrast_plot,
        #                os.path.join(save_path, f'pre_result_snr_{snr}.png'))

        # # CDF图
        # contrast_plot = {key: value[i] for key, value in contrast_model_result.items()}
        # plot_v_cdf(ground_truth, model_pred[i], contrast_plot,
        #            os.path.join(save_path, f'pre_cdf_snr_{snr}.png'))
        #
        # # 计算分位数和百分比
        # model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent = \
        #     calculate_cdf_and_quantiles(ground_truth, model_pred[i], contrast_plot,
        #                                 quantiles=[90], cal_percent=[10])
        # print(f'SNR {snr}:')
        # print('Model quantile:', model_quantile)
        # print('Contrast quantile:', contrast_quantile)
        # print('Model percentage:', model_cal_percent)
        # print('Contrast percentage:', contrast_cal_percent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    test_type = 'sep'
    file_test_id = 'file_test1'
    # 数据路径参数
    root_path = os.path.abspath('../../../')
    snap = 10
    snr = 0
    rho = 1
    dataset_path = os.path.join(root_path, 'data', 'ULA_data', 'test', f'M_8_k_3_snap_{snap}_snr_{snr}_rho{rho}_min_sep')  # 注意要修改数据路径

    parser.add_argument('--root_path', type=str, default=root_path)
    parser.add_argument('--dataset_path', type=str, default=dataset_path)

    # 测试类型参数
    parser.add_argument('--test_type', type=str, default=test_type,
                        help='monte_carlo[10,13,16] or monte_carlo[10,20,30] or random_input')
    parser.add_argument('--file_test_id', type=str, default=file_test_id,
                        help='file_test1 or file_test2')

    # 其他参数
    parser.add_argument('--snr', type=list, default=snr)
    parser.add_argument('--snap', type=int, default=snap)
    parser.add_argument('--rho', type=float, default=rho)
    parser.add_argument('--signal_range', type=tuple, default=(-60, 60))
    parser.add_argument('--intervals', type=list,
                        default=[[[4, 4]], [[6, 6]], [[8, 8]], [[10, 10]], [[12, 12]], [[14, 14]]])

    args = parser.parse_args()

    args.save_root = os.path.join(root_path, 'data_save', 'vit', 'sep_snr_0')

    main(args)
