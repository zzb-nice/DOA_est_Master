import os
import json
import pandas as pd


def get_all_file(path, suffix='.pth'):
    pth_file_names = []
    # 遍历指定路径
    for filename in os.listdir(path):
        # 检查文件是否以 suffix 结尾
        if filename.endswith(suffix):
            pth_file_names.append(filename)

    return pth_file_names


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_args(argparser, file):
    with open(file, 'w') as f:
        json.dump(vars(argparser), f, indent=4)


def read_csv_results(snap, snr_list, csv_path):
    data = pd.read_csv(csv_path, header=0, index_col=0)

    # Find the row corresponding to the snap
    snap_row = data.loc[f'snap_{snap}']

    if not snap_row.empty:
        snr_columns = [f"snr_{snr}" for snr in snr_list]
        result_data = snap_row[snr_columns].values
        return result_data
    else:
        raise ValueError(f"Snap {snap} not found in the dataset.")


def read_csv_results_snap(snr, snap_list, csv_path):
    data = pd.read_csv(csv_path, header=0, index_col=0)

    # Find the row corresponding to the snap
    snr_row = data.loc[f'snr_{snr}']

    if not snr_row.empty:
        snap_columns = [f"snap_{snap}" for snap in snap_list]
        result_data = snr_row[snap_columns].values
        return result_data
    else:
        raise ValueError(f"Snr {snr} not found in the dataset.")


def load_all_test_result(root, snap, snr_list):
    result_data = {}  # model_name: np.ndarray 形式
    files = os.listdir(root)
    for file in files:
        if file.endswith('.csv'):
            if  file.startswith('RMSE_mean') or file.startswith('model_rmse'):
                result_data['DOA-ViT'] = read_csv_results(snap, snr_list, os.path.join(root, file))  # load the model RMSE

    contrast_model_root = os.path.join(root, 'contrast_model')   # load the contrast model RMSE
    files = os.listdir(contrast_model_root)
    for file in files:
        if file.endswith('.csv'):
            model_name = file.split('_rmse_')[0]
            data = read_csv_results(snap, snr_list, os.path.join(contrast_model_root, file))
            result_data[model_name] = data
    return result_data


def load_all_test_result_snap(root, snr, snap_list):
    result_data = {}  # model_name: np.ndarray 形式
    files = os.listdir(root)
    for file in files:
        if file.startswith('RMSE_mean') and file.endswith('.csv') or file.startswith('model') and file.endswith('.csv') :
            result_data['DOA-ViT'] = read_csv_results_snap(snr, snap_list, os.path.join(root, file))  # load the model RMSE

    contrast_model_root = os.path.join(root, 'contrast_model')   # load the contrast model RMSE
    files = os.listdir(contrast_model_root)
    for file in files:
        if file.endswith('.csv'):
            model_name = file.split('_rmse_')[0]
            data = read_csv_results_snap(snr, snap_list, os.path.join(contrast_model_root, file))
            result_data[model_name] = data
    return result_data


if __name__ == '__main__':
    read_csv_results(10, [-20, -15, -10, -5, 0, 5], '/home/xd/zbb_Code/研二code/DOA_deep_learn/results/M=8/k=3/vit/'
                                                    'vit_6_64*12_0_0_range_snr/random_input_sep_3_snap_10_(-60, 60)/'
                                                    'validation_loss_[0.13627].csv')
