import matlab.engine
import numpy as np
from scipy.io import savemat, loadmat


class matlab_l1_svd:
    def __init__(self, m_file, start=-60, end=60, step=1):
        # 启动 MATLAB 引擎
        self.eng = matlab.engine.start_matlab()
        self.eng.eval("disp('MATLAB has been start')", nargout=0)
        self.eng.cd(m_file, nargout=0)
        self._search_grid = np.arange(start, end + 0.001, step)

    def save_used_mat(self, dataset, save_root):
        savemat(save_root, {'y_t': np.stack(dataset.y_t, 0), 'grid': self._search_grid,
                            'steer_vec': dataset.get_A(self._search_grid, None),
                            'true_doa': np.stack(dataset.doa)})

    def predict(self, input_mat, out_mat, k, snr, M, snap, return_sp=False):
        result = self.eng.python_call_l1_SVD_omp_plus(input_mat, out_mat, k, snr, M, snap)
        # result = self.eng.python_call_l1_SVD_snap(input_mat, out_mat, k, snr, M, snap)

        load_result = loadmat(out_mat)
        if return_sp:
            return load_result['succ_vec'].squeeze(0).astype(np.bool_), load_result['est_DOA'], load_result['est_Ps']
        else:
            return load_result['succ_vec'].squeeze(0).astype(np.bool_), load_result['est_DOA']
