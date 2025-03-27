%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: zbb
% Date: 12/9/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
snap = 10; % number of snapshots
threshold_vec = 1:2:10;
SNR_dB = -10; % SNR values

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(-1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
THETA_angles = -60:1:60;  % grid

% 添加文件加载路径和参数
load_root = '/home/xd/DOA_code/open_source_code/article_implement/R_fit_method/results/R_fit_M_8_k_3/file_test1_random_input/';  % 请修改为您的数据文件夹路径
file_name = ['R_ext_' num2str(snap) '_snr_' num2str(SNR_dB) '.mat'];
load_file = [load_root file_name];
data = load(load_file);

% 从加载的数据中获取参数
SOURCE_K = size(data.true_doa,2);
ULA_N = size(data.R_ext,2);
N_sim = size(data.true_doa,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RMSE_RMUSIC = zeros(1,length(threshold_vec));
RMSE_R_cons = zeros(1,length(threshold_vec));
succ_ratio = zeros(1,length(threshold_vec));

succ_l1svd = zeros(length(threshold_vec),N_sim);  % 矩阵表明是否成功计算

for ii=1:length(threshold_vec)
    threshold  = threshold_vec(ii);
    
    % 计算噪声功率
    noise_power = 1*10^(-SNR_dB/20);
    
    % save parameter for parfor
    rmse_rm_i = zeros(1,N_sim);
    rmse_R_cons_i = zeros(1,N_sim);
    succ_R_cons_i = zeros(1,N_sim);
    
    h = waitbar(0, 'Processing...'); % 添加进度条
    for i=1:N_sim
        % 从加载的数据中获取协方差矩阵和真实角度
        R = double(squeeze(data.R_ext(i,:,:)));
        ang_gt = double(data.true_doa(i,:).');
        
        % % 检查R是否为Hermitian矩阵
        % if ~isequal(R, R')
        %     error('Error: 协方差矩阵R不是Hermitian矩阵！');
        % end
        
        % Root-MUSIC estimator 
        doas_sam_rm = sort(rootmusicdoa((R+R')/2+1*eye(ULA_N), SOURCE_K))';
        ang_sam_rm = sort(doas_sam_rm);
        rmse_rm_i(i) = norm(ang_sam_rm - ang_gt)^2;
       
        % l1_SVD
        % [succ,ang_est_R_cons, sp_est] = R_construct(R,ULA_N,threshold,SOURCE_K, THETA_angles,noise_power^2);
        R(eye(size(R)) == 1) = 1;   % don't use the diagnal
        [succ,ang_est_R_cons, sp_est] = R_construct(R,ULA_N,threshold,SOURCE_K, THETA_angles,0);
        if succ
            succ_R_cons_i(i) = 1;
            rmse_R_cons_i(i) = norm(sort(ang_est_R_cons) - ang_gt)^2;
        end
        
        waitbar(i / N_sim, h, sprintf('Progress: %.2f%%', (i / N_sim) * 100));
    end
    close(h);
    
    RMSE_RMUSIC(ii) = sqrt(sum(rmse_rm_i/SOURCE_K)/N_sim);
    RMSE_R_cons(ii) = sqrt(sum(rmse_R_cons_i/SOURCE_K)/sum(succ_R_cons_i));
    succ_ratio(ii) = sum(succ_R_cons_i)/N_sim;
end

time_tot = toc/60; % in minutes
disp(['success ratio is' num2str(succ_ratio)])

figure(1);
plot(threshold_vec, RMSE_RMUSIC,'-o', 'LineWidth', 2);
hold on;
plot(threshold_vec, RMSE_R_cons,'-o', 'LineWidth', 2);
hold off;
grid on;
legend('R-MUSIC','R-cons');

[val,ind] = min(RMSE_R_cons);
best_thresh = threshold_vec(ind);
disp(['The best threshold value is ',num2str(best_thresh)]);
disp(['Best RMSE_R_cons: ', num2str(RMSE_R_cons(ind))]);

% 添加结果保存
% save([load_root 'optimize_threshold_results.mat'], 'RMSE_RMUSIC', 'RMSE_R_cons', 'succ_ratio', 'best_thresh');
exportgraphics(gcf,[load_root 'optimize_threshold_RMSE.png']);