
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adapted from the code by Georgios K. Papageorgiou.
% Date: 12/9/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load_root = '/home/xd/zbb_Code/研二code/DOA_deep_learn/article_implement/ASL/results/ASL_run2_drop_out_0/used_random_input_sep_3_snrset_0_(-60, 60)';
snap = 42; % number of snapshots
SNR_dB = 5; % SNR values
file_name = ['construct_snap_' num2str(snap) '_snr_' num2str(SNR_dB) '.mat'];
load_file = [load_root file_name];   % load 之后不用生成信号
data = load(load_file);

threshold_vec = 0.1:0.1:1;
SOURCE_K = size(data.true_doa,2); % number of sources/targets - Kmax
ULA_N = size(data.R,2);
N_sim = size(data.true_sep,1);
N_sim = 500;   % test 前500个
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
THETA_angles = -3:0.1:3;

SOURCE_amp = ones(SOURCE_K,1).^2;
noise_amp = min(SOURCE_amp)*10^(-SNR_dB/20);

RMSE_RMUSIC = zeros(1,length(threshold_vec));
RMSE_l1SVD = zeros(1,length(threshold_vec));
succ_ratio = zeros(1,length(threshold_vec));

% addpath("/home/xd/zbb_Code/matlab/compress_sensing/ASL/cs_method/R_construct/OMP")
addpath("/home/xd/zbb_Code/matlab/compress_sensing/compress_sensing_algorithm/R_construct/OMP");
for ii=1:length(threshold_vec)
    threshold  = threshold_vec(ii);
    
    % save parameter for parfor
    mse_rm_i = zeros(1,N_sim);
    mse_l1svd_i = zeros(1,N_sim);
    succ_l1svd_i = zeros(1,N_sim);
    
    for i=1:N_sim
    Rx = double(squeeze(data.R(i,:,:)));
    ang_gt = double(data.true_doa(i,:).');
    pred_sep = double(data.pred_sep(i,:));
    % Root-MUSIC estimator 
    doas_sam_rm = sort(rootmusicdoa(Rx, SOURCE_K))';
    ang_sam_rm = sort(doas_sam_rm);
    % RMSE calculation - degrees
    mse_rm_i(i) = norm(ang_sam_rm - ang_gt)^2;
   
    % l1_SVD
    % [succ,ang_est_l1svd, sp_est] = ASL_R_construct_k_1(Rx,ULA_N,threshold,THETA_angles,pred_sep,noise_amp^2);
    [succ,ang_est_l1svd, sp_est] = R_construct(Rx,ULA_N,threshold,2,THETA_angles,noise_amp^2);
    if succ
        succ_l1svd_i(i) = 1;
        % mse_l1svd_i(i) = norm([ang_est_l1svd;ang_est_l1svd+pred_sep] - ang_gt)^2;
        mse_l1svd_i(i) = norm(ang_est_l1svd - ang_gt)^2;
    end
        
    end
   
    RMSE_RMUSIC(ii) = sqrt(sum(mse_rm_i/SOURCE_K)/N_sim);
    RMSE_l1SVD(ii) = sqrt(sum(mse_l1svd_i/SOURCE_K)/sum(succ_l1svd_i));
    succ_ratio(ii) = sum(succ_l1svd_i)/N_sim;
    
end

time_tot = toc/60; % in minutes
disp(['success ratio is' num2str(succ_ratio)])

figure(1);

plot(threshold_vec, RMSE_RMUSIC,'-o', 'LineWidth', 2);
hold on;
plot(threshold_vec, RMSE_l1SVD,'-o', 'LineWidth', 2);
hold off;
grid on;
legend('R-MUSIC','ASL2', 'interpreter','latex');
% exportgraphics(gcf,'figure.png', 'Resolution', 300)

[val,ind] = min(RMSE_l1SVD);
best_thresh = threshold_vec(ind);
disp(['The best threshold value is ',num2str(best_thresh)]);
RMSE_l1SVD(ind)

% % 文件形式，方便服务器操作
% fileID = fopen('output.txt', 'a');  % 使用 'a' 以追加方式打开文件
% fprintf(fileID, 'RMSE_l1SVD: %s\n', num2str(RMSE_l1SVD));
% fclose(fileID);
