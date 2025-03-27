
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adapted from the code by Georgios K. Papageorgiou.
% Date: 12/9/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
tic;
rng(14);
% firstly transform codes in rootmusicdoa
% u = doas/(-2*pi*elSpacing); to u = -doas/(-2*pi*elSpacing);
% to keep the steer vector in matlab the same as python
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% change SNR
snap = 10; % number of snapshots
SNR_dB = [-20,-15,-10,-5,0,5]; % SNR values
threshold_vec = [100,100,100,100,100,100];  % make sure the success ratio>90%
if length(threshold_vec) ~= length(SNR_dB) 
    error('Error: The lengths of the two vectors are not equal. The program is terminated.')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(-1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
THETA_angles = -60:1:60;

% load data to known N_sim
load_root = '/home/xd/DOA_code/open_source_code/article_implement/ASL/results/ASL_M_8_k_3/file_test1_random_input_rho0.5/';
file_name = ['construct_snap_' num2str(snap) '_snr_' num2str(SNR_dB(1)) '.mat'];
load_file = [load_root file_name];   % load 之后不用生成信号
data = load(load_file);

SOURCE_K = size(data.true_doa,2); % number of sources/targets - Kmax
ULA_N = size(data.R,2);
N_sim = size(data.true_sep,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RMSE_RMUSIC = zeros(1,length(threshold_vec));
RMSE_R_cons = zeros(1,length(threshold_vec));
succ_ratio = zeros(1,length(threshold_vec));

pred_result = zeros(length(threshold_vec),N_sim,SOURCE_K);
succ_l1svd = zeros(1,length(threshold_vec));  % 矩阵表明是否成功计算

for ii=1:length(threshold_vec)
    noise_amp = 1*10^(-SNR_dB(ii)/20);
    threshold  = threshold_vec(ii);

    % load data
    file_name = ['construct_snap_' num2str(snap) '_snr_' num2str(SNR_dB(ii)) '.mat'];
    load_file = [load_root file_name];   % load 之后不用生成信号
    data = load(load_file);

    SOURCE_K = size(data.true_doa,2); % number of sources/targets - Kmax
    ULA_N = size(data.R,2);
    N_sim = size(data.true_sep,1);
   
    % save parameter for parfor
    mse_rm_i = zeros(1,N_sim);
    mse_l1svd_i = zeros(1,N_sim);
    succ_l1svd_i = zeros(1,N_sim);

    h = waitbar(0, 'Processing...'); % 初始化进度条
    for i=1:N_sim

    Rx = double(squeeze(data.R(i,:,:)));
    ang_gt = double(data.true_doa(i,:).');
    pred_sep = double(data.pred_sep(i,:));

    % Root-MUSIC estimator 
    doas_sam_rm = sort(rootmusicdoa((Rx+Rx')/2+1*eye(ULA_N), SOURCE_K))';
    ang_sam_rm = sort(doas_sam_rm);
    % RMSE calculation - degrees
    mse_rm_i(i) = norm(ang_sam_rm - ang_gt)^2;
   
    % l1_SVD
    % ang_sep = (ang_gt(2:end)-ang_gt(1))';  % 用真正的角度真实值作为先验
    ang_sep = pred_sep;  % 实际中必须用预测角度真实值作为先验
    [succ,ang_est, sp_est] = ASL_R_construct_k_3(Rx,ULA_N,threshold, THETA_angles,ang_sep,noise_amp^2);
    % [succ,ang_est, sp_est] = ASL_R_construct_k_n(Rx,ULA_N,threshold, THETA_angles,pred_sep,noise_amp^2,SOURCE_K);

    if succ
        succ_l1svd_i(i) = 1;
        mse_l1svd_i(i) = norm(sort(ang_est) - ang_gt)^2;
        pred_result(ii,i,:) = ang_est;
    end
    waitbar(i / N_sim, h, sprintf('Progress: %.2f%%', (i / N_sim) * 100)); 
    end
    close(h);

    succ_l1svd(ii) = sum(succ_l1svd_i)/length(succ_l1svd_i);
    RMSE_RMUSIC(ii) = sqrt(sum(mse_rm_i/SOURCE_K)/N_sim);
    RMSE_R_cons(ii) = sqrt(sum(mse_l1svd_i/SOURCE_K)/sum(succ_l1svd_i));
    
end
disp("l1_svd success ratio is "+string(succ_l1svd(:)))
time_tot = toc/60; % in minutes

figure(1);

plot(SNR_dB, RMSE_RMUSIC ,'-o', 'LineWidth', 2);
hold on;
plot(SNR_dB, RMSE_R_cons,'-o', 'LineWidth', 2);
hold off;
grid on;
legend('R-MUSIC','R-cons');

save([load_root 'pred_results_and_RMSE.mat'],"RMSE_R_cons","pred_result")
exportgraphics(gcf,[load_root 'RMSE.png']);