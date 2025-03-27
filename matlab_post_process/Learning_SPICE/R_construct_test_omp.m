%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: zbb
% Date: 12/9/2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% change SNR
snap = 10; % number of snapshots
SNR_dB = [-20,-15,-10,-5,0,5]; % SNR values
threshold_vec = [1,1,1,1,1,1];  % make sure the success ratio>90%
if length(threshold_vec) ~= length(SNR_dB)
    error('Error: The lengths of the two vectors are not equal. The program is terminated.')
end

% load data to known N_sim
load_root = '/home/xd/DOA_code/open_source_code/article_implement/Learning_SPICE/results/R_fit_M_8_k_3_origin/file_test1_monte_carlo[10,13,16]/';
file_name = ['R_ext_' num2str(snap) '_snr_' num2str(SNR_dB(1)) '.mat'];
load_file = [load_root file_name];   % load 之后不用生成信号
data = load(load_file);

SOURCE_K = size(data.true_doa,2); % number of sources/targets - Kmax
ULA_N = size(data.R_ext,2);
N_sim = size(data.true_doa,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(-1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

THETA_angles = -60:1:60;

RMSE_RMUSIC = zeros(1,length(threshold_vec));
RMSE_R_cons = zeros(1,length(threshold_vec));
pred_result = zeros(length(threshold_vec),N_sim,SOURCE_K);
succ_l1svd = zeros(1,length(threshold_vec));

for ii=1:length(threshold_vec)
    noise_amp = 1*10^(-SNR_dB(ii)/20);
    threshold = threshold_vec(ii);
    
    % load data
    file_name = ['R_ext_' num2str(snap) '_snr_' num2str(SNR_dB(ii)) '.mat'];
    load_file = [load_root file_name];
    data = load(load_file);
    
    mse_rm_i = zeros(1,N_sim);
    mse_l1svd_i = zeros(1,N_sim);
    succ_l1svd_i = zeros(1,N_sim);
    
    h = waitbar(0, 'Processing...');
    for i=1:N_sim
        Rx = double(squeeze(data.R_ext(i,:,:)));
        ang_gt = double(data.true_doa(i,:).');
        
        % Root-MUSIC estimator 
        doas_sam_rm = sort(rootmusicdoa((Rx+Rx')/2+1*eye(ULA_N), SOURCE_K))';
        ang_sam_rm = sort(doas_sam_rm);
        mse_rm_i(i) = norm(ang_sam_rm - ang_gt)^2;
        
        % l1_SVD
        Rx(eye(size(Rx)) == 1) = 1;   % don't use the diagnal
        [succ,ang_est, sp_est] = R_construct(Rx,ULA_N,threshold,SOURCE_K, THETA_angles,0);
        
        pred_result(ii,i,:) = ang_est;
        mse_l1svd_i(i) = norm(sort(ang_est) - ang_gt)^2;
        if succ
            succ_l1svd_i(i) = 1;
            % mse_l1svd_i(i) = norm(sort(ang_est) - ang_gt)^2;  % 只计算succ的
        end
        waitbar(i / N_sim, h, sprintf('Progress: %.2f%%', (i / N_sim) * 100));
    end
    close(h);
    
    succ_l1svd_i = squeeze(logical(succ_l1svd_i));
    succ_l1svd(ii) = sum(succ_l1svd_i)/length(succ_l1svd_i);
    RMSE_RMUSIC(ii) = sqrt(sum(mse_rm_i/SOURCE_K)/N_sim);
    RMSE_R_cons(ii) = sqrt(sum(mse_l1svd_i/SOURCE_K/N_sim));
    % RMSE_R_cons(ii) = sqrt(sum(mse_l1svd_i/SOURCE_K/sum(succ_l1svd_i)));
end

figure(1);

plot(SNR_dB, RMSE_RMUSIC ,'-o', 'LineWidth', 2);
hold on;
plot(SNR_dB, RMSE_R_cons,'-o', 'LineWidth', 2);
hold off;
grid on;
legend('R-MUSIC','R-cons');

% 保存结果
save([load_root 'pred_results_and_RMSE.mat'],"RMSE_R_cons","pred_result")
exportgraphics(gcf,[load_root 'RMSE.png']);