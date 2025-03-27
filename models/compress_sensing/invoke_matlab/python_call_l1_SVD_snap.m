function [succ_vec, est_DOA, est_Ps] = python_call_l1_SVD_snap(load_root,save_root,SOURCE_K,snr,M,snap)
% l1_svd is optimized for ULA M=8,k=3,snr=[-20, -15, -10, -5, 0, 5],snap=10
% and min_sep=2, sep_range=(-60,60)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if M==8 && SOURCE_K==3 && snr==-10
    threshold_vec = [0.1,13,21,41,45,50];
    snap_dict=[1,5,10,30,50,100];
end
idx = snap_dict == snap;
threshold = threshold_vec(idx);
disp("select threshold: "+string(threshold))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load_data = load(load_root);
A_dic = double(load_data.steer_vec);  % transform to 双精度
% The grid and dictionary for the compressed sensing method
THETA_angles = load_data.grid;
ULA_N = size(load_data.y_t,2);
NGrids = length(THETA_angles);
Y_all = double(load_data.y_t);
% operator if snap==1
if ismatrix(Y_all)
    Y_all = cat(3,Y_all);
end

% data to store
num_samples = size(load_data.y_t,1);
succ_vec = false(1,num_samples);
est_DOA = zeros(num_samples,SOURCE_K);
est_Ps = zeros(num_samples,length(THETA_angles));

for num_sample=1:num_samples
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ndims(Y_all(num_sample,:,:))==3   % operate snap==1
        Y = squeeze(Y_all(num_sample,:,:));
    elseif ismatrix(Y_all(num_sample,:,:))
        Y = Y_all(num_sample,:,:).';
    end
    % Calculate the \ell_2,1 SVD
    [~,~,V] = svd(Y);
    Ydr = Y*V;
    
    % Orthogonal Matching Pursuit (OMP) implementation
    S_est_dr = zeros(NGrids, size(Y, 2)); % Initialize sparse solution
    residual = Ydr;                       % Initialize residual
    selected_indices = [];                % Support set
    
    for k = 1:100
        % 1. Correlation step: find the index of the most correlated column
        correlations = sum(abs(A_dic' * residual),2);
        [~, idx] = max(correlations);
        selected_indices = [selected_indices, idx];
    
        % 2. Least-squares update on the selected support
        A_selected = A_dic(:, selected_indices); % Submatrix of selected atoms
        S_ls = pinv(A_selected) * Ydr;          % Least-squares solution
    
        % 3. Update residual
        residual = Ydr - A_selected * S_ls;
    
        % Check stopping criterion (residual norm)
        if norm(residual, 'fro') < threshold
            break;
        end
    end
    
    % Reconstruct sparse solution
    
    S_est_dr(selected_indices, :) = S_ls;
    S_est = S_est_dr;   % Final sparse solution
    Ps = sum(abs(S_est).^2, 2);   % Power spectrum

    % 找topk个峰值，而不是直接用topk
    [pks, locs] = findpeaks(Ps);
    if length(pks)<SOURCE_K
        succ = false;
        ang_est = sort(THETA_angles(locs));
        ang_est = [ang_est,NaN(1,SOURCE_K-length(ang_est))];
    else
        succ = true;
        [k_pks,pk_locs] = maxk(pks,SOURCE_K);
        loc = locs(pk_locs);
        ang_est = sort(THETA_angles(loc));
    end
    succ_vec(num_sample) = succ;
    est_DOA(num_sample,:) = ang_est;
    est_Ps(num_sample,:) = Ps;
end
save(save_root,'succ_vec','est_DOA','est_Ps');
end

