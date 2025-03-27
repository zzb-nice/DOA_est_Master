function [succ, ang_est, Ps] = R_construct(R, ULA_N, threshold, SOURCE_K, THETA_angles,noise_power)
% INPUTS:
% R: the sample covariance estimate
% ULA_N: the number of sensors in the array
% threshold: the convergence threshold
% SOURCE_K: the number of sources
% THETA_angles: the grid

% OUTPUTS:
% succ: whether the estimation was successful
% ang_est: the DoA estimates
% Ps: the power spectrum

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_steer_vec = @(x, N) exp(-1j * pi * sin(deg2rad(x)) * (0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The grid and dictionary for the compressed sensing method
NGrids = length(THETA_angles);
A_dic = zeros(ULA_N * ULA_N, NGrids);
for n = 1:NGrids
    ULA_gridn = ULA_steer_vec(THETA_angles(n), ULA_N);
    A_dic_mtx = kron(ULA_gridn, ULA_gridn');
    A_dic(:, n) = A_dic_mtx(:);
end

% Flatten input covariance matrix R
Rdr = R(:);
em = eye(ULA_N)*noise_power;
em = em(:);

% Orthogonal Matching Pursuit (OMP) implementation
eta_dr = zeros(NGrids, 1); % Initialize sparse solution
residual = Rdr - em;       % Initialize residual
selected_indices = [];     % Support set

for k = 1:SOURCE_K
    % 1. Correlation step: find the index of the most correlated column
    correlations = abs(A_dic' * residual);
    [~, idx] = max(correlations);
    selected_indices = [selected_indices, idx];

    % 2. Least-squares update on the selected support
    A_selected = A_dic(:, selected_indices); % Submatrix of selected atoms
    eta_ls = pinv(A_selected) * (Rdr - em);  % Least-squares solution

    % 3. Update residual
    residual = Rdr - em - A_selected * eta_ls;

    % Check stopping criterion (residual norm)
    if norm(residual, 'fro') < threshold
        break;
    end
end

% Reconstruct sparse solution
eta_dr(selected_indices) = eta_ls;
Ps = abs(eta_dr);          % Power spectrum

% Find peaks for DoA estimation
[pks, locs] = findpeaks(Ps);
if length(pks) < SOURCE_K
    succ = false;
    % ang_est = [sort(THETA_angles(locs)) nan(1,SOURCE_K-length(pks))]';
    ang_est = sort([THETA_angles(locs) zeros(1,SOURCE_K-length(pks))]');
else
    succ = true;
    [k_pks, pk_locs] = maxk(pks, SOURCE_K);
    loc = locs(pk_locs);
    ang_est = sort(THETA_angles(loc))';
end

end
