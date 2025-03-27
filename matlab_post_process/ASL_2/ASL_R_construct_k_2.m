function [succ, ang_est, Ps] = ASL_R_construct_k_2(R, ULA_N, threshold, THETA_angles, sep, noise_power)
% INPUTS:
% R: the sample covariance matrix
% ULA_N: the number of sensors in the array
% threshold: reconstruction error tolerance
% THETA_angles: the grid
% sep: angular separation
% noise_power: noise variance
%
% OUTPUTS:
% succ: success flag (true if estimation succeeds)
% ang_est: estimated angles of arrival
% Ps: power spectrum over the grid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_steer_vec = @(x, N) exp(1j * pi * sin(deg2rad(x)) * (0:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The grid and dictionary for the compressed sensing method
NGrids = length(THETA_angles);
THETA_angles_2 = THETA_angles + sep;

A_dic = zeros(ULA_N * ULA_N, NGrids);
A_dic_2 = zeros(ULA_N * ULA_N, NGrids);
for n = 1:NGrids
    ULA_gridn = ULA_steer_vec(THETA_angles(n), ULA_N);
    A_dic_mtx = kron(ULA_gridn, ULA_gridn');
    A_dic(:, n) = A_dic_mtx(:);

    ULA_gridn = ULA_steer_vec(THETA_angles_2(n), ULA_N);
    A_dic_mtx = kron(ULA_gridn, ULA_gridn');
    A_dic_2(:, n) = A_dic_mtx(:);
end

% Combine dictionaries
A_combined = A_dic + A_dic_2;
Rdr = R(:);
em = ones(ULA_N);
em = noise_power * em(:);

% Initialization for OMP
residual = Rdr - em; % Initial residual
selected_atoms = []; % Indices of selected atoms
coef_est = zeros(NGrids, 1); % Coefficient estimates

% Iteratively select atoms
% for iter = 1:NGrids
for iter = 1:1
    % Compute correlation
    correlation = abs(A_combined' * residual);
    
    % Select the atom with maximum correlation
    [~, new_atom] = max(correlation);
    selected_atoms = [selected_atoms, new_atom];
    
    % Extract the sub-dictionary corresponding to selected atoms
    A_sub = A_combined(:, selected_atoms);
    
    % Solve least squares to update coefficients
    coef_temp = pinv(A_sub) * (Rdr - em);
    
    % Update residual
    residual = Rdr - em - A_sub * coef_temp;
    
    % Check stopping condition
    if norm(residual, 'fro') <= threshold
        break;
    end
end

% Store final coefficients
coef_est(selected_atoms) = coef_temp;

% Compute power spectrum
Ps = abs(coef_est).^2;

% Find the top peak
SOURCE_K = 1;
[pks, locs] = findpeaks(Ps);
if length(pks) < SOURCE_K
    succ = false;
    ang_est = sort(THETA_angles(locs))';
else
    succ = true;
    [~, pk_locs] = maxk(pks, SOURCE_K);
    loc = locs(pk_locs);
    ang_est = sort(THETA_angles(loc))';
end

end
