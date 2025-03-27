function angles = generate_random_angles(range_A, range_B, num, k, min_sep)
    % 输入:
    % range_A, range_B: 角度生成范围 [range_A, range_B]
    % num: 生成的组数
    % k: 每组的角度数量
    % min_sep: 组内角度的最小间隔
    % 输出:
    % angles: 一个 num × k 的矩阵，每行是一个生成的角度组
    
    % 检查输入是否合法
    if range_B - range_A < (k - 1) * min_sep
        error('范围不足以满足间隔要求，请调整参数');
    end
    
    angles = zeros(num, k); % 预分配空间
    
    for i = 1:num
        group = []; % 存储当前组的角度
        while length(group) < k
            % 在范围内生成随机角度
            candidate = range_A + (range_B - range_A) * rand;
            % 检查是否与当前组中已有角度的最小间隔满足条件
            if all(abs(candidate - group) >= min_sep)
                group = [group, candidate];
            end
        end
        angles(i, :) = sort(group); % 将角度排序后存储
    end
end
