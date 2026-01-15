function [Best,Data0] = CEA(Data,BU,BD,problem_name, progressFileID, decompFileID)

if nargin < 5 || isempty(progressFileID)
    progressFileID = -1;
end
if nargin < 6 || isempty(decompFileID)
    decompFileID = -1;
end

dim = size(Data,2)-1;
num0 = size(Data,1);
pc = 1.0;
pm = 1/dim;
bu = BU;
bd = BD;

max_group_size_default = 500;
max_group_size_early = 100;
max_group_size_late = 500;


max_group_size = max_group_size_default;
rdg3_epsilon_n = max_group_size;
rdg3_epsilon_s = max_group_size;

l = sqrt(0.001^2 * dim);
N = 20;
gmax = 20;
FEs = 1000;

[pipeline_groups, pipeline_evals] = pipeline_groups_for_problem(problem_name, dim, bd(1), bu(1));


fes = 400 + pipeline_evals;
switch_samples = 200+fes;
Data0 = Data;

base_full_group = [];
active_max_group_size = max_group_size;

precomputed_groups = {};
Cr_fixed = [];

if ~isempty(pipeline_groups)
    if numel(pipeline_groups) == 1 && numel(pipeline_groups{1}) == dim
        base_full_group = pipeline_groups{1};
        active_max_group_size = select_max_group_size(size(Data0,1), max_group_size_early, max_group_size_late, switch_samples);
        if dim > active_max_group_size
            pipeline_groups = split_group_by_size(base_full_group, active_max_group_size);
        else
            pipeline_groups = {base_full_group};
        end
    end
    precomputed_groups = pipeline_groups;
% // else
% //     X_data = Data0(:,1:dim);
% //     y_data = Data0(:,dim+1);
% //     Cr_fixed = compute_cr_from_DIG(X_data, y_data, 5, 0.02);
% //     precomputed_groups = split_dims_by_cr(Cr_fixed, dim, max_group_size);
end

if ~isempty(precomputed_groups)
    T = numel(precomputed_groups);
% // else
% //     T = ceil(dim / max_group_size);
end
% // if T < 1
% //     T = 1;
% // end
K = ceil(T * 0.3);

while fes <= FEs

    if ~isempty(base_full_group)
        current_samples = size(Data0, 1);
        new_max_group_size = select_max_group_size(current_samples, max_group_size_early, max_group_size_late, switch_samples);
        if new_max_group_size ~= active_max_group_size
            active_max_group_size = new_max_group_size;
            if dim > active_max_group_size
                precomputed_groups = split_group_by_size(base_full_group, active_max_group_size);
            else
                precomputed_groups = {base_full_group};
            end
            T = numel(precomputed_groups);
            K = ceil(T * 0.3);
            % K = T;

        end
    end

    Data = Data_Process(Data0, num0);
    num = size(Data,1);
    Best = min(Data(:,dim+1));

    if progressFileID > 0
        fprintf(progressFileID, '%d,%.15g\n', fes, Best);
    end

    if decompFileID > 0
        fprintf(decompFileID, '\n[fes=%d] Best=%.15g\n', fes, Best);
        if ~isempty(pipeline_groups)
            fprintf(decompFileID, 'Pipeline groups (count=%d):\n', numel(precomputed_groups));
            for g = 1:numel(precomputed_groups)
                grp = precomputed_groups{g};
                fprintf(decompFileID, '  g%d: [%d..%d] size=%d\n', g, grp(1), grp(end), numel(grp));
            end
        else
            [~, order] = sort(Cr_fixed, 'descend');
            topk = min(20, numel(order));
            fprintf(decompFileID, 'Top-%d vars by |DIG| (index:score):\n', topk);
            for t = 1:topk
                fprintf(decompFileID, '  %d: %.6g\n', order(t), Cr_fixed(order(t)));
            end
        end
    end

    group_size_for_rbf = max_group_size;
    if ~isempty(base_full_group)
        group_size_for_rbf = active_max_group_size;
    end
    [Model,Sind,Xind,Train,Test,W,B,C,P] = Low_dim_RBF(Data,num,dim,Cr_fixed,T,group_size_for_rbf,precomputed_groups);
    [Ens,Mind,Sim] = Selective_Ensemble(Model,Data,Train,Test,K,W,B,C,P);

    [X,Y] = Sur_Coevolution(Data,Ens,Train,Xind,Mind,Sim,bu,bd,N,gmax,W,B,C,P);
    new = Infill_solution_Selection(Data,[X,Y],l);
    fprintf('Progress: fes = %d best = %.15g\n', fes, Best);
    if ~isempty(new)
        newY = compute_objective(new(:,1:end-1), dim, problem_name);
        Data0 = [Data0; [new(:,1:end-1), newY]];
        fes = fes + length(newY);
    end

end
Best = min(Data0(:,dim+1));

end

function Data = Data_Process(Data,num0)
num = size(Data,1);
if num > num0+400
    [~,ind] = sort(Data(:,end));
    c = num0+400+1;
    Data(ind(c:end),:) = [];
end
end

function func_id = parse_cec2013_func_id(problem_name)
    func_id = [];
    if isempty(problem_name)
        return;
    end
    if isnumeric(problem_name)
        func_id = double(problem_name);
        return;
    end
    lower_name = lower(char(problem_name));
    if ~startsWith(lower_name, 'cec2013_')
        return;
    end
    suffix = lower_name(numel('cec2013_')+1:end);
    if startsWith(suffix, 'f')
        suffix = suffix(2:end);
    end
    if all(isstrprop(suffix, 'digit'))
        func_id = str2double(suffix);
    end
end

function groups = expand_range_groups(range_groups, dim)
    if isempty(range_groups)
        groups = {};
        return;
    end
    groups = cell(size(range_groups,1), 1);
    seen = false(1, dim);
    for i = 1:size(range_groups,1)
        l = double(range_groups(i,1));
        r = double(range_groups(i,2));
        if l < 0 || r >= dim || l > r
            error('invalid group range');
        end
        idx = (l:r) + 1;
        if any(seen(idx))
            error('overlapping group range');
        end
        seen(idx) = true;
        groups{i} = idx;
    end
    if ~all(seen)
        missing = find(~seen);
        miss_preview = missing(1:min(10, numel(missing)));
        error('group coverage mismatch: missing %s', mat2str(miss_preview));
    end
end

function [groups, eval_used] = pipeline_groups_for_problem(problem_name, dim, lb, ub)
    groups = {};
    eval_used = 0;
    func_id = parse_cec2013_func_id(problem_name);
    if isempty(func_id)
        return;
    end
    rng_state = rng;
    cleanup = onCleanup(@() rng(rng_state));
    try
        info = pipeline_rbf_rdg3_true_verify.configure_pipeline('func_id', func_id, 'dim', dim, 'lb', lb, 'ub', ub);

        if isfield(info, 'dimension') && info.dimension ~= dim
            error('CEC2013 dimension mismatch');
        end
        [range_groups, eval_used] = pipeline_rbf_rdg3_true_verify.run_pipeline_once(true);
        groups = expand_range_groups(range_groups, dim);
    catch ME
        fprintf(2, '[WARN] Pipeline decomposition failed: %s\n', ME.message);
        groups = {};
        eval_used = 0;
    end
end

function max_group_size = select_max_group_size(sample_count, early_size, late_size, switch_samples)
    if sample_count <= switch_samples
        max_group_size = early_size;
    else
        max_group_size = late_size;
    end
end

function groups = split_group_by_size(base, max_group_size)
    groups = {};
    gi = 1;
    n = numel(base);
    for start = 1:max_group_size:n
        stop = min(start + max_group_size - 1, n);
        groups{gi} = base(start:stop);
        gi = gi + 1;
    end
end

function groups = split_dims_by_cr(Cr, dim, max_group_size)
    max_group_size = max(1, int32(max_group_size));
    if dim <= 0
        groups = {};
        return;
    end
    cr = reshape(Cr, 1, []);
    if numel(cr) ~= dim
        order = 1:dim;
    else
        cr(~isfinite(cr)) = 0;
        [~, order] = sort(cr, 'descend');
    end
    groups = {};
    gi = 1;
    for start = 1:max_group_size:dim
        stop = min(start + max_group_size - 1, dim);
        group = order(start:stop);
        if ~isempty(group)
            groups{gi} = sort(group);
            gi = gi + 1;
        end
    end
end

% function Cr = compute_cr_from_DIG(X, y, k, rho)
%     if nargin < 3 || isempty(k)
%         k = 5;
%     end
%     if nargin < 4 || isempty(rho)
%         rho = 0.02;
%     end
%     [~, ~, I] = build_dependency_graph_knn(X, y, k, rho, 1e-12);
%     D = size(X,2);
%     if all(I(:) == 0)
%         Cr = ones(1, D) / D;
%         return;
%     end
%     abs_I = abs(I);
%     imp = sum(abs_I, 2);
%     s = sum(imp);
%     if s == 0
%         Cr = ones(1, D) / D;
%     else
%         Cr = (imp ./ s).';
%     end
% end

% function [edges, M, I] = build_dependency_graph_knn(X, y, k, rho, eps_val)
%     if nargin < 5 || isempty(eps_val)
%         eps_val = 1e-12;
%     end
%     [N, D] = size(X);
%     if N <= 1
%         edges = zeros(0, 2);
%         M = zeros(1, D);
%         I = zeros(D, D);
%         return;
%     end
%     k = min(k, N - 1);
% 
%     if exist('pdist2', 'file') == 2
%         dists = pdist2(X, X);
%     else
%         dists = zeros(N, N);
%         for i = 1:N
%             diff = X - X(i,:);
%             dists(i,:) = sqrt(sum(diff.^2, 2));
%         end
%     end
% 
%     [~, idx] = sort(dists, 2, 'ascend');
% 
%     pair_mask = false(N, N);
%     for a = 1:N
%         for t = 2:(k+1)
%             b = idx(a, t);
%             if a < b
%                 pair_mask(a, b) = true;
%             elseif b < a
%                 pair_mask(b, a) = true;
%             end
%         end
%     end
% 
%     [pa, pb] = find(triu(pair_mask, 1));
% 
%     M = zeros(1, D);
%     I = zeros(D, D);
%     for p = 1:numel(pa)
%         a = pa(p);
%         b = pb(p);
%         dx = X(b,:) - X(a,:);
%         df = y(b) - y(a);
%         norm_val = sum(abs(dx)) + eps_val;
%         s = abs(dx) ./ norm_val;
%         w = abs(df);
% 
%         M = M + w * s;
% 
%         active = find(s > (1.0 / D));
%         for i_idx = 1:numel(active)
%             i = active(i_idx);
%             for j_idx = (i_idx + 1):numel(active)
%                 j = active(j_idx);
%                 val = w * s(i) * s(j);
%                 I(i, j) = I(i, j) + val;
%                 I(j, i) = I(j, i) + val;
%             end
%         end
%     end
% 
%     scores = I(triu(true(D), 1));
%     if all(scores == 0)
%         edges = zeros(0, 2);
%         return;
%     end
%     threshold = quantile_linear(scores, 1 - rho);
%     [ei, ej] = find(triu(I, 1) >= threshold);
%     edges = [ei, ej];
% end

% function qv = quantile_linear(data, q)
%     data = sort(data(:));
%     n = numel(data);
%     if n == 0
%         qv = NaN;
%         return;
%     end
%     if n == 1
%         qv = data(1);
%         return;
%     end
%     idx = (n - 1) * q;
%     lo = floor(idx);
%     hi = ceil(idx);
%     if lo == hi
%         qv = data(lo + 1);
%     else
%         w = idx - lo;
%         qv = data(lo + 1) * (1 - w) + data(hi + 1) * w;
%     end
% end
