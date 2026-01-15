classdef pipeline_rbf_rdg3_true_verify
    % CEC2013 LSGO: Budgeted Hierarchical Split (500->250->...)
    % - True objective only (no surrogate)
    % - Uses RDG-style INTERACT test (4 eval / repeat per split)
    % - If separable: keep splitting until budget runs out or min_size reached
    % - If NOT separable at the top split: (optional) fallback to coarse block graph

    methods (Static)
        function info = configure_pipeline(varargin)
            S = pipeline_rbf_rdg3_true_verify.state();
            if mod(nargin, 2) ~= 0
                error('configure_pipeline expects name/value pairs.');
            end
            for i = 1:2:nargin
                key = varargin{i};
                val = varargin{i+1};
                if isstring(key) || ischar(key)
                    key = lower(char(key));
                else
                    error('Option names must be strings.');
                end
                switch key
                    case {'func_id','funcid'}
                        S.func_id = double(val);
                    case 'seed'
                        S.seed = double(val);
                    case {'budget_max','budget'}
                        S.budget_max = double(val);
                    case 'repeats_per_split'
                        S.repeats_per_split = double(val);
                    case 'min_group_size'
                        S.min_group_size = double(val);
                    case 'conservative'
                        S.conservative = logical(val);
                    case 'use_random_base'
                        S.use_random_base = logical(val);
                    case 'split_largest_first'
                        S.split_largest_first = logical(val);
                    case 'do_fallback_blocks'
                        S.do_fallback_blocks = logical(val);
                    case 'block_size'
                        S.block_size = double(val);
                    case 'repeats_block_max'
                        S.repeats_block_max = double(val);
                    case 'conservative_block'
                        S.conservative_block = logical(val);
                    case {'dimension','dim'}
                        S.D = double(val);
                    case {'lower','lb'}
                        S.lb = double(val);
                    case {'upper','ub'}
                        S.ub = double(val);
                    otherwise
                        error('Unknown option: %s', key);
                end
            end

            rng(S.seed, 'twister');
            S = pipeline_rbf_rdg3_true_verify.init_benchmark(S);
            S.budget_used = 0;
            S.pipeline_result = [];
            pipeline_rbf_rdg3_true_verify.state(S);
            info = S.info;
        end

        function varargout = run_pipeline_once(return_usage)
            if nargin < 1
                return_usage = false;
            end
            S = pipeline_rbf_rdg3_true_verify.state();
            if ~isempty(S.pipeline_result)
                if return_usage
                    varargout{1} = S.pipeline_result;
                    varargout{2} = S.budget_used;
                else
                    varargout{1} = S.pipeline_result;
                end
                return;
            end

            fprintf('Function info: dimension=%d lower=%.6g upper=%.6g\n', S.D, S.lb, S.ub);
            fprintf('\n=== Stage-1: try hierarchical 500->250->... splitting ===\n');
            [final_groups, S] = pipeline_rbf_rdg3_true_verify.hierarchical_split(S, S.repeats_per_split, S.min_group_size);

            if S.do_fallback_blocks && size(final_groups,1) == 1 ...
                    && final_groups(1,1) == 1 && final_groups(1,2) == S.D
               if mod(S.D, S.block_size) ~= 0
                    fprintf(2, '[WARN] D=%d not divisible by block_size=%d; last block will absorb remainder.\n', ...
                        S.D, S.block_size);
                end
                
                [adj, ~, ~, S] = pipeline_rbf_rdg3_true_verify.build_block_graph(...
                    S, S.block_size, S.repeats_block_max, S.conservative_block);
                
                M = max(1, floor(S.D / S.block_size));  % 例: 905/90 -> 10 ブロック（最後が95）
                
                edges = sum(sum(triu(adj, 1)));
                total_pairs = M * (M - 1) / 2;
                
                fprintf('\nAdjacency summary:\n');
                fprintf('edges: %d / %d\n', edges, total_pairs);
                fprintf('budget: %d / %d\n', S.budget_used, S.budget_max);
                
                comps = pipeline_rbf_rdg3_true_verify.connected_components_from_adj(adj);
                fprintf('\nConnected components (block indices):\n');
                for k = 1:numel(comps)
                    comp = comps{k};
                    ranges = zeros(numel(comp), 2);
                    for j = 1:numel(comp)
                        b = comp(j);
                        ranges(j,1) = (b - 1) * S.block_size;
                        if b < M
                            ranges(j,2) = b * S.block_size - 1;
                        else
                            ranges(j,2) = S.D - 1;   % 最後だけ端数吸収
                        end
                    end
                    fprintf('  comp%d: blocks=%s var_ranges=%s\n', k-1, mat2str(comp), mat2str(ranges));
                end

            end

            fprintf('\n=== Final Groups ===\n');
            sizes = final_groups(:,2) - final_groups(:,1) + 1;
            for i = 1:size(final_groups,1)
                fprintf('  [%4d, %4d]  size=%d\n', final_groups(i,1)-1, final_groups(i,2)-1, sizes(i));
            end
            fprintf('\n#groups=%d  size stats: min=%d mean=%.1f max=%d\n', ...
                size(final_groups,1), min(sizes), mean(sizes), max(sizes));
            fprintf('Final budget used: %d/%d\n', S.budget_used, S.budget_max);

            out_groups = sprintf('hier_groups_func%d_budget%d_rep%d_min%d.npy', ...
                S.func_id, S.budget_max, S.repeats_per_split, S.min_group_size);
            final_groups_zero = int32(final_groups - 1);
            pipeline_rbf_rdg3_true_verify.save_npy_int32(out_groups, final_groups_zero);
            fprintf('Saved: %s\n', out_groups);

            S.pipeline_result = final_groups_zero;
            pipeline_rbf_rdg3_true_verify.state(S);

            if return_usage
                varargout{1} = final_groups_zero;
                varargout{2} = S.budget_used;
            else
                varargout{1} = final_groups_zero;
            end
        end
    end

    methods (Static, Access = private)
        function S = default_state()
            S.seed = 42;
            S.func_id = 4;
            S.budget_max = 200;
            S.repeats_per_split = 3;
            S.conservative = true;
            S.use_random_base = true;
            S.min_group_size = 1;
            S.split_largest_first = true;
            S.do_fallback_blocks = true;
            S.block_size = 100;
            S.repeats_block_max = 3;
            S.conservative_block = true;
            S.pipeline_result = [];
            S.budget_used = 0;
            S.D = 1000;
            S.lb = -100;
            S.ub = 100;
            S = pipeline_rbf_rdg3_true_verify.init_benchmark(S);
        end

        function S = state(new_state)
            persistent STATE
            if isempty(STATE)
                STATE = pipeline_rbf_rdg3_true_verify.default_state();
            end
            if nargin > 0
                STATE = new_state;
            end
            S = STATE;
        end

        function S = init_benchmark(S)
            S.midval = (S.lb + S.ub) / 2.0;
            S.gamma_k = pipeline_rbf_rdg3_true_verify.gamma_k_for_dim(S.D);
            S.info = struct('dimension', double(S.D), 'lower', double(S.lb), 'upper', double(S.ub));
        end

        function gamma_k = gamma_k_for_dim(dim)
            mu = eps(1.0);
            k = sqrt(double(dim)) + 2.0;
            if k * mu >= 1.0
                gamma_k = inf;
            else
                gamma_k = (k * mu) / (1.0 - k * mu);
            end
        end

        function x0 = make_base_point(S)
            if S.use_random_base
                x0 = S.lb + (S.ub - S.lb) * rand(1, S.D);
            else
                x0 = ones(1, S.D) * S.lb;
            end
        end

        function x = set_range(~, x0, l, r, value)
            x = x0;
            x(l:r) = value;
        end

        function x = set_two_ranges(~, x0, l1, r1, v1, l2, r2, v2)
            x = x0;
            x(l1:r1) = v1;
            x(l2:r2) = v2;
        end

        function S = ensure_budget(S, k, msg)
            if S.budget_used + k > S.budget_max
                error(msg);
            end
        end

        function [val, S] = eval_f(S, x)
            if size(x,1) ~= 1
                x = x(:).';
            end
            val = compute_objective(x, S.D, S.func_id);
            val = double(val(1));
            S.budget_used = S.budget_used + 1;
        end

        function [inter, S] = interact_test_one_repeat_ranges(S, l1, r1, l2, r2, x0)
            S = pipeline_rbf_rdg3_true_verify.ensure_budget(S, 1, 'Budget exhausted before y0.');
            [y0, S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x0);

            S = pipeline_rbf_rdg3_true_verify.ensure_budget(S, 1, 'Budget exhausted before y_ul.');
            x_ul = pipeline_rbf_rdg3_true_verify.set_range([], x0, l1, r1, S.ub);
            [y_ul, S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x_ul);

            S = pipeline_rbf_rdg3_true_verify.ensure_budget(S, 1, 'Budget exhausted before y_lm.');
            x_lm = pipeline_rbf_rdg3_true_verify.set_range([], x0, l2, r2, S.midval);
            [y_lm, S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x_lm);

            S = pipeline_rbf_rdg3_true_verify.ensure_budget(S, 1, 'Budget exhausted before y_um.');
            x_um = pipeline_rbf_rdg3_true_verify.set_two_ranges([], x0, l1, r1, S.ub, l2, r2, S.midval);
            [y_um, S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x_um);

            delta1 = y0 - y_ul;
            delta2 = y_lm - y_um;
            diff_abs = abs(delta1 - delta2);
            eps_err = S.gamma_k * (abs(y0) + abs(y_ul) + abs(y_lm) + abs(y_um));

            inter = diff_abs > eps_err;
        end

        function [res, S] = is_separable_by_interact(S, l, r, split, repeats, conservative)
            l1 = l;
            r1 = split;
            l2 = split + 1;
            r2 = r;
            need = 4 * repeats;
            if S.budget_used + need > S.budget_max
                res = [];
                return;
            end

            for rep = 1:repeats
                x0 = pipeline_rbf_rdg3_true_verify.make_base_point(S);
                [inter, S] = pipeline_rbf_rdg3_true_verify.interact_test_one_repeat_ranges(S, l1, r1, l2, r2, x0);
                if conservative && inter
                    res = false;
                    return;
                end
            end
            res = true;
        end

        function [finalized, S] = hierarchical_split(S, repeats, min_group_size)
            groups = [1, S.D];
            finalized = zeros(0, 2);

            while ~isempty(groups)
                if S.split_largest_first
                    sizes = groups(:,2) - groups(:,1) + 1;
                    [~, order] = sort(sizes, 'descend');
                    groups = groups(order, :);
                end
                l = groups(1,1);
                r = groups(1,2);
                groups(1,:) = [];
                size_lr = r - l + 1;

                if size_lr <= min_group_size
                    finalized = [finalized; l r];
                    continue;
                end

                split = floor((l + r) / 2);
                if split >= r
                    finalized = [finalized; l r];
                    continue;
                end

                [res, S] = pipeline_rbf_rdg3_true_verify.is_separable_by_interact(S, l, r, split, repeats, S.conservative);
                if isempty(res)
                    finalized = [finalized; l r];
                elseif res
                    groups = [groups; l split; split+1 r];
                    fprintf('[SPLIT] (%d-%d) -> (%d-%d) | (%d-%d)  separable? True  budget %d/%d\n', ...
                        l-1, r-1, l-1, split-1, split, r-1, S.budget_used, S.budget_max);
                else
                    finalized = [finalized; l r];
                    fprintf('[KEEP ] (%d-%d)  separable? False  budget %d/%d\n', ...
                        l-1, r-1, S.budget_used, S.budget_max);
                end
            end

            finalized = sortrows(finalized, 1);
        end

        function [adj, counts, repeats, S] = build_block_graph(S, block_size, repeats_max, conservative_edge)
         
            M = max(1, floor(S.D / block_size));
            pairs = M * (M - 1) / 2;
            cost_per_repeat = 1 + 2 * M + pairs;

            remaining = S.budget_max - S.budget_used;
            repeats = min(repeats_max, floor(remaining / cost_per_repeat));
            repeats = max(0, repeats);

            fprintf('\n=== Fallback: block dependency test ===\n');
            fprintf('Blocks: M=%d (BLOCK_SIZE=%d), per repeat cost=%d, repeats=%d\n', ...
                M, block_size, cost_per_repeat, repeats);
            fprintf('Expected total evals <= %d (budget %d)\n', ...
                S.budget_used + repeats * cost_per_repeat, S.budget_max);

            counts = zeros(M, M);
            for rep = 1:repeats
                if S.budget_used + cost_per_repeat > S.budget_max
                    break;
                end

                x0 = pipeline_rbf_rdg3_true_verify.make_base_point(S);
                [y0, S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x0);

                y_ul = zeros(M, 1);
                y_lm = zeros(M, 1);

                for i = 1:M
                    l = (i - 1) * block_size + 1;
                    if i < M
                        r = i * block_size;
                    else
                        r = S.D;  % ★最後だけ吸収
                    end
                    x_ul = pipeline_rbf_rdg3_true_verify.set_range([], x0, l, r, S.ub);
                    [y_ul(i), S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x_ul);
                end

                for j = 1:M
                    l = (j - 1) * block_size + 1;
                    if j < M
                        r = j * block_size;
                    else
                        r = S.D;  % ★最後だけ吸収
                    end
                    x_lm = pipeline_rbf_rdg3_true_verify.set_range([], x0, l, r, S.midval);
                    [y_lm(j), S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x_lm);
                end

                for i = 1:M
                    li = (i - 1) * block_size + 1;
                    if i < M
                        ri = i * block_size;
                    else
                        ri = S.D;          % ★最後ブロックは端数吸収
                    end
                
                    for j = (i + 1):M
                        lj = (j - 1) * block_size + 1;
                        if j < M
                            rj = j * block_size;
                        else
                            rj = S.D;      % ★最後ブロックは端数吸収
                        end
                
                        x_um = pipeline_rbf_rdg3_true_verify.set_two_ranges([], x0, li, ri, S.ub, lj, rj, S.midval);
                        [y_um, S] = pipeline_rbf_rdg3_true_verify.eval_f(S, x_um);
                        ...
                    end
                end


                fprintf('Repeat %d/%d done. budget used %d/%d\n', rep, repeats, S.budget_used, S.budget_max);
            end

            adj = zeros(M, M);
            if repeats > 0
                if conservative_edge
                    adj(counts > 0) = 1;
                else
                    adj(counts >= (floor(repeats/2) + 1)) = 1;
                end
            end
            adj(1:M+1:end) = 0;
        end

        function comps = connected_components_from_adj(adj)
            M = size(adj, 1);
            seen = false(M, 1);
            comps = {};
            ci = 1;
            for s = 1:M
                if seen(s)
                    continue;
                end
                stack = s;
                seen(s) = true;
                comp = [];
                while ~isempty(stack)
                    u = stack(end);
                    stack(end) = [];
                    comp(end+1) = u;
                    nbrs = find(adj(u, :) ~= 0);
                    for v = nbrs
                        if ~seen(v)
                            seen(v) = true;
                            stack(end+1) = v;
                        end
                    end
                end
                comps{ci} = sort(comp);
                ci = ci + 1;
            end
        end

        function save_npy_int32(filename, arr)
            if ~isa(arr, 'int32')
                arr = int32(arr);
            end
            shape = size(arr);
            if numel(shape) == 2
                shape_str = sprintf('(%d, %d)', shape(1), shape(2));
            else
                shape_str = sprintf('(%d,)', numel(arr));
                arr = arr(:);
                shape = size(arr);
            end
            header = sprintf('{''descr'': ''<i4'', ''fortran_order'': False, ''shape'': %s, }', shape_str);
            header = [header, char(10)];
            pad_len = mod(16 - mod(10 + numel(header), 16), 16);
            header = [header, repmat(' ', 1, pad_len)];
            header_len = numel(header);

            fid = fopen(filename, 'w', 'ieee-le');
            if fid < 0
                error('Failed to open file for writing: %s', filename);
            end
            cleanup = onCleanup(@() fclose(fid));

            fwrite(fid, uint8([147 78 85 77 80 89]), 'uint8');
            fwrite(fid, uint8([1 0]), 'uint8');
            fwrite(fid, uint16(header_len), 'uint16');
            fwrite(fid, header, 'char');

            if numel(shape) == 2
                data = arr.';
            else
                data = arr;
            end
            fwrite(fid, data, 'int32');
        end
    end
end
