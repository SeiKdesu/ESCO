function run_esco(varargin)
% run_esco -- CLI対応：指定した関数を順番に実行して結果を保存
%
% 使い方例（Mac/Linux）:
%   matlab -batch "run_esco('--funcs','2','--dim','1000','--bd','-5','--bu','5')"
%   matlab -batch "run_esco('--func_range','1:3','--dim','1000','--auto_bounds','1')"
%   matlab -batch "run_esco('--funcs','2,3,5','--dim','1000','--bd','-5','--bu','5','--initial_points','400')"
%
% 引数:
%   --funcs        例: "2" / "2,3,5" / "cec2013_f2,cec2013_f3"
%   --func_range   例: "1:15"
%   --dim          例: "1000"
%   --bd           例: "-5"
%   --bu           例: "5"
%   --bounds       例: "-5,5"  (bd,bu をまとめて指定)
%   --initial_points 例: "400"
%   --auto_bounds  1なら関数ごとの推奨境界を自動設定（bd/bu未指定時）
%   --result_dir   例: "result"
%   --bench_dir    例: "lsgo_2013_benchmarks"（run_esco.m の場所からの相対でもOK）
%   --continue_on_error  1ならエラーでも次へ進む（夜回し用）

    opts = parse_args(varargin{:});

    % --- addpath: run_esco.m の位置を基準にする（pwd依存を避ける）
    here = fileparts(mfilename('fullpath'));
    bench_path = opts.bench_dir;
    if ~isfolder(bench_path)
        cand = fullfile(here, bench_path);
        if isfolder(cand)
            bench_path = cand;
        end
    end
    addpath(genpath(bench_path));

    fprintf('[INFO] benchmark_func: %s\n', which('benchmark_func'));
    if isempty(which('benchmark_func'))
        error('benchmark_func が見つかりません。--bench_dir を確認してください。');
    end

    % --- 実行する関数IDリスト
    fids = parse_func_list(opts);
    fprintf('[INFO] funcs = [%s]\n', strjoin(string(fids), ', '));

    if ~exist(opts.result_dir, 'dir')
        mkdir(opts.result_dir);
    end

    % 夜回し用：全体ログ
    log_path = fullfile(opts.result_dir, sprintf('runlog_%s.txt', datestr(now,'yyyymmdd_HHMMSS')));
    diary(log_path);
    fprintf('[INFO] log -> %s\n', log_path);

    % --- 順番に実行
    for k = 1:numel(fids)
        fid = fids(k);
        problem_name = sprintf('cec2013_f%d', fid);

        % bounds決定
        [bd_scalar, bu_scalar] = decide_bounds(opts, fid);
        dim = opts.dim;

        bu = ones(1, dim) * bu_scalar;
        bd = ones(1, dim) * bd_scalar;

        fprintf('\n=============================\n');
        fprintf('[RUN] %s, dim=%d, bd=%.6g, bu=%.6g\n', problem_name, dim, bd_scalar, bu_scalar);

        try
            run_one(problem_name, dim, bd, bu, opts);
            fprintf('[DONE] %s\n', problem_name);
        catch ME
            fprintf(2, '[ERROR] %s failed: %s\n', problem_name, ME.message);
            fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));

            if ~opts.continue_on_error
                diary off;
                rethrow(ME);
            end
        end
    end

    diary off;
end

% =============================
% 1回分の実行本体
% =============================
function run_one(problem_name, dim, bd, bu, opts)

    % --- 2. 初期データセット
    initial_points = opts.initial_points;
    initial_pop = initialize_pop(initial_points, dim, bu, bd);

    initial_obj = compute_objective(initial_pop, dim, problem_name);

    % NaNが出たら即止め（次へ進むかは上位で制御）
    bad = ~isfinite(initial_obj);
    if any(bad)
        fprintf(2,'[BAD] initial_obj: bad=%d/%d\n', nnz(bad), numel(bad));
        error('initial_obj contains NaN/Inf. bounds/benchmark_func を確認してください。');
    end

    Data = [initial_pop, initial_obj];

    % --- 3. 結果ファイル
    result_dir = opts.result_dir;
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    result_filename = fullfile(result_dir, sprintf('%s_result_%s.txt', problem_name, timestamp));
    progress_filename = fullfile(result_dir, sprintf('%s_progress_%s.txt', problem_name, timestamp));
    decomposition_filename = fullfile(result_dir, sprintf('%s_decomposition_%s.txt', problem_name, timestamp));

    resultFileID = fopen(result_filename, 'w');
    progressFileID = fopen(progress_filename, 'w');
    decompFileID = fopen(decomposition_filename, 'w');

    cleanup = onCleanup(@() close_all(resultFileID, progressFileID, decompFileID));

    fprintf(progressFileID, 'Evaluations,Best_Value\n');
    fprintf(decompFileID, '==== Variable Decomposition Results ====\n');

    fprintf('ESCOアルゴリズムを開始します...\n');
    fprintf('問題: %s, 次元数: %d\n', problem_name, dim);

    % --- CEA呼び出し
    [Best_Solution, Final_Data] = CEA(Data, bu, bd, problem_name, progressFileID, decompFileID);

    % --- 結果表示
    fprintf('\n最適化が完了しました。\n');
    fprintf('見つかった最適値: %f\n', Best_Solution);

    % 最適解
    [~, min_index] = min(Final_Data(:, end));
    Best_Variables = Final_Data(min_index, 1:dim);

    % --- 保存
    fprintf(resultFileID, '==== ESCO Final Result ====\n');
    fprintf(resultFileID, 'Problem: %s\n', problem_name);
    fprintf(resultFileID, 'Dimension: %d\n', dim);
    fprintf(resultFileID, 'Bounds: [%.6g, %.6g]\n', bd(1), bu(1));
    fprintf(resultFileID, 'Best Objective Value Found: %.10f\n', Best_Solution);
    fprintf(resultFileID, 'Best Variables:\n');
    for i = 1:length(Best_Variables)
        fprintf(resultFileID, '%.10f\n', Best_Variables(i));
    end

    fprintf('結果を %s フォルダに3つのファイルとして保存しました。\n', result_dir);
    fprintf('  - %s\n  - %s\n  - %s\n', result_filename, progress_filename, decomposition_filename);
end

function close_all(a,b,c)
    if a>0, fclose(a); end
    if b>0, fclose(b); end
    if c>0, fclose(c); end
end

% =============================
% 引数パース
% =============================
function opts = parse_args(varargin)
    % defaults
    opts.dim = 1000;
    opts.initial_points = 400;
    opts.funcs = "";
    opts.func_range = "";
    opts.bd = NaN;
    opts.bu = NaN;
    opts.bounds = "";
    opts.auto_bounds = 0;
    opts.result_dir = "result";
    opts.bench_dir = "lsgo_2013_benchmarks";
    opts.continue_on_error = 1;

    % 形式: '--key','value', ...
    i = 1;
    while i <= numel(varargin)
        key = string(varargin{i});
        if startsWith(key, "--")
            key = extractAfter(key, 2);
        end

        if i == numel(varargin)
            error('arg "%s" has no value', key);
        end
        val = string(varargin{i+1});

        switch lower(key)
            case "dim"
                opts.dim = str2double(val);
            case "initial_points"
                opts.initial_points = str2double(val);
            case "funcs"
                opts.funcs = val;
            case "func_range"
                opts.func_range = val;
            case "bd"
                opts.bd = str2double(val);
            case "bu"
                opts.bu = str2double(val);
            case "bounds"
                opts.bounds = val; % "-5,5" みたいに渡す
            case "auto_bounds"
                opts.auto_bounds = str2double(val);
            case "result_dir"
                opts.result_dir = val;
            case "bench_dir"
                opts.bench_dir = val;
            case "continue_on_error"
                opts.continue_on_error = str2double(val);
            otherwise
                error('unknown option: %s', key);
        end

        i = i + 2;
    end

    if isnan(opts.dim) || opts.dim <= 0
        error('--dim must be positive');
    end
    if isnan(opts.initial_points) || opts.initial_points <= 0
        error('--initial_points must be positive');
    end
end

function fids = parse_func_list(opts)
    if strlength(opts.funcs) > 0
        % "2,3,5" or "cec2013_f2,cec2013_f3"
        parts = split(opts.funcs, ",");
        fids = [];
        for p = parts'
            s = strtrim(p);
            if startsWith(s, "cec2013_f")
                s = extractAfter(s, "cec2013_f");
            end
            fid = str2double(s);
            if ~isfinite(fid)
                error('invalid --funcs entry: %s', p);
            end
            fids(end+1) = fid; %#ok<AGROW>
        end
        fids = unique(fids, 'stable');
        return;
    end

    if strlength(opts.func_range) > 0
        % "1:15"
        toks = split(opts.func_range, ":");
        if numel(toks) ~= 2
            error('invalid --func_range (use "a:b")');
        end
        a = str2double(strtrim(toks(1)));
        b = str2double(strtrim(toks(2)));
        if ~isfinite(a) || ~isfinite(b)
            error('invalid --func_range numbers');
        end
        if a <= b
            fids = a:b;
        else
            fids = a:-1:b;
        end
        return;
    end

    % どっちも無いなら f2 を既定
    fids = 2;
end

function [bd_scalar, bu_scalar] = decide_bounds(opts, fid)
    % 明示指定があればそれを使う
    if strlength(opts.bounds) > 0
        % "-5,5"
        vv = split(opts.bounds, ",");
        if numel(vv) ~= 2
            error('--bounds must be like "-5,5"');
        end
        bd_scalar = str2double(strtrim(vv(1)));
        bu_scalar = str2double(strtrim(vv(2)));
        return;
    end
    if isfinite(opts.bd) && isfinite(opts.bu)
        bd_scalar = opts.bd;
        bu_scalar = opts.bu;
        return;
    end

    % 自動境界（bd/bu 未指定時のみ）
    if opts.auto_bounds == 1
        [bd_scalar, bu_scalar] = default_bounds_for_cec2013(fid);
        return;
    end

    % 何も指定されてない場合は無難に [-100,100]
    bd_scalar = -100;
    bu_scalar =  100;
end

function [bd, bu] = default_bounds_for_cec2013(fid)
    % よく使うものだけ実用的にマップ（必要なら増やしてOK）
    % Rastrigin系: [-5,5]
    if any(fid == [2,5,9])
        bd = -5; bu = 5; return;
    end
    % Ackley系: [-32,32]
    if any(fid == [3,6,10])
        bd = -32; bu = 32; return;
    end
    % その他はまず [-100,100]
    bd = -100; bu = 100;
end
