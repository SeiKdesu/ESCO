
function [ obj ] =compute_objective(POP,c,problem_name)
% Usage: [ obj ] =compute_objectives(POP,c,problem_name)

% Input:
% problem_name  - Benchmark Problem
% c             -No. of Decision Variables
% POP           -Population of Decision Variables
%
% Output:
% obj           - Calculated Objective Value
%

obj=[];
n=size(POP,1);
X=POP(:,1:c);
if isnumeric(problem_name)
        func_id = problem_name;
        obj = call_cec2013_lsgo(X, func_id);
        return;
    end
pname = lower(char(problem_name));

    % --- CEC2013 LSGO route ---
    if startsWith(pname, 'cec2013') || startsWith(pname, 'lsgo2013') || startsWith(pname, 'cec13')
        tok = regexp(pname, 'f(\d+)$', 'tokens', 'once');
        if isempty(tok)
            error("CEC2013 usage: problem_name must end with f#, e.g. 'cec2013_f1'.");
        end
        func_id = str2double(tok{1});
        obj = call_cec2013_lsgo(X, func_id);
        return;
    end
switch pname
    case 'ellipsoid'
        P=[1:c];
        P=ones(size(POP,1),1)*P;
        obj=sum(P.*(POP(:,1:c).^2),2);
    case 'rosenbrock'
        P=100*(POP(:,2:c)-POP(:,1:c-1).^2).^2;
        obj=sum(P,2)+sum((POP(:,1:c-1)-1).^2,2);
    case 'rastrigin'
        obj=10*c+sum(POP(:,1:c).^2-10*cos(2*pi*POP(:,1:c)),2);
    case 'ackley'
        a=20;
        b=0.2;
        cc=2*pi;
        obj=0-a*exp(0-b*(mean(POP(:,1:c).^2,2)).^0.5)-exp(mean(cos(cc*POP(:,1:c)),2))+a+exp(1);
    case 'griewank'
        P=[1:c].^0.5;
        P=ones(size(POP,1),1)*P;
        obj=sum(POP(:,1:c).^2,2)/4000+1-prod(cos(POP(:,1:c)./P),2);
    case 'rastrigin_rot_func'
        obj = fun(POP(:,1:c),10);
    case 'hybrid_rot_func1'
        obj = fun(POP(:,1:c),16);
    case 'hybrid_rot_func2_narrow'
        obj = fun(POP(:,1:c),19);
end
end

function f = call_cec2013_lsgo(X, func_id)
    if exist('benchmark_func', 'file') ~= 2
        error("Can't find benchmark_func.m. Check addpath to lsgo_2013_benchmarks.");
    end

    % Reset initial_flag ONLY when func_id changes (avoid huge slowdown)
    persistent last_id mode
    global initial_flag;
    if isempty(last_id) || last_id ~= func_id
        initial_flag = 0;
        last_id = func_id;
        mode = "";  % re-detect orientation
    end

    % Detect expected orientation once
    if mode == ""
        try
            t = benchmark_func(X(1,:), func_id);
            if numel(t) == 1
                mode = "row";  % accepts (N x D)
            else
                t2 = benchmark_func(X(1,:).', func_id);
                if numel(t2) == 1
                    mode = "col"; % expects (D x N) or (D x 1)
                else
                    error("benchmark_func output not scalar in both orientations.");
                end
            end
        catch
            % If row call errors, try column
            t2 = benchmark_func(X(1,:).', func_id);
            if numel(t2) == 1
                mode = "col";
            else
                rethrow(lasterror);
            end
        end
    end

    % Evaluate population
    n = size(X,1);
    f = zeros(n,1);

    if mode == "row"
        % Try vectorized
        try
            out = benchmark_func(X, func_id);
            f = out(:);
            return;
        catch
            % fallback row-by-row
            for i=1:n
                tmp = benchmark_func(X(i,:), func_id);
                f(i) = tmp(1);
            end
        end
    else
        % col mode: vectorized uses (D x N)
        try
            out = benchmark_func(X.', func_id);  % (D x N)
            f = out(:);
            return;
        catch
            % fallback col-by-col
            for i=1:n
                tmp = benchmark_func(X(i,:).', func_id); % (D x 1)
                f(i) = tmp(1);
            end
        end
    end
end


