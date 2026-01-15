function [Model,Sind,Xind,Train,Test,W,B,C,P] = Low_dim_RBF(S,n,d,Cr,T,max_group_size,groups)

X = S(:,1:d);
Y = S(:,d+1);
t = 3/4;
% if nargin < 6 || isempty(max_group_size)
%     max_group_size = 50;
% end
% if nargin < 7
%     groups = [];
% end

% if ~isempty(groups)
%     groups = normalize_groups(groups, d);
% else
%     groups = split_dims_by_cr(Cr, d, max_group_size);
% end
if isempty(groups)
    groups = {1:d};
end
T = numel(groups);

Train = cell(1,T);
Test = cell(1,T);
Sind = cell(1,T);
Xind = cell(1,T);
Model = cell(1,T);
if n < 400
    ClusterNum = ceil(n*0.60);
else
    ClusterNum = 150;
end
nc0 = ClusterNum;

W = zeros(T,nc0);
B = zeros(1,T);
C = cell(1,T);
P = zeros(nc0,T);
for i = 1 : T

    ind = rand(n,T);
    I = ind(:,i) < t;

    xind = groups{i};

    [ W2,B2,Centers,Spreads ] = RBF1( X(I,xind),Y(I,1),nc0);
    W(i,:)=W2;
    B(i)=B2;
    C{i}=Centers;
    P(:,i)=Spreads;
    Sind{i} = find(I == true);
    Xind{i} = xind;
    Train{i} = [X(I,xind),Y(I,1)];
    Test{i} = [X(~I,xind),Y(~I,1)];

end

end
% 
% function groups = split_dims_by_cr(Cr, dim, max_group_size)
%     max_group_size = max(1, int32(max_group_size));
%     if dim <= 0
%         groups = {};
%         return;
%     end
%     cr = reshape(Cr, 1, []);
%     if numel(cr) ~= dim
%         order = 1:dim;
%     else
%         cr(~isfinite(cr)) = 0;
%         [~, order] = sort(cr, 'descend');
%     end
%     groups = {};
%     gi = 1;
%     for start = 1:max_group_size:dim
%         stop = min(start + max_group_size - 1, dim);
%         group = order(start:stop);
%         if ~isempty(group)
%             groups{gi} = sort(group);
%             gi = gi + 1;
%         end
%     end
% end

% function cleaned = normalize_groups(groups, dim)
%     if ~iscell(groups)
%         if ismatrix(groups)
%             groups = num2cell(groups, 2);
%         else
%             error('groups must be a cell array or numeric matrix.');
%         end
%     end
%     cleaned = {};
%     seen = false(1, dim);
%     ci = 1;
%     for i = 1:numel(groups)
%         group = groups{i};
%         if isempty(group)
%             continue;
%         end
%         uniq = unique(int32(group(:))).';
%         uniq = sort(uniq);
%         if isempty(uniq)
%             continue;
%         end
%         if uniq(1) < 1 || uniq(end) > dim
%             error('group index out of range');
%         end
%         if any(seen(uniq))
%             error('overlapping groups detected');
%         end
%         seen(uniq) = true;
%         cleaned{ci} = double(uniq);
%         ci = ci + 1;
%     end
%     if ~all(seen)
%         missing = find(~seen);
%         miss_preview = missing(1:min(10, numel(missing)));
%         error('group coverage mismatch: missing %s', mat2str(miss_preview));
%     end
% end

% function index = roulette(Cr,dim,max_group_size)
% 
% if nargin >= 3 && ~isempty(max_group_size)
%     rn = max(1, min(dim, int32(max_group_size)));
% else
%     rn = dim;
% end
% Cr = reshape(Cr,1,[]);
% Cr = Cr + max(min(Cr),0);
% Cr = cumsum(Cr);
% Cr = Cr./max(Cr);
% index   = arrayfun(@(S)find(rand<=Cr,1),1:rn);
% [~,uni] = unique(index);
% index = index(sort(uni));
% end
