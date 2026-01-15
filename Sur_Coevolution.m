function [X,Y] = Sur_Coevolution(Data,Ens,Train,Xind,Mind,Sim,bu,bd,N,gmax,W,B,C,P)

dim = length(bu); %Œ¨ 
g = 0;
[~,ind] = sort(Data(:,end));
S = Data(ind(1:ceil(N/2)),:);
pc=1.0;  %Crossover Probability Ωª≤Ê∏≈¬ 
pm=1/dim;  %Mutation Probability ±‰“¸¸≈¬ 

rng('shuffle');
k = randperm(length(Ens),1);
model = Ens{k};  xind = Xind{Mind(k)};
ldim = length(xind);  %∏®÷˙◊”À—À˜µƒŒ¨ 
train = Train{Mind(k)};
X = S(:,1:dim);  Y = S(:,dim+1);
% aX = S(:,xind);  aY = S(:,dim+1);
X0 = initialize_pop(N/2,dim,bu,bd); %≥ı ºªØ÷˜À—À˜≥ı º÷÷»∫X
Y0 = Ens_predictor(X0,Ens,Train,Xind,Mind,Sim,W,B,C,P);
X = [X;X0];  Y = [Y;Y0];
aX = initialize_pop(N,ldim,bu(:,xind),bd(:,xind)); %≥ı ºªØ∏®÷˙À—À˜µƒ≥ı º÷÷»∫aX
% aY = rbfpredict(model,train(:,1:end-1),aX);
i = Mind(k);
aY = RBF_predictor(W(i,:),B(i),C{i},P(:,i),aX);
while g <= gmax
    
    X1 = SBX(X,bu,bd,pc,N/2);
    X2 = mutation(X1(1:N/2,:),bu,bd,pm,N/2);
    aX1 = SBX(aX,bu(:,xind),bd(:,xind),pc,N/2);
    aX2 = mutation(aX1(1:N/2,:),bu(:,xind),bd(:,xind),pm,N/2);
%     X2 = DE(X,Y,bu,bd,0.8,0.8,6);  % 6 --> DE/best/1/bin
%     aX2 = DE(aX,aY,bu(:,xind),bd(:,xind),0.8,0.8,6);
    X3 = X2;
    p_joint = 0.3;
    if rand < p_joint && numel(Xind) >= 2
        % Joint update: combine two blocks and apply SBX on merged indices.
        joint_blocks = randperm(numel(Xind), 2);
        joint_ind = [Xind{joint_blocks(1)}, Xind{joint_blocks(2)}];
        joint_ind = unique(joint_ind);
        jointX1 = SBX(X(:,joint_ind), bu(:,joint_ind), bd(:,joint_ind), pc, N/2);
        jointX2 = mutation(jointX1(1:N/2,:), bu(:,joint_ind), bd(:,joint_ind), pm, N/2);
        X3(:,joint_ind) = jointX2;
    else
        X3(:,xind) = aX2;
    end
    X = [X;X2;X3];
    aX3 = X2(:,xind);
    aX = [aX;aX2;aX3];
    y = Ens_predictor([X2;X3],Ens,Train,Xind,Mind,Sim,W,B,C,P);
%     ay = rbfpredict(model,train(:,1:end-1),[aX2;aX3]);
    ay = RBF_predictor(W(i,:),B(i),C{i},P(:,i),[aX2;aX3]);
    Y = [Y;y];
    aY = [aY;ay];
%     y = Ens_predictor([X2;X3],Ens,Train,Xind,Mind);
%     Y = [Y;y];
%     Y = Ens_predictor(X,Ens,Train,Xind,Mind);
%     aY = rbfpredict(model,train(:,1:end-1),aX);
    [sY,is1] = sort(Y);  %  ”¶÷µ≈≈–
    [saY,is2] = sort(aY);
    X = X(is1(1:N),:);  %—°‘Ò«∞N∏
    Y = Y(is1(1:N),1);
    aX = aX(is2(1:N),:);
    aY = aY(is2(1:N),1);
    g = g + 1;
end

end

function y = Ens_predictor(X,Ens,Train,Xind,Mind,Sim,W,B,C,P)

N = size(X,1);
K = length(Ens);
pre = zeros(N,K);
for i = 1 : K
    
    S = Train{Mind(i)};
%     pre(:,i) = rbfpredict(Ens{i},S(:,1:end-1),X(:,Xind{Mind(i)}));
    I = Mind(i);
    pre(:,i) = RBF_predictor(W(I,:),B(I),C{I},P(:,I),X(:,Xind{Mind(i)}));
end
% y = mean(pre,2);
sim = 1./Sim;
w = sim(Mind(1:K))/sum(sim(Mind(1:K)));
wpre = repmat(w,N,1) .* pre;
y = sum(wpre,2);
end
