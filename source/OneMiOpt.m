function [F_critical, Y_critical] = OneMiOpt(F, Y, expense)

%%%%%%%%%%%%%%%
%%% maximize the mi by injecting more samples, use constrianed optimization
% F = randi(3, 100, 1);
% Y = randi(5, 100, 1);
% M = 15;

M = expense;
data = [F Y];
benchmark = mi(F, Y);
N = size(F, 1);

% calculate the distributions
% NOTE: the outcomes need to be consecutive nums between (1, *)
margF = zeros(1, size(unique(F),1));
for p = 1:size(unique(F),1)
    margF(p) = size(F(F == p),1)/N;
end

margY = zeros(1, size(unique(Y),1));
for p = 1:size(unique(Y),1)
    margY(p) = size(Y(Y == p),1)/N;
end

joints = zeros(size(unique(F),1), size(unique(Y),1));
for p = 1:N
    joints(F(p), Y(p)) = joints(F(p), Y(p)) + 1;
end
joints = joints/N;

% now we create the objective (func handle) by string2func
% non zero joint outcomes, each row: [f\inF, y\inY, p(f,y)]
nonzero_joints = [];
for i = 1:size(unique(F),1)
    for j = 1:size(unique(Y),1)
        if joints(i,j) > 0
            nonzero_joints = [nonzero_joints; [i, j]];
        end
    end
end
F2 = unique(nonzero_joints(:, 1));
Y2 = unique(nonzero_joints(:, 2));

% now we build up the objective, start by the first term
maximizer = @(m) 0;
for i = F2'
    pf = margF(i);
    % corresponding indices of m's for F = f (i.e., i)
    T1 = find(nonzero_joints(:, 1) == i);
    part = @(m) -1*((pf*N + sum(m(T1)))/(N+M))*log((pf*N + sum(m(T1)))/(N+M))/log(2);
    maximizer = @(m) maximizer(m) + part(m);
end
% now the second term
for i = Y2'
    py = margY(i);
    % corresponding indices of m's for Y = y (i.e., i)
    T2 = find(nonzero_joints(:, 2) == i);
    part = @(m) -1*((py*N + sum(m(T2)))/(N+M))*log((py*N + sum(m(T2)))/(N+M))/log(2);
    maximizer = @(m) maximizer(m) + part(m);
end
% now the thrid term
for i = 1:size(nonzero_joints, 1)
    joint_outcome = nonzero_joints(i ,:);
    pfy = joints(joint_outcome(1), joint_outcome(2));
    % form the func
    part =@(m) (pfy*N+m(i))/(N+M)*log((pfy*N+m(i))/(N+M))/log(2);
    maximizer = @(m) maximizer(m) + part(m);
end
lambda = N/(N+M);

% the f\in F in injected dataset is F2, similarly Y2
F1 = setdiff(1:size(unique(F),1), F2);
Y1 = setdiff(1:size(unique(Y),1), Y2);
for i = F1
    part =  @(m) lambda*margF(i)*log(lambda*margF(i))/log(2);
    maximizer = @(m) maximizer(m) - part(m);
end
for i = Y1
    part =  @(m) lambda*margY(i)*log(lambda*margY(i))/log(2);
    maximizer = @(m) maximizer(m) - part(m);
end
maximizer = @(m) -1*maximizer(m);
A = [];
b = [];
Aeq = ones(1, size(nonzero_joints, 1));
beq = M;
lb = zeros(1, size(nonzero_joints, 1));
ub = M*ones(1, size(nonzero_joints, 1));
m0 = ones(1, size(nonzero_joints, 1));
% solve the constrained optimization
m = fmincon(maximizer, m0, A, b, Aeq, beq, lb, ub);
m = round(m);
for i = 1:size(m, 2)
    if m(i)>0
        for j = 1:m(i)
            data = [data; nonzero_joints(i ,:)];
        end
    end
end
F_critical = data(:, 1);
Y_critical = data(:, 2);


%benchmark
%mi(F_critical, Y_critical)










