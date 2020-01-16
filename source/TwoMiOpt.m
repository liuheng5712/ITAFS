function [Fp_critical, F_critical, Y_critical] = TwoMiOpt(Fp, F, Y, expense)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% this function optmizes two mutual informations : arg max: I(F,Y) - I(Fp,Y) to be > 0
%%%% input:  max_expense is maximal points allowed to inject
%%%% output: three poisoned vectors, expense is actual points injected

% test
% F = randi([1, 5], 100, 1);
% Fp = randi([3, 7], 100, 1);
% Y = randi([-1, 1], 100, 1);
% expense = 15;


F_backup = F;
Fp_backup = Fp;
Y_backup = Y;

N = size(F, 1);
M = expense;

%%% rescale the outcomes to consecutive natural nums starting at 1
%%% first rescale F
observation = 1;
F_set = unique(F);
for p = F_set'
    if p <= 0
        F(F == p) = (observation + max(F_set))*ones(size(F(F == p), 1),1);
        observation = observation + 1;
    end
end
observation = 1;
F_set = unique(F);
for p = F_set'
    F(F == p) = observation*ones(size(F(F == p), 1),1);
    observation = observation + 1;
end
%%% rescale Fp
observation = 1;
Fp_set = unique(Fp);
for p = Fp_set'
    if p <= 0
        Fp(Fp == p) = (observation + max(Fp_set))*ones(size(Fp(Fp == p), 1),1);
        observation = observation + 1;
    end
end
observation = 1;
Fp_set = unique(Fp);
for p = Fp_set'
    Fp(Fp == p) = observation*ones(size(Fp(Fp == p), 1),1);
    observation = observation + 1;
end
%%%% rescale Y
observation = 1;
Y_set = unique(Y);
for p = Y_set'
    if p <= 0
        Y(Y == p) = (observation + max(Y_set))*ones(size(Y(Y == p), 1),1);
        observation = observation + 1;
    end
end
observation = 1;
Y_set = unique(Y);
for p = Y_set'
    Y(Y == p) = observation*ones(size(Y(Y == p), 1),1);
    observation = observation + 1;
end

% objective is to argmax(I(F;Y) - I(Fp;Y))
%%% maximize the mi by injecting more samples, use constrianed optimization
data = [F Fp Y];
% calculate the distributions
% NOTE: the outcomes need to be consecutive nums between (1, *)
% marginal distributions
margFp = zeros(1, size(unique(Fp),1));
for p = 1:size(unique(Fp),1)
    margFp(p) = size(Fp(Fp == p),1)/N;
end
margF = zeros(1, size(unique(F),1));
for p = 1:size(unique(F),1)
    margF(p) = size(F(F == p),1)/N;
end
margY = zeros(1, size(unique(Y),1));
for p = 1:size(unique(Y),1)
    margY(p) = size(Y(Y == p),1)/N;
end
% joint distributions
joints = zeros(size(unique(F),1), size(unique(Y),1));
for p = 1:N
    joints(F(p), Y(p)) = joints(F(p), Y(p)) + 1;
end
joints = joints/N;
joints_p = zeros(size(unique(Fp),1), size(unique(Y),1));
for p = 1:N
    joints_p(Fp(p), Y(p)) = joints_p(Fp(p), Y(p)) + 1;
end
joints_p = joints_p/N;

% we first handle the occurances in injected region
% index and filter the non-occured
% the order is F, F_p, Y
track = [];
for i = 1:size(unique(F),1)
    for j = 1:size(unique(Fp),1)
        for k = 1:size(unique(Y),1)
            % if the two 2-dim joint outcome existed
            if size(find(data(:, 1) == i & data(:, 3) == k), 1) > 0 | size(find(data(:, 2) == j & data(:, 3) == k), 1)
                track = [track; [i, j, k]];
            end
        end
    end
end
F2 = unique(track(:, 1));
Fp2 = unique(track(:, 2));

% now we formulate the objective, start by the first term
maximizer = @(m) 0;
for i = F2'
    pf = margF(i);
    % corresponding indices of m's for F = f (i.e., i)
    T1 = find(track(:, 1) == i);
    part = @(m) -1*((pf*N + sum(m(T1)))/(N+M))*log((pf*N + sum(m(T1)))/(N+M))/log(2);
    maximizer = @(m) maximizer(m) + part(m);
end
% second term
for j = Fp2'
    pfp = margFp(j);
    % corresponding indices of m's for Fp = fp (i.e., j)
    T2 = find(track(:, 2) == j);
    part = @(m) ((pfp*N + sum(m(T2)))/(N+M))*log((pfp*N + sum(m(T2)))/(N+M))/log(2);
    maximizer = @(m) maximizer(m) + part(m);
end
% third term
for i = 1:size(unique(F),1)
    for k = 1:size(unique(Y),1)
        pfy = joints(i,k);
        if pfy > 0
            % corresponding indices of m's for (F,Y) = (f,y) (i.e., (i,k))
            T3 = find((track(:, 1) == i) & (track(:, 3) == k));
            part = @(m) ((pfy*N + sum(m(T3)))/(N+M))*log((pfy*N + sum(m(T3)))/(N+M))/log(2);
            maximizer = @(m) maximizer(m) + part(m);
        end
    end
end
% fourth term
for j = 1:size(unique(Fp),1)
    for k = 1:size(unique(Y),1)
        pfpy = joints_p(j, k);
        if pfpy > 0
            % corresponding indices of m's for (Fp,Y) = (fp,y) (i.e., (j,k))
            T4 = find((track(:, 2) == j) & (track(:, 3) == k));
            part = @(m) -1*((pfpy*N + sum(m(T4)))/(N+M))*log((pfpy*N + sum(m(T4)))/(N+M))/log(2);
            maximizer = @(m) maximizer(m) + part(m);
        end
    end
end
% then we handle the distributions not in injected region
F1 = setdiff(1:size(unique(F),1), F2);
Fp1 = setdiff(1:size(unique(Fp),1), Fp2);
for i = F1
    part = @(m) (N/(N+M))*margF(i)*log((N/(N+M))*margF(i))/log(2);
    maximizer = @(m) maximizer(m) - part(m);
end
for i = Fp1
    part = @(m) (N/(N+M))*margFp(i)*log((N/(N+M))*margFp(i))/log(2);
    maximizer = @(m) maximizer(m) + part(m);
end

objective = @(m) -1*maximizer(m);
% solve the constrained optimization with given expense
A = [];
b = [];
Aeq = ones(1, size(track, 1));
beq = M;
lb = zeros(1, size(track, 1));
ub = M*ones(1, size(track, 1));
m0 = zeros(1, size(track, 1));    

opts = optimset('Display','off', 'MaxIter', 5000, 'MaxFunEvals', 5000);
m = fmincon(objective, m0, A, b, Aeq, beq, lb, ub, [], opts);

m = round(m);
for i = 1:size(m, 2)
    if m(i)>0
        for j = 1:m(i)
            data = [data; track(i ,:)];
        end
    end
end
F_critical = data(:, 1);
Fp_critical = data(:, 2);
Y_critical = data(:, 3);  

% test the benchmark and hypothesis
% origin = mi(F_backup,Y_backup) - mi(Fp_backup, Y_backup)
% hypothesis = mi(F_critical,Y_critical) - mi(Fp_critical,Y_critical)


% Now we scale back to original outcomes
for i = N+1:size(F_critical,1)
    new_position = find(F_critical == F_critical(i));
    new_position = new_position(1);
    F_critical(i) = F_backup(new_position);
    
    new_position = find(Fp_critical == Fp_critical(i));
    new_position = new_position(1);
    Fp_critical(i) = Fp_backup(new_position);
    
    new_position = find(Y_critical == Y_critical(i));
    new_position = new_position(1);
    Y_critical(i) = Y_backup(new_position);
end
F_critical(1:N) = F_backup;
Fp_critical(1:N) = Fp_backup;
Y_critical(1:N) = Y_backup;

