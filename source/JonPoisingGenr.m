function [F0_critical, F1_critical, Y_critical, expense] = JonPoisingGenr(F0, F1, Y, max_expense)

% this is the over version, semi_greedy algorithm
%Performs the collaborated malicious injection of three vectors s.t.
%I(F1;Y) - I(F0;Y) is greater than zero, increase I(F1;Y) decrease I(F0;Y)
%Inputs: 
%F0, F1, Y: R.V.'s, maximum expense of appended points
%Outputs:
%F0_critical, F1_critical, Y: appended vectors; expense used

%setting up parameters manually
% F0 = randi([1,5],50,1);
% F0(F0 == 1) = 0;
% F1 = randi([1,5],50,1);
% Y = randi([1,5],50,1);
% max_expense = 0.5;


F0_backup = F0;
F1_backup = F1;
Y_backup = Y;


minimum_table = [];
n = size(F0, 1); 
points = ceil(n*max_expense);
expense = [];

%r.v.'s preprocessing: observation substitution, %this is for possible negative values in Y which can cause problem
observation = 1;
F0_set = unique(F0);
for p = F0_set'
    if p <= 0
        F0(F0 == p) = (observation + max(F0_set))*ones(size(F0(F0 == p), 1),1);
        observation = observation + 1;
    end
end
observation = 1;
F0_set = unique(F0);
for p = F0_set'
    F0(F0 == p) = observation*ones(size(F0(F0 == p), 1),1);
    observation = observation + 1;
end

observation = 1;
F1_set = unique(F1);
for p = F1_set'
    if p <= 0
        F1(F1 == p) = (observation + max(F1_set))*ones(size(F1(F1 == p), 1),1);
        observation = observation + 1;
    end
end
observation = 1;
F1_set = unique(F1);
for p = F1_set'
    F1(F1 == p) = observation*ones(size(F1(F1 == p), 1),1);
    observation = observation + 1;
end

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

%calculate the marginals
marginal_F0 = zeros(1, size(unique(F0),1));
for p = 1:size(unique(F0),1)
    marginal_F0(p) = size(F0(F0 == p),1)/n;
end

marginal_F1 = zeros(1, size(unique(F1),1));
for p = 1:size(unique(F1),1)
    marginal_F1(p) = size(F1(F1 == p),1)/n;
end

marginal_Y = zeros(1, size(unique(Y),1));
for p = 1:size(unique(Y),1)
    marginal_Y(p) = size(Y(Y == p),1)/n;
end

%joint prob of p(f0,y) and p(f1,y)
joints0 = zeros(size(unique(F0),1), size(unique(Y),1)); 
for p = 1:n
    joints0(F0(p), Y(p)) = joints0(F0(p), Y(p)) + 1;
end
joints0 = joints0/n;

joints1 = zeros(size(unique(F1),1), size(unique(Y),1)); 
for p = 1:n
    joints1(F1(p), Y(p)) = joints1(F1(p), Y(p)) + 1;
end
joints1 = joints1/n;


%find the initial solution of greedy poisoning method, which'll updating
score = zeros(size(unique(F0),1), size(unique(F1),1), size(unique(Y),1)); 
benchmark0 = mi(F0, Y);
benchmark1 = mi(F1, Y);
MItrend0 = benchmark0;  %keep track of I(F0;Y)
MItrend1 = benchmark1;  %keep track of I(F1;Y)
F0_critical = F0;  %new vectors will return
F1_critical = F1;  %new vectors will return
Y_critical = Y;    %new vectors will return
V0_critical = 0; %critical value of F0
V1_critical = 0; %critical value of F1
VY_critical = 0; %critical value of F0
minimum = 1000;

for p = 1:size(unique(F0),1)
    for q = 1:size(unique(F1),1)
        for s = 1:size(unique(Y),1)
            if(joints0(p,s) ~= 0) & (joints1(q,s) ~= 0)
                score(p,q,s) = benchmark0 + log(marginal_F0(p)*marginal_Y(s)/joints0(p,s))/log(2);
                score(p,q,s) = benchmark1 + log(marginal_F1(q)*marginal_Y(s)/joints1(q,s))/log(2) - score(p,q,s);
                if minimum > score(p,q,s)
                   minimum = score(p,q,s);
                   V0_critical = p;
                   V1_critical = q;
                   VY_critical = s;
                end
            end
        end
    end
end
stop_criteria = minimum/2;

delta1 = (1 - joints1(V1_critical, VY_critical)) - (1 - marginal_F1(V1_critical)) - (1 - marginal_Y(VY_critical));
beta1 = (h(F1)+marginal_F1(V1_critical)*log(marginal_F1(V1_critical))/log(2)) + (h(Y)+marginal_Y(VY_critical)*log(marginal_Y(VY_critical))/log(2));
beta1 = beta1 - (h([F1,Y]) + joints1(V1_critical,VY_critical)*log(joints1(V1_critical,VY_critical))/log(2));

delta0 = (1 - joints0(V0_critical, VY_critical)) - (1 - marginal_F0(V0_critical)) - (1 - marginal_Y(VY_critical));
beta0 = (h(F0)+marginal_F0(V0_critical)*log(marginal_F0(V0_critical))/log(2)) + (h(Y)+marginal_Y(VY_critical)*log(marginal_Y(VY_critical))/log(2));
beta0 = beta0 - (h([F0,Y]) + joints0(V0_critical,VY_critical)*log(joints0(V0_critical,VY_critical))/log(2));

%Appending bins
current_size = n;
for r = 1:points
    

    if minimum <= stop_criteria
        minimum_table = [minimum_table minimum];

        F0_critical = [F0_critical; V0_critical];
        F1_critical = [F1_critical; V1_critical];
        Y_critical = [Y_critical; VY_critical];

        MItrend0 = [MItrend0 mi(F0_critical, Y_critical)];
        MItrend1 = [MItrend1 mi(F1_critical, Y_critical)];
    end
    lambda = current_size/(n+r);
    
    % if goal achieved
    if mi(F0_critical, Y_critical) < mi(F1_critical, Y_critical) 
        expense = [expense r];
        break;
    end
    
    %parts depending on lambda to calculate Phi'(lambda) function
    part1 = (1-marginal_F1(V1_critical))*log(lambda*marginal_F1(V1_critical)+1-lambda)/log(2);
    part1 = part1 + (1-marginal_Y(VY_critical))*log(lambda*marginal_Y(VY_critical)+1-lambda)/log(2);
    part1 = part1 + (joints1(V1_critical,VY_critical)-1)*log(lambda*joints1(V1_critical,VY_critical)+1-lambda)/log(2);
    part0 = (1-marginal_F0(V0_critical))*log(lambda*marginal_F0(V0_critical)+1-lambda)/log(2);
    part0 = part0 + (1-marginal_Y(VY_critical))*log(lambda*marginal_Y(VY_critical)+1-lambda)/log(2);
    part0 = part0 + (joints0(V0_critical,VY_critical)-1)*log(lambda*joints0(V0_critical,VY_critical)+1-lambda)/log(2);
    minimum = delta1*log(lambda)/log(2) + beta1 + part1 - (delta0*log(lambda)/log(2) + beta0 + part0);
    
    %probabilty rescaling and score updating
    if minimum > stop_criteria
        for p = 1:size(unique(F0),1)
            if p == V0_critical
                marginal_F0(p) = lambda*marginal_F0(p)+1-lambda;
            else
                marginal_F0(p) = lambda*marginal_F0(p);
            end
        end
        for q = 1:size(unique(F1),1)
            if q == V1_critical
                marginal_F1(q) = lambda*marginal_F1(q)+1-lambda;
            else
                marginal_F1(q) = lambda*marginal_F1(q);
            end
        end
        for s = 1:size(unique(Y),1)
            if s == VY_critical
                marginal_Y(s) = lambda*marginal_Y(s)+1-lambda;
            else
                marginal_Y(s) = lambda*marginal_Y(s);
            end
        end
        for p = 1:size(unique(F0),1)
            for s = 1:size(unique(Y),1)
                if (p == V0_critical) & (s == VY_critical)
                    joints0(p,s) = lambda*joints0(p,s)+1-lambda;
                else
                    joints0(p,s) = lambda*joints0(p,s);
                end
            end
        end  
        for s = 1:size(unique(Y),1)
            for q = 1:size(unique(F1),1)
                if (q == V1_critical) & (s == VY_critical)
                    joints1(q,s) = lambda*joints1(q,s)+1-lambda;
                else
                    joints1(q,s) = lambda*joints1(q,s);
                end
            end
        end
        minimum = 10000;
        for s = 1:size(unique(Y),1)
            for p = 1:size(unique(F0),1)
                for q = 1:size(unique(F1),1)
                    if joints0(p,s) ~= 0 & joints1(q,s) ~= 0
                        score(p,q,s) = mi(F0_critical, Y_critical) + log(marginal_F0(p)*marginal_Y(s)/joints0(p,s))/log(2);
                        score(p,q,s) = mi(F1_critical, Y_critical) + log(marginal_F1(q)*marginal_Y(s)/joints1(q,s))/log(2) - score(p,q,s);
                        if minimum > score(p,q,s)
                            minimum = score(p,q,s);
                            V0_critical = p;
                            V1_critical = q;
                            VY_critical = s;
                        end
                    end  
                end
            end
        end
        stop_criteria = minimum/2;
        
        delta1 = (1 - joints1(V1_critical, VY_critical)) - (1 - marginal_F1(V1_critical)) - (1 - marginal_Y(VY_critical));
        beta1 = (h(F1_critical)+marginal_F1(V1_critical)*log(marginal_F1(V1_critical))/log(2)) + (h(Y_critical)+marginal_Y(VY_critical)*log(marginal_Y(VY_critical))/log(2));
        beta1 = beta1 - (h([F1_critical,Y_critical]) + joints1(V1_critical,VY_critical)*log(joints1(V1_critical,VY_critical))/log(2));

        delta0 = (1 - joints0(V0_critical, VY_critical)) - (1 - marginal_F0(V0_critical)) - (1 - marginal_Y(VY_critical));
        beta0 = (h(F0_critical)+marginal_F0(V0_critical)*log(marginal_F0(V0_critical))/log(2)) + (h(Y)+marginal_Y(VY_critical)*log(marginal_Y(VY_critical))/log(2));
        beta0 = beta0 - (h([F0_critical,Y_critical]) + joints0(V0_critical,VY_critical)*log(joints0(V0_critical,VY_critical))/log(2));
        current_size = n + r;
    end
end

if size(expense,1) ~= 0
    expense = expense(1);
else
    expense = -1;
end

%we convert three vectors back to their original labelling
for i = n+1:size(F0_critical,1)
    new_position = find(F0_critical == F0_critical(i));
    new_position = new_position(1);
    F0_critical(i) = F0_backup(new_position);
    
    new_position = find(F1_critical == F1_critical(i));
    new_position = new_position(1);
    F1_critical(i) = F1_backup(new_position);
    
    new_position = find(Y_critical == Y_critical(i));
    new_position = new_position(1);
    Y_critical(i) = Y_backup(new_position);
end
F0_critical(1:n) = F0_backup;
F1_critical(1:n) = F1_backup;
Y_critical(1:n) = Y_backup;


% figure
% plot(MItrend0);
% hold on
% plot(MItrend1);
% legend('MItrend0','MItrend1', 'location', 'best');
% xlabel('% points interpolated')
% ylabel('mutual information')
% title('MI updating w.r.t. interpolation')
