function [] = AttackAlternative(index, round, Toselect, TrTeRatio)

% this is the new version, where the poisoning is semi_greedy
%Inputs: index: data index; round: which bootstrap round, Toselect:
%features to select; TrTeRatio: ratio for split training and testing
%Output: we will save all intermediate results

% addpath /home/hengl/matlab/bin/scripts/ITAFS/source/
% addpath /home/hengl/matlab/bin/MIToolbox/
% addpath /home/hengl/matlab/bin/FEAST/


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test
% index = 34; 
% round = 25;
% Toselect = 0.15; 
% TrTeRatio = 0.75; %means 90% for training and 10% for testing

fid = fopen('/home/hengl/matlab/bin/scripts/ITAFS/names2');
file_name = {};
for i=1:34
    line = fgets(fid);
    if ischar(line)
        file_name{i} = line;
    end
end
expression = '[A-Za-z0-9-_]*';
for i=1:34
    str = regexp(file_name{i}, expression);
    file_name{i} = file_name{i}(1: str(1,2)-2);
end
path = strcat('/home/hengl/matlab/bin/scripts/ITAFS/data/', file_name{index}, '.mat');
file = load(path); 
data = struct2cell(file);
data = data{1};

%data bootstrapping
%rng(round);
n = size(data,1);
bootstrap = randi([1,n],1,n);
data = data(bootstrap ,:);

%training and testing data split
TrTeRatio = ceil(TrTeRatio * n);
train_data = data(1:TrTeRatio ,:);
test_data = data(TrTeRatio+1:n ,:);

n = size(train_data, 1);
f = size(train_data, 2) - 1;
y = train_data(:, size(train_data, 2));
Toselect = ceil(f*Toselect);

% Collect results in these vectors:
% get the results when data is clean
benchmark_mrmr = feast('mrmr', Toselect, train_data(:,1:f), y);
benchmark_jmi = feast('jmi', Toselect, train_data(:,1:f), y);
benchmark_mifs = feast('mifs', Toselect, train_data(:,1:f), y);
benchmark_cife = feast('cife', Toselect, train_data(:,1:f), y);
benchmark_cmim = feast('cmim', Toselect, train_data(:,1:f), y);
benchmark_icap = feast('icap', Toselect, train_data(:,1:f), y);
benchmark_disr = feast('disr', Toselect, train_data(:,1:f), y);

best_feature = benchmark_mrmr(1);
% keep track of AverageConsistency of SP posioning for 3 objectives
SPConsistency_mrmr = [1];  
SPConsistency_jmi = [1];
SPConsistency_mifs = [1];
SPConsistency_cife = [1];
SPConsistency_cmim = [1];
SPConsistency_icap = [1];
SPConsistency_disr = [1];

% keep track of data points appended at each poisoning
Expense = [0]; 

%get the arranged feature sequence w.r.t. MI
reles = [];  
for i=1:f
    reles = [reles mi(train_data(:,i),y)];
end
feature_set = 1:f; %To get rid of the all-zero columns
[B, I] = sort(reles, 'descend');
sorted_features = feature_set(I);

if f <= 500
    num_to_poison = f;
else
    num_to_poison = 500;
end
for i = 2:num_to_poison
    expense = n;
    % we only use 100 or less poisoning cost choices
    if size(1:n, 2) < 1000
        range = 1:n;
    else
        range = linspace(1,n,100);
    end
    for ii = range
        % F0_critical: poisoned originally best feature, F1_critical: poisoned newly best feature
        [F0_critical, F1_critical, Y_critical] = TwoMiOpt(train_data(:, best_feature), train_data(:, sorted_features(i)), y, ceil(ii));
        if mi(F1_critical, Y_critical) - mi(F0_critical, Y_critical) > 0
            expense = size(F0_critical, 1) - n;
            break
        end
    end
    if mi(F1_critical, Y_critical) - mi(F0_critical, Y_critical) > 0%the poisoning is successful with allowed expense
        Expense(i) = expense;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %we can use random sampling to get the carrier for malicious features, BUT Rememeber to change the mrmr and mifs SOP poisoning codes ！！！
        malicious_data = [train_data; train_data(randi([1,n],1,expense) ,:)];
        %SP Poisoning: insert three malicious vectors to above carrying data
        malicious_data(:, best_feature) = F0_critical(1:n+expense);
        malicious_data(:, sorted_features(i)) = F1_critical(1:n+expense);
        malicious_data(:, f+1) = Y_critical(1:n+expense);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %1. testing consistency using mRMR objective
        selected = feast('mrmr', Toselect, malicious_data(:,1:f), malicious_data(:,f+1));
        temp_consistency = [];
        for k = 1:Toselect
            c = size(intersect(selected(1:k), benchmark_mrmr(1:k)),1);
            temp_consistency = [temp_consistency (c*f - k*k)/(k*f - k*k)];
        end
        if mean(temp_consistency) <= min(SPConsistency_mrmr)
            train_SP_mrmr = malicious_data; %keep track of the most malicious scenario
            SPselected_mrmr = selected;
        end
        SPConsistency_mrmr(i) = mean(temp_consistency);

        %2. testing consistency using JMI objective
        selected = feast('jmi', Toselect, malicious_data(:,1:f), malicious_data(:,f+1));
        temp_consistency = [];
        for k = 1:Toselect
            c = size(intersect(selected(1:k), benchmark_jmi(1:k)),1);
            temp_consistency = [temp_consistency (c*f - k*k)/(k*f - k*k)];
        end
        if mean(temp_consistency) <= min(SPConsistency_jmi)
            train_SP_jmi = malicious_data; %keep track of the most malicious scenario
            SPselected_jmi = selected;
        end
        SPConsistency_jmi(i) = mean(temp_consistency);

        %3. testing consistency using MIFS objective
        selected = feast('mifs', Toselect, malicious_data(:,1:f), malicious_data(:,f+1));
        temp_consistency = [];
        for k = 1:Toselect
            c = size(intersect(selected(1:k), benchmark_mifs(1:k)),1);
            temp_consistency = [temp_consistency (c*f - k*k)/(k*f - k*k)];
        end
        if mean(temp_consistency) <= min(SPConsistency_mifs)
            train_SP_mifs = malicious_data; %keep track of the most malicious scenario
            SPselected_mifs = selected;
        end
        SPConsistency_mifs(i) = mean(temp_consistency);
        
        %5. testing consistency using CIFE objective
        selected = feast('cife', Toselect, malicious_data(:,1:f), malicious_data(:,f+1));
        temp_consistency = [];
        for k = 1:Toselect
            c = size(intersect(selected(1:k), benchmark_cife(1:k)),1);
            temp_consistency = [temp_consistency (c*f - k*k)/(k*f - k*k)];
        end
        if mean(temp_consistency) <= min(SPConsistency_cife)
            train_SP_cife = malicious_data; %keep track of the most malicious scenario
            SPselected_cife = selected;
        end
        SPConsistency_cife(i) = mean(temp_consistency);
        
        %6. testing consistency using CMIM objective
        selected = feast('cmim', Toselect, malicious_data(:,1:f), malicious_data(:,f+1));
        temp_consistency = [];
        for k = 1:Toselect
            c = size(intersect(selected(1:k), benchmark_cmim(1:k)),1);
            temp_consistency = [temp_consistency (c*f - k*k)/(k*f - k*k)];
        end
        if mean(temp_consistency) <= min(SPConsistency_cmim)
            train_SP_cmim = malicious_data; %keep track of the most malicious scenario
            SPselected_cmim = selected;
        end
        SPConsistency_cmim(i) = mean(temp_consistency);
        
        %7. testing consistency using ICAP objective
        selected = feast('icap', Toselect, malicious_data(:,1:f), malicious_data(:,f+1));
        temp_consistency = [];
        for k = 1:Toselect
            c = size(intersect(selected(1:k), benchmark_icap(1:k)),1);
            temp_consistency = [temp_consistency (c*f - k*k)/(k*f - k*k)];
        end
        if mean(temp_consistency) <= min(SPConsistency_icap)
            train_SP_icap = malicious_data; %keep track of the most malicious scenario
            SPselected_icap = selected;
        end
        SPConsistency_icap(i) = mean(temp_consistency);
        
        %8. testing consistency using DISR objective
        selected = feast('disr', Toselect, malicious_data(:,1:f), malicious_data(:,f+1));
        temp_consistency = [];
        for k = 1:Toselect
            c = size(intersect(selected(1:k), benchmark_disr(1:k)),1);
            temp_consistency = [temp_consistency (c*f - k*k)/(k*f - k*k)];
        end
        if mean(temp_consistency) <= min(SPConsistency_disr)
            train_SP_disr = malicious_data; %keep track of the most malicious scenario
            SPselected_disr = selected;
        end
        SPConsistency_disr(i) = mean(temp_consistency);
    else
        Expense(i) = n; 
        
        SPConsistency_mrmr(i) = 1;
        SPConsistency_jmi(i) = 1;
        SPConsistency_mifs(i) = 1;
        SPConsistency_cife(i) = 1;
        SPConsistency_cmim(i) = 1;
        SPConsistency_icap(i) = 1;
        SPConsistency_disr(i) = 1;
    end
end

% in some cases it will fail to poison due to cost is too high
try
    %Now we get the classification accuracy of knn model from various training data on testing data 
    cross_validation_knn = zeros(3,7);
    %We measure jmi first
    mdl = fitcknn(train_data(:, benchmark_jmi), train_data(:,f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, benchmark_jmi));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(1,1) = err;

    mdl = fitcknn(train_SP_jmi(:, SPselected_jmi), train_SP_jmi(:, f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, SPselected_jmi));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(2,1) = err;

    %we now measure mrmr 
    mdl = fitcknn(train_data(:, benchmark_mrmr), train_data(:,f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, benchmark_mrmr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(1,2) = err;

    mdl = fitcknn(train_SP_mrmr(:, SPselected_mrmr), train_SP_mrmr(:, f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, SPselected_mrmr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(2,2) = err;

    %we now measure mifs
    mdl = fitcknn(train_data(:, benchmark_mifs), train_data(:,f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, benchmark_mifs));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(1,3) = err;

    mdl = fitcknn(train_SP_mifs(:, SPselected_mifs), train_SP_mifs(:, f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, SPselected_mifs));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(2,3) = err;

    %we now measure cife
    mdl = fitcknn(train_data(:, benchmark_cife), train_data(:,f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, benchmark_cife));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(1,4) = err;

    mdl = fitcknn(train_SP_cife(:, SPselected_cife), train_SP_cife(:, f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, SPselected_cife));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(2,4) = err;

    %we now measure cmim
    mdl = fitcknn(train_data(:, benchmark_cmim), train_data(:,f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, benchmark_cmim));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(1,5) = err;

    mdl = fitcknn(train_SP_cmim(:, SPselected_cmim), train_SP_cmim(:, f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, SPselected_cmim));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(2,5) = err;

    %we now measure icap
    mdl = fitcknn(train_data(:, benchmark_icap), train_data(:,f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, benchmark_icap));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(1,6) = err;

    mdl = fitcknn(train_SP_icap(:, SPselected_icap), train_SP_icap(:, f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, SPselected_icap));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(2,6) = err;

    %we now measure disr
    mdl = fitcknn(train_data(:, benchmark_disr), train_data(:,f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, benchmark_disr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(1,7) = err;

    mdl = fitcknn(train_SP_disr(:, SPselected_disr), train_SP_disr(:, f+1), 'NumNeighbors', 5);
    resp = predict(mdl, test_data(:, SPselected_disr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_knn(2,7) = err;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Now we get the classification accuracy of decision tree model from various training data on testing data 
    cross_validation_tree = zeros(3,7);
    %We measure jmi first
    mdl = fitctree(train_data(:, benchmark_jmi), train_data(:,f+1));
    resp = predict(mdl, test_data(:, benchmark_jmi));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(1,1) = err;

    mdl = fitctree(train_SP_jmi(:, SPselected_jmi), train_SP_jmi(:, f+1));
    resp = predict(mdl, test_data(:, SPselected_jmi));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(2,1) = err;

    %we now measure mrmr 
    mdl = fitctree(train_data(:, benchmark_mrmr), train_data(:,f+1));
    resp = predict(mdl, test_data(:, benchmark_mrmr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(1,2) = err;

    mdl = fitctree(train_SP_mrmr(:, SPselected_mrmr), train_SP_mrmr(:, f+1));
    resp = predict(mdl, test_data(:, SPselected_mrmr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(2,2) = err;

    %we now measure mifs
    mdl = fitctree(train_data(:, benchmark_mifs), train_data(:,f+1));
    resp = predict(mdl, test_data(:, benchmark_mifs));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(1,3) = err;

    mdl = fitctree(train_SP_mifs(:, SPselected_mifs), train_SP_mifs(:, f+1));
    resp = predict(mdl, test_data(:, SPselected_mifs));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(2,3) = err;

    %we now measure cife
    mdl = fitctree(train_data(:, benchmark_cife), train_data(:,f+1));
    resp = predict(mdl, test_data(:, benchmark_cife));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(1,4) = err;

    mdl = fitctree(train_SP_cife(:, SPselected_cife), train_SP_cife(:, f+1));
    resp = predict(mdl, test_data(:, SPselected_cife));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(2,4) = err;

    %we now measure cmim
    mdl = fitctree(train_data(:, benchmark_cmim), train_data(:,f+1));
    resp = predict(mdl, test_data(:, benchmark_cmim));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(1,5) = err;

    mdl = fitctree(train_SP_cmim(:, SPselected_cmim), train_SP_cmim(:, f+1));
    resp = predict(mdl, test_data(:, SPselected_cmim));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(2,5) = err;

    %we now measure icap
    mdl = fitctree(train_data(:, benchmark_icap), train_data(:,f+1));
    resp = predict(mdl, test_data(:, benchmark_icap));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(1,6) = err;

    mdl = fitctree(train_SP_icap(:, SPselected_icap), train_SP_icap(:, f+1));
    resp = predict(mdl, test_data(:, SPselected_icap));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(2,6) = err;

    %we now measure disr
    mdl = fitctree(train_data(:, benchmark_disr), train_data(:,f+1));
    resp = predict(mdl, test_data(:, benchmark_disr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(1,7) = err;

    mdl = fitctree(train_SP_disr(:, SPselected_disr), train_SP_disr(:, f+1));
    resp = predict(mdl, test_data(:, SPselected_disr));
    err = (size(test_data, 1) - size(find(resp == test_data(:, f+1)),1))/size(test_data, 1);
    cross_validation_tree(2,7) = err;

    save(strcat(file_name{index}, num2str(round)));
catch ME
    disp(ME)
end

% h = figure;
% yyaxis left
% plot(1:f, NAveConsistency_mrmr)
% hold on 
% plot(1:f, SPConsistency_mrmr)
% plot(1:f, SOPConsistency_mrmr)
% legend('Sampled','Technique1','Technique2','location','best')
% ylabel('consistency')
% 
% yyaxis right
% plot(1:f, Expense/n)
% ylabel('expense')
% title(file_name{index})
% xlabel('$X^i_t$', 'Interpreter', 'latex')

% h = figure;
% subplot(3,2,1)
% plot(1:f, AveConsistency_mrmr)
% hold on
% plot(1:f, BAveConsistency_mrmr)
% legend('1orPoisnMRMR', 'SampledMRMR')
% xlabel('$X^i_t$', 'Interpreter', 'latex')
% ylabel('Kuncheva consistency')
% title(strcat('consistency', '-', file_name{index}))
% 
% subplot(3,2,2)
% plot(1:f, AveConsistency_jmi)
% hold on
% plot(1:f, BAveConsistency_jmi)
% legend('1orPoisnJMI', 'SampledJMI')
% xlabel('$X^i_t$', 'Interpreter', 'latex')
% ylabel('Kuncheva consistency')
% title(strcat('consistency', '-', file_name{index}))
% 
% subplot(3,2,3)
% plot(1:f, AveConsistency_mifs)
% hold on
% plot(1:f, BAveConsistency_mifs)
% legend('1orPoisnMifs', 'SampledMifs')
% xlabel('$X^i_t$', 'Interpreter', 'latex')
% ylabel('Kuncheva consistency')
% title(strcat('consistency', '-', file_name{index}))
% 
% subplot(3,2,4)
% plot(1:f, AveConsistency_mrmr)
% hold on
% plot(1:f, AveConsistency_jmi)
% plot(1:f, AveConsistency_mifs)
% legend('mrmr', 'jmi', 'mifs', 'location', 'best')
% xlabel('$X^i_t$', 'Interpreter', 'latex')
% ylabel('Kuncheva consistency')
% title(strcat('consistency', '-', file_name{index}))
% 
% subplot(3,2,5)
% plot(1:f, AveConsistency_mrmr)
% hold on
% plot(1:f, AveConsistency_mrmr2)
% plot(1:f, zeros(1,f))
% legend('mrmr-1order', 'mrmr-2order', 'zero', 'location', 'best')
% xlabel('$X^i_t$', 'Interpreter', 'latex')
% ylabel('Kuncheva consistency')
% title(strcat('consistency', '-', file_name{index}))
% 
% subplot(3,2,6)
% plot(1:f, Expense/n)
% xlabel('$X^i_t$', 'Interpreter', 'latex')
% ylabel('points consumed')
% title(strcat('cost', '-', file_name{index}))

%saveas(h, strcat(file_name{index}, '.eps'), 'eps2c')
%delete(gcp('nocreate'));

%clear;clc
