% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %clean and collect results
% clear;clc
% location = '/home/hengl/matlab/bin/scripts/ITAFS/results/';
% data = 'penglymp';
% boots = 7;
% 
% bootstrap = [];
% 
% % store the consistencies across different rounds
% SPConsistency_mrmr = [];
% SPConsistency_jmi = [];
% SPConsistency_mifs = [];
% SPConsistency_cife = [];
% SPConsistency_cmim = [];
% SPConsistency_icap = [];
% SPConsistency_disr = [];
% 
% % store the malicious dataset (associated w/ minimal poisoning
% % consistency) at each round
% train_SP_mrmr = {};
% train_SP_jmi = {};
% train_SP_mifs = {};
% train_SP_cife = {};
% train_SP_cmim = {};
% train_SP_icap = {};
% train_SP_disr = {};
% 
% Expense = [];
% 
% CVknn = {};
% CVtree = {};
% for i = 1:boots
%  path = strcat(location, data, 'EW', num2str(i));
% 
%  %concatenating different consistency results to matrix
%  %consistency for jmi
%  temp = load(path, 'SPConsistency_jmi');
%  SPConsistency_jmi = [SPConsistency_jmi; temp.('SPConsistency_jmi')];
% 
%  %consistency for mifs
%  temp = load(path, 'SPConsistency_mifs');
%  SPConsistency_mifs = [SPConsistency_mifs; temp.('SPConsistency_mifs')];
% 
%  %consistency for mrmr
%  temp = load(path, 'SPConsistency_mrmr');
%  SPConsistency_mrmr = [SPConsistency_mrmr; temp.('SPConsistency_mrmr')];
% 
%  %consistency for cife
%  temp = load(path, 'SPConsistency_cife');
%  SPConsistency_cife = [SPConsistency_cife; temp.('SPConsistency_cife')];
% 
%  %consistency for cmim
%  temp = load(path, 'SPConsistency_cmim');
%  SPConsistency_cmim = [SPConsistency_cmim; temp.('SPConsistency_cmim')];
% 
%  %consistency for icap
%  temp = load(path, 'SPConsistency_icap');
%  SPConsistency_icap = [SPConsistency_icap; temp.('SPConsistency_icap')];
% 
%  %consistency for disr
%  temp = load(path, 'SPConsistency_disr');
%  SPConsistency_disr = [SPConsistency_disr; temp.('SPConsistency_disr')];
% 
%  % collect the malicious dataset (associated w/ minimal poisoning consistency)
%  % malicious dataset for mrmr
%  temp = load(path, 'train_SP_mrmr');
%  train_SP_mrmr{i} = temp;
% 
%  % malicious dataset for jmi
%  temp = load(path, 'train_SP_jmi');
%  train_SP_jmi{i} = temp;
% 
%  % malicious dataset for mifs
%  temp = load(path, 'train_SP_mifs');
%  train_SP_mifs{i} = temp;
% 
%  % malicious dataset for cife
%  temp = load(path, 'train_SP_cife');
%  train_SP_cife{i} = temp;
% 
%  % malicious dataset for cmim
%  temp = load(path, 'train_SP_cmim');
%  train_SP_cmim{i} = temp;
% 
%  % malicious dataset for icap
%  temp = load(path, 'train_SP_icap');
%  train_SP_icap{i} = temp;
% 
%  % malicious dataset for disr
%  temp = load(path, 'train_SP_disr');
%  train_SP_disr{i} = temp;
% 
% 
%  %collect the acurracy results
%  temp = load(path, 'cross_validation_knn');
%  CVknn{i} = temp.('cross_validation_knn');
%  temp = load(path, 'cross_validation_tree');
%  CVtree{i} = temp.('cross_validation_tree');
% 
%  temp = load(path, 'Expense');
%  Expense = [Expense; temp.('Expense')];
% 
%  temp = load(path, 'bootstrap');
%  bootstrap = [bootstrap; temp.('bootstrap')];
% 
% end
% load(path, 'Toselect', 'n', 'f');
% 
% knn_average = zeros(3,7);
% for i = 1:boots
%  knn_average = knn_average + CVknn{i};
% end
% knn_average = knn_average/30;
% 
% tree_average = zeros(3,7);
% for i = 1:boots
%  tree_average = tree_average + CVtree{i};
% end
% tree_average = tree_average/30;
% 
% %find all the costs
% cost_knn = {};
% cost_tree = {};
% for i = 1:boots
%  cost_knn{i} = [0 0 0 0 0 0 0];
%  cost_tree{i} = [0 0 0 0 0 0 0];
% 
%  %for SP poisoning
%  row_add_knn = [];
%  row_add_tree = [];
%  %for jmi
%  consistency = SPConsistency_jmi(i,:);
%  costs = Expense(i,:);
%  minimum = min(consistency);
%  where = find(consistency == minimum);
%  where = where(size(where,2));
%  cost = costs(where);
%  row_add_knn = [row_add_knn cost];
%  row_add_tree = [row_add_tree cost];
% 
%  %for mrmr
%  consistency = SPConsistency_mrmr(i,:);
%  costs = Expense(i,:);
%  minimum = min(consistency);
%  where = find(consistency == minimum);
%  where = where(size(where,2));
%  cost = costs(where);
%  row_add_knn = [row_add_knn cost];
%  row_add_tree = [row_add_tree cost];
% 
%  %for mifs
%  consistency = SPConsistency_mifs(i,:);
%  costs = Expense(i,:);
%  minimum = min(consistency);
%  where = find(consistency == minimum);
%  where = where(size(where,2));
%  cost = costs(where);
%  row_add_knn = [row_add_knn cost];
%  row_add_tree = [row_add_tree cost]; 
% 
%  %for cife
%  consistency = SPConsistency_cife(i,:);
%  costs = Expense(i,:);
%  minimum = min(consistency);
%  where = find(consistency == minimum);
%  where = where(size(where,2));
%  cost = costs(where);
%  row_add_knn = [row_add_knn cost];
%  row_add_tree = [row_add_tree cost];
% 
%  %for cmim
%  consistency = SPConsistency_cmim(i,:);
%  costs = Expense(i,:);
%  minimum = min(consistency);
%  where = find(consistency == minimum);
%  where = where(size(where,2));
%  cost = costs(where);
%  row_add_knn = [row_add_knn cost];
%  row_add_tree = [row_add_tree cost];
% 
%  %for icap
%  consistency = SPConsistency_icap(i,:);
%  costs = Expense(i,:);
%  minimum = min(consistency);
%  where = find(consistency == minimum);
%  where = where(size(where,2));
%  cost = costs(where);
%  row_add_knn = [row_add_knn cost];
%  row_add_tree = [row_add_tree cost];
% 
%  %for disr
%  consistency = SPConsistency_disr(i,:);
%  costs = Expense(i,:);
%  minimum = min(consistency);
%  where = find(consistency == minimum);
%  where = where(size(where,2));
%  cost = costs(where);
%  row_add_knn = [row_add_knn cost];
%  row_add_tree = [row_add_tree cost];
% 
%  cost_knn{i} = [cost_knn{i}; row_add_knn];
%  cost_tree{i} = [cost_tree{i}; row_add_knn];
% end
% 
% cost_knn_average = zeros(2,7);
% cost_tree_average = zeros(2,7);
% for i = 1:boots
%  cost_knn_average = cost_knn_average + cost_knn{i};
% end
% cost_knn_average = cost_knn_average/30;
% for i = 1:boots
%  cost_tree_average = cost_tree_average + cost_tree{i};
% end
% cost_tree_average = cost_tree_average/30;
% 
% clearvars -except CVknn CVtree Expense train_SP_mrmr train_SP_jmi train_SP_mifs train_SP_cife train_SP_cmim train_SP_icap train_SP_disr SPConsistency_jmi SPConsistency_mifs SPConsistency_mrmr SPConsistency_cife SPConsistency_cmim SPConsistency_icap SPConsistency_disr Toselect bootstrap data f knn_average n tree_average cost_knn cost_tree cost_knn_average cost_tree_average  
% save(data)
% clear;clc

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %plot the consistency results
% data = 'wine';
% 
% h1 = figure;
% left_color = [0 0 0];
% right_color = [0 0 0];
% set(h1, 'defaultAxesColorOrder', [left_color; right_color]);
% 
% yyaxis left
% hold on 
% f = size(SPConsistency_cife, 2);
% plot(1:f, mean(SPConsistency_jmi), '-o', 'color', 'b',  'lineWidth', 1.5, 'MarkerSize', 2)
% plot(1:f, mean(SPConsistency_mrmr), '-o', 'color', 'g', 'lineWidth', 1.5, 'MarkerSize', 2)
% plot(1:f, mean(SPConsistency_mifs), '-o', 'color', 'r', 'lineWidth', 1.5, 'MarkerSize', 2)
% plot(1:f, mean(SPConsistency_cmim), '-o', 'color', 'c', 'lineWidth', 1.5, 'MarkerSize', 2)
% plot(1:f, mean(SPConsistency_icap), '-o', 'color', 'm', 'lineWidth', 1.5, 'MarkerSize', 2)
% plot(1:f, mean(SPConsistency_disr), '-o', 'color', 'y', 'lineWidth', 1.5, 'MarkerSize', 2)
% ylabel('Consistency', 'FontSize', 20)
% 
% box on
% yyaxis right
% plot(1:f, mean(Expense)/n, 'lineWidth', 2, 'color', 'black')
% ylabel('Injected Samples', 'FontSize', 20)
% xlabel('$F_i$', 'Interpreter', 'latex', 'FontSize', 20)
% axis tight
% 
% lgd = legend('JMI', 'MRMR', 'MIFS', 'CMIM', 'ICAP', 'DISR', 'Cost', 'location','north');
% set(lgd, 'FontSize', 10)
% saveas(h1, data, 'eps2c')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here for each dataset, we plot the box plot for different FS objectives
% so we can compare their performance when under attack
% data = 'wine';
% h = figure;
% location = '/home/hengl/matlab/bin/scripts/ITAFS/results/version2/';
% load(strcat(location, data, '.mat'));
% 
% consistency = [mean(SPConsistency_jmi)', mean(SPConsistency_mifs)', mean(SPConsistency_mrmr)', mean(SPConsistency_cmim)', mean(SPConsistency_icap)', mean(SPConsistency_disr)'];
% boxplot(consistency, {'JMI', 'MIFS', 'MRMR', 'CMIM', 'ICAP', 'DISR'});
% xlabel('Feature Selection Objective', 'FontSize', 15);
% ylabel('Consistency', 'FontSize', 15);
% saveas(h, strcat(data, 'box'), 'eps2c')


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %get the accuracy results, also conduct the wilcoxon comparison for each FS objective
% %alg: 1 for jmi, 2 for mrmr, 3 for mifs, 4 for cife, 5 for cmim, 6 for icap, 7 for disr
% location = '/home/hengl/matlab/bin/scripts/ITAFS/results/version2/'; 
% datasets = {'lungcancer', 'breast', 'congress', 'heart', 'ionosphere', 'krvskp', 'landsat', 'parkinsons', 'pengcolon', 'penglung', 'semeion', 'sonar', 'spect', 'splice', 'waveform', 'wine'}; 
% 
% % we store the results for each FS to do wilcoxon ranked sum test
% JMI = [];
% MRMR = [];
% MIFS = [];
% CMIM = [];
% ICAP = [];
% DISR = [];
% for i = 1:16
%     load(strcat(location, datasets{i})); 
%     line = strcat(data, '&', num2str(n), '&'); 
%     for alg = [1 2 3 5 6 7]
%         knn_average = round(knn_average, 4);
%         tree_average = round(tree_average, 4);
%         cost_knn_average = round(cost_knn_average, 0);
%         cost_tree_average = round(cost_tree_average, 0);
%         if knn_average(1,alg) < knn_average(2,alg)
%             line = strcat(line, num2str(knn_average(1,alg)), '&', num2str(knn_average(2,alg)), '(', num2str(cost_knn_average(2,alg)) ,')', '&');
%         else
%             line = strcat(line, num2str(knn_average(1,alg)), '&', '{\bf', num2str(knn_average(2,alg)), '(', num2str(cost_knn_average(2,alg)) ,')', '}', '&');
%         end
%         if tree_average(1,alg) < tree_average(2,alg)
%             line = strcat(line, num2str(tree_average(1,alg)), '&', num2str(tree_average(2,alg)),'(', num2str(cost_knn_average(2,alg)) ,')', '&');
%         else
%             line = strcat(line, num2str(tree_average(1,alg)), '&', '{\bf', num2str(tree_average(2,alg)),'(', num2str(cost_knn_average(2,alg)) ,')', '}', '&');
%         end
%         
%         % collect the results for wilcoxon ranked sum test
%         if alg == 1
%             JMI = [JMI; [knn_average(1,alg), knn_average(2,alg), tree_average(1,alg), tree_average(2,alg)]];
%         elseif alg == 2
%             MRMR = [MRMR; [knn_average(1,alg), knn_average(2,alg), tree_average(1,alg), tree_average(2,alg)]];
%         elseif alg == 3
%             MIFS = [MIFS; [knn_average(1,alg), knn_average(2,alg), tree_average(1,alg), tree_average(2,alg)]];
%         elseif alg == 5
%             CMIM = [CMIM; [knn_average(1,alg), knn_average(2,alg), tree_average(1,alg), tree_average(2,alg)]];
%         elseif alg == 6
%             ICAP = [ICAP; [knn_average(1,alg), knn_average(2,alg), tree_average(1,alg), tree_average(2,alg)]];
%         elseif alg == 7
%             DISR = [DISR; [knn_average(1,alg), knn_average(2,alg), tree_average(1,alg), tree_average(2,alg)]];
%         end
%     end
%     line = strcat(line, '\\');
%     %disp(line)
% end
% % 
% [p, h, stats1] = signrank(JMI(:,1),JMI(:,2)); 
% [p, h, stats2] = signrank(JMI(:,3),JMI(:,4)); 
% [p, h, stats3] = signrank(MRMR(:,1),MRMR(:,2)); 
% [p, h, stats4] = signrank(MRMR(:,3),MRMR(:,4)); 
% [p, h, stats5] = signrank(MIFS(:,1),MIFS(:,2)); 
% [p, h, stats6] = signrank(MIFS(:,3),MIFS(:,4)); 
% [p, h, stats7] = signrank(CMIM(:,1),CMIM(:,2)); 
% [p, h, stats8] = signrank(CMIM(:,3),CMIM(:,4)); 
% [p, h, stats9] = signrank(ICAP(:,1),ICAP(:,2)); 
% [p, h, stats10] = signrank(ICAP(:,3),ICAP(:,4)); 
% [p, h, stats11] = signrank(DISR(:,1),DISR(:,2)); 
% [p, h, stats12] = signrank(DISR(:,3),DISR(:,4)); 
% stats1
% stats2
% stats3
% stats4
% stats5
% stats6
% stats7
% stats8
% stats9
% stats10
% stats11
% stats12

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we here do T-sne plot
% clear;clc
% location = '/home/hengl/matlab/bin/scripts/ITAFS/results/version2/'; 
% dataset = 'parkinsons';
% load(strcat(location, strcat(dataset, '.mat'))); 
% TrTeRatio = ceil(0.75 * n); % training size, other things are malicious
% 
% % fetch the generated malicious data
% train_SP_jmi = train_SP_jmi{1}.train_SP_jmi;
% train_SP_mrmr = train_SP_mrmr{1}.train_SP_mrmr;
% train_SP_mifs = train_SP_mifs{1}.train_SP_mifs;
% train_SP_cmim = train_SP_cmim{1}.train_SP_cmim;
% train_SP_icap = train_SP_icap{1}.train_SP_icap;
% train_SP_disr = train_SP_disr{1}.train_SP_disr;
% 
% % generate the labels of malicious/benign
% dictionary = {'benign', 'malicious'};
% train_SP_jmi_label = [ones(TrTeRatio, 1); 2*ones( size(train_SP_jmi,1)-TrTeRatio, 1 )];
% train_SP_mrmr_label = [ones(TrTeRatio, 1); 2*ones( size(train_SP_mrmr,1)-TrTeRatio, 1 )];
% train_SP_mifs_label = [ones(TrTeRatio, 1); 2*ones( size(train_SP_mifs,1)-TrTeRatio, 1 )];
% train_SP_cmim_label = [ones(TrTeRatio, 1); 2*ones( size(train_SP_cmim,1)-TrTeRatio, 1 )];
% train_SP_icap_label = [ones(TrTeRatio, 1); 2*ones( size(train_SP_icap,1)-TrTeRatio, 1 )];
% train_SP_disr_label = [ones(TrTeRatio, 1); 2*ones( size(train_SP_disr,1)-TrTeRatio, 1 )];
% 
% % now plot the T-sne's in one figure
% h1 = figure;
% Y = tsne(train_SP_jmi(:, 1:f));
% gscatter(Y(:,1), Y(:,2), dictionary(train_SP_jmi_label)');
% xlabel('Dim 1', 'FontSize', 10)
% ylabel('Dim 2', 'FontSize', 10)
% title('JMI', 'FontSize', 10)
% set(legend, 'FontSize', 10)
% saveas(h1, strcat(dataset, '_JMI'), 'eps2c')
% 
% h2 = figure;
% Y = tsne(train_SP_mrmr(:, 1:f));
% gscatter(Y(:,1), Y(:,2), dictionary(train_SP_mrmr_label)');
% xlabel('Dim 1', 'FontSize', 10)
% ylabel('Dim 2', 'FontSize', 10)
% title('MRMR', 'FontSize', 10)
% saveas(h2, strcat(dataset, '_MRMR'), 'eps2c')
% 
% h3 = figure;
% Y = tsne(train_SP_mifs(:, 1:f));
% gscatter(Y(:,1), Y(:,2), dictionary(train_SP_mifs_label)');
% xlabel('Dim 1', 'FontSize', 10)
% ylabel('Dim 2', 'FontSize', 10)
% title('MIFS', 'FontSize', 10)
% saveas(h3, strcat(dataset, '_MIFS'), 'eps2c')
% 
% h4 = figure;
% Y = tsne(train_SP_cmim(:, 1:f));
% gscatter(Y(:,1), Y(:,2), dictionary(train_SP_cmim_label)');
% xlabel('Dim 1', 'FontSize', 10)
% ylabel('Dim 2', 'FontSize', 10)
% title('CMIM', 'FontSize', 10)
% saveas(h4, strcat(dataset, '_CMIM'), 'eps2c')
% 
% h5 = figure;
% Y = tsne(train_SP_icap(:, 1:f));
% gscatter(Y(:,1), Y(:,2), dictionary(train_SP_icap_label)');
% xlabel('Dim 1', 'FontSize', 10)
% ylabel('Dim 2', 'FontSize', 10)
% title('ICAP', 'FontSize', 10)
% saveas(h5, strcat(dataset, '_ICAP'), 'eps2c')
% 
% h6 = figure;
% Y = tsne(train_SP_disr(:, 1:f));
% gscatter(Y(:,1), Y(:,2), dictionary(train_SP_disr_label)');
% xlabel('Dim 1', 'FontSize', 10)
% ylabel('Dim 2', 'FontSize', 10)
% title('DISR', 'FontSize', 10)
% saveas(h6, strcat(dataset, '_DISR'), 'eps2c')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we plot the effects of MIP and DMIP algorithms
F = randi([1, 5], 1000, 1);
Fp = randi([1, 5], 1000, 1);
Y = randi([1, 5], 1000, 1);

mip = [];
baseline = mi(F,Y) * ones(491, 1)';
dmip = [];

for rho = 10:500
    %[F_critical, Y_critical] = OneMiOpt(F, Y, rho);
    %mip = [mip mi(F_critical, Y_critical)];
    
    [Fp_critical, F_critical, Y_critical] = TwoMiOpt(Fp, F, Y, rho);
    dmip = [dmip; [mi(Fp_critical, Y_critical) mi(F_critical, Y_critical)]];
end

% h1 = figure;
% hold on
% grid on
% plot(10:500, mip, '-o', 'color', 'b',  'lineWidth', 1.5, 'MarkerSize', 2);
% plot(10:500, baseline, '-o', 'color', 'g', 'lineWidth', 1.5, 'MarkerSize', 2);
% xlabel('$\rho$', 'Interpreter', 'latex',  'FontSize', 30);
% ylabel('I(F;Y)', 'Interpreter', 'latex', 'FontSize', 30);
% lgd = legend('MIP', 'Baseline', 'location','best');
% set(lgd, 'FontSize', 20)

h2 = figure;
hold on
grid on
plot(10:500, dmip(:, 1)', '-o', 'color', 'b',  'lineWidth', 1.5, 'MarkerSize', 2);
plot(10:500, dmip(:, 2)', '-o', 'color', 'g', 'lineWidth', 1.5, 'MarkerSize', 2);
plot(10:500, dmip(:, 2)' - dmip(:, 1)', '-o', 'color', 'r', 'lineWidth', 1.5, 'MarkerSize', 2);
xlabel('$\rho$', 'Interpreter', 'latex',  'FontSize', 30);
ylabel('$I(F;Y)-I(F_p;Y)$', 'Interpreter', 'latex', 'FontSize', 23);
lgd = legend('I(Fp;Y)', 'I(F;Y)', 'DMIP', 'location','northwest');
set(lgd, 'FontSize', 20)

%saveas(h1, 'OneMiOpt', 'eps2c')
%saveas(h2, 'TwoMiOpt', 'eps2c')





