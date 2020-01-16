% in this script we use greedy injection to calculate the required injection 


index = 23;
dataname = 'pengcolon';
TrTeRatio = 0.75; 

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
oldn = size(data,1);
TrTe = ceil(TrTeRatio * oldn);
rounds = 20;

Expense = [];
for r = 1:rounds
    
    data = data(randi([1,oldn],1,oldn),:);
    train_data = data(1:TrTe ,:);  % here we don't need the test data
    y = train_data(:, size(train_data, 2));
    n = size(train_data, 1);
    f = size(train_data, 2) - 1;

    reles = [];  
    for i=1:f
        reles = [reles mi(train_data(:,i),y)];
    end
    feature_set = 1:f; %To get rid of the all-zero columns
    [B, I] = sort(reles, 'descend');
    sorted_features = feature_set(I);

    expense = [0];
    if f > 500
        limit = 500;
    else
        limit = f;
    end
    for i = sorted_features(2:limit)
        fakebest = train_data(:, i);
        fakebest_outcomes = unique(fakebest);
        best = train_data(:, sorted_features(1));
        best_outcomes = unique(best);
        y = train_data(:, size(train_data, 2));
        y_outcomes = unique(y);

        cost = 0;
        while mi(fakebest, y) < mi(best, y) && cost <= n
            criticals = [];
            maximum = -10000;
            for i1 = fakebest_outcomes'
                for i2 = best_outcomes'
                    for i3 = y_outcomes'
                        temp = mi([fakebest;i1],[y;i3]) - mi([best;i2],[y;i3]);
                        if temp > maximum
                            maximum = temp;
                            critical = [i1 i2 i3];
                        end
                    end
                end
            end
            cost = cost + 1;
            fakebest = [fakebest; critical(1)];
            best = [best; critical(2)];
            y = [y; critical(3)];        
        end
        expense = [expense cost];
    end
    Expense = [Expense; expense];
end
greedyExpense = Expense;
clearvars -except greedyExpense dataname
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot the comparison of injection number

location = '/home/hengl/matlab/bin/scripts/ITAFS/results/version2/';
load(strcat(location, dataname, '.mat'));

h1 = figure;
box on
hold on
length = size(Expense, 2);
plot(1:length, mean(Expense)/n, 'lineWidth', 2, 'color', 'black')
plot(1:length, mean(greedyExpense)/n, 'lineWidth', 2, 'color', 'red')
ylabel('Injected Samples', 'FontSize', 20)
xlabel('$F_i$', 'Interpreter', 'latex', 'FontSize', 20)
axis tight
lgd = legend('DMIP', 'Greedy', 'location','best');
set(lgd, 'FontSize', 20)

saveas(h1, dataname, 'eps2c')
