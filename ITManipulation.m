% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %manipulate the mutual information by injecting malicious data points
addpath /home/hengl/matlab/bin/scripts/ITAFS/source/
index = 4;
limit = 0.2; %maximal percent of points we want to inject

fid = fopen('/home/hengl/matlab/bin/scripts/ITAFS/names2');
file_name = {};
for i=1:30
    line = fgets(fid);
    if ischar(line)
        file_name{i} = line;
    end
end
expression = '[A-Za-z0-9-_]*';
for i=1:30
    str = regexp(file_name{i}, expression);
    file_name{i} = file_name{i}(1: str(1,2)-2);
end
path = strcat('data/', file_name{index}, '.mat');
file = load(path); 
data = struct2cell(file);
data = data{1};
n = size(data,1);
f = size(data, 2) - 1;
y = data(:, size(data, 2));
points = ceil(n*limit);

%perform the poisoning
F1 = data(:, randi(f));
F2 = data(:, randi(f));

%feature preprocessing: observation substitution
observation = 1;
F1_set = unique(F1);
for p = F1_set'
    F1(F1 == p) = observation*ones(size(F1(F1 == p), 1),1);
    observation = observation + 1;
end
observation = 1;
F2_set = unique(F2);
for p = F2_set'
    F2(F2 == p) = observation*ones(size(F2(F2 == p), 1),1);
    observation = observation + 1;
end

%calculate the marginal and joint probs
marginal_F1 = zeros(1, size(unique(F1),1));
for p = 1:size(unique(F1),1)
    marginal_F1(p) = size(F1(F1 == p),1)/n;
end
marginal_F2 = zeros(1, size(unique(F2),1));
for p = 1:size(unique(F2),1)
    marginal_F2(p) = size(F2(F2 == p),1)/n;
end
joints = zeros(size(unique(F1),1), size(unique(F2),1));
for p = 1:n
    joints(F1(p), F2(p)) = joints(F1(p), F2(p)) + 1;
end
joints = joints/n;

%find the solution of optimized poisoning method
maximum = -10000;
F1_critical = 0;
F2_critical = 0;
benchmark = mi(F1, F2);
intial_direction = zeros(size(unique(F1),1), size(unique(F2),1));
for p = 1:size(unique(F1),1)
    for q = 1:size(unique(F2),1)
        if(joints(p,q) ~= 0)
            intial_direction(p,q) = benchmark + log(marginal_F1(p)*marginal_F2(q)/joints(p,q))/log(2);
            if maximum < intial_direction(p,q)
                maximum = intial_direction(p,q);
                F1_critical = p;
                F2_critical = q;
            end
        end
    end
end
stop_criteria = maximum/2;

delta = (1-joints(F1_critical, F2_critical)) - (1-marginal_F1(F1_critical)) - (1-marginal_F2(F2_critical));
beta = (h(F1)+marginal_F1(F1_critical)*log(marginal_F1(F1_critical))/log(2)) + (h(F2)+marginal_F2(F2_critical)*log(marginal_F2(F2_critical))/log(2));
beta = beta - (h([F1,F2]) + joints(F1_critical,F2_critical)*log(joints(F1_critical,F2_critical))/log(2));

%appended the solutions using greedy batch
greedybatch = [F1 F2];
greedybatch_score = [benchmark];
mar_F1 = marginal_F1;
mar_F2 = marginal_F2;
joints_prob = joints;   
current_size = n;

for v = 1:points

    if maximum > stop_criteria
        greedybatch = [greedybatch; [F1_critical F2_critical]];
        greedybatch_score = [greedybatch_score mi(greedybatch(:,1), greedybatch(:,2))];
    end
    lambda = current_size/(n+v);
    
    %three parts depending on lambda to calculate Phi'(lambda) function
    part1 = (1-mar_F1(F1_critical))*log(lambda*mar_F1(F1_critical)+1-lambda)/log(2);
    part2 = (1-mar_F2(F2_critical))*log(lambda*mar_F2(F2_critical)+1-lambda)/log(2);
    part3 = (joints_prob(F1_critical,F2_critical)-1)*log(lambda*joints_prob(F1_critical,F2_critical)+1-lambda)/log(2); 
    maximum = delta*log(lambda)/log(2) + beta + part1 + part2 + part3;
    
    %we use sigma to prevent decreasing too slow: i.e.,maximum < sigma
    if maximum < stop_criteria %means need to update critical values of tow vectors
        %we update the distributions first and find the criticals
        maximum = -10000;
        for s = 1:size(unique(F1),1)
            if s == F1_critical
                mar_F1(s) = lambda*mar_F1(s)+1-lambda;
            else
                mar_F1(s) = lambda*mar_F1(s);
            end
        end
        for r = 1:size(unique(F2),1)
            if r == F2_critical
                mar_F2(r) = lambda*mar_F2(r)+1-lambda;
            else
                mar_F2(r) = lambda*mar_F2(r);
            end
        end
        temp = 0;
        temp_critical1 = 0;
        temp_critical2 = 0;
        for s = 1:size(unique(F1),1)
            for r = 1:size(unique(F2),1)
                if (s == F1_critical) & (r == F2_critical)
                    joints_prob(s,r) = lambda*joints_prob(s,r)+1-lambda;
                else
                    joints_prob(s,r) = lambda*joints_prob(s,r);
                end
                if joints_prob(s,r) ~= 0
                    temp = mi(greedybatch(:,1), greedybatch(:,2)) - log(joints_prob(s,r)/(mar_F1(s)*mar_F2(r)))/log(2);
                    if maximum < temp
                        maximum = temp;
                        temp_critical1 = s;
                        temp_critical2 = r;
                    end
                end
            end
        end
        F1_critical = temp_critical1;
        F2_critical = temp_critical2;
        delta = (1-joints_prob(F1_critical,F2_critical)) - (1-mar_F1(F1_critical)) - (1-mar_F2(F2_critical));
        beta = (h(greedybatch(:,1))+mar_F1(F1_critical)*log(mar_F1(F1_critical))/log(2)) + (h(greedybatch(:,2))+mar_F2(F2_critical)*log(mar_F2(F2_critical))/log(2));
        beta = beta - (h([greedybatch(:,1),greedybatch(:,2)]) + joints_prob(F1_critical,F2_critical)*log(joints_prob(F1_critical,F2_critical))/log(2));
        
        current_size = n+v;
        stop_criteria = maximum/2;
    end
end

figure
plot(greedybatch_score);
hold on
plot(benchmark*ones(1,1+points));
legend('SP', 'benchmark', 'Location', 'best');
xlabel('% points interpolated')
ylabel('mutual information')
title('MI updating w.r.t. interpolation')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%manipulate the joint mutual information by injecting data samples: version0

% addpath /home/hengl/matlab/bin/scripts/ITAFS/source/
% 
% index = 4;
% fid = fopen('/home/hengl/matlab/bin/scripts/ITAFS/names2');
% file_name = {};
% for i=1:30
%     line = fgets(fid);
%     if ischar(line)
%         file_name{i} = line;
%     end
% end
% expression = '[A-Za-z0-9-_]*';
% for i=1:30
%     str = regexp(file_name{i}, expression);
%     file_name{i} = file_name{i}(1: str(1,2)-2);
% end
% path = strcat('data/', file_name{index}, '.mat');
% file = load(path); 
% data = struct2cell(file);
% data = data{1};
% n = size(data,1);
% f = size(data, 2) - 1;
% %maximal percent of points we want to inject
% maxloop = 10;
% sigma = 0.5;
% limit = 0.5; 
% points = ceil(n*limit);
% %perform the poisoning
% 
% F1 = data(:, randi(f));
% F2 = data(:, randi(f));
% Y = data(:, f+1);
% 
% %feature preprocessing: observation substitution
% observation = 1;  %Relabeling F1
% F1_set = unique(F1);
% for p = F1_set'
%     if p <= 0
%         F1(F1 == p) = (observation + max(F1_set))*ones(size(F1(F1 == p), 1),1);
%         observation = observation + 1;
%     end
% end
% observation = 1;
% F1_set = unique(F1);
% for p = F1_set'
%     F1(F1 == p) = observation*ones(size(F1(F1 == p), 1),1);
%     observation = observation + 1;
% end
% observation = 1;   %Relabeling F2
% F2_set = unique(F2);
% for p = F2_set'
%     if p <= 0
%         F2(F2 == p) = (observation + max(F2_set))*ones(size(F2(F2 == p), 1),1);
%         observation = observation + 1;
%     end
% end
% observation = 1;
% F2_set = unique(F2);
% for p = F2_set'
%     F2(F2 == p) = observation*ones(size(F2(F2 == p), 1),1);
%     observation = observation + 1;
% end
% observation = 1;   %Relabeling Y
% Y_set = unique(Y);
% for p = Y_set'
%     if p <= 0
%         Y(Y == p) = (observation + max(Y_set))*ones(size(Y(Y == p), 1),1);
%         observation = observation + 1;
%     end
% end
% observation = 1;
% Y_set = unique(Y);
% for p = Y_set'
%     Y(Y == p) = observation*ones(size(Y(Y == p), 1),1);
%     observation = observation + 1;
% end
% 
% %calculate the marginal and joint probs
% marginal_Y = zeros(1, size(unique(Y),1));  %distribution of class label
% for p = 1:size(unique(Y),1)
%     marginal_Y(p) = size(Y(Y == p),1)/n;
% end
% joints = zeros(size(unique(F1),1), size(unique(F2),1));  %joint distribution of (F1,F2)
% trio = zeros(size(unique(F1),1), size(unique(F2),1), size(unique(Y),1)); %3-d distribution of (F1,F2,Y)
% for p = 1:n
%     joints(F1(p), F2(p)) = joints(F1(p), F2(p)) + 1;
%     trio(F1(p), F2(p), Y(p)) = trio(F1(p), F2(p), Y(p)) + 1;
% end
% joints = joints/n;
% trio = trio/n;
% %sort the probabilities, get the available support
% marginal_Y_ascend = sort(marginal_Y, 'ascend');
% joints_ascend = joints(:);
% joints_ascend = joints_ascend';
% joints_ascend = sort(joints_ascend(joints_ascend > 0), 'ascend');
% trio_ascend = trio(:);
% trio_ascend = trio_ascend';
% trio_ascend = sort(trio_ascend(trio_ascend > 0), 'ascend');
% 
% %appended the solutions using greedy batch
% benchmark = mi(joint([F1,F2],[]), Y);
% current_size = n;
% while current_size < n + points    %we let theta1 be P(y), theta2 be P(f1,f2), theta3 be P(f1,f2,y)
% 
%     %all variables and parameter initialized 
%     lambda = current_size/(current_size + 1);
%     theta1 = 0;
%     theta2 = 0;
%     theta3 = 0;
%     while theta1 == 0 || theta2 == 0 || theta3 == 0
%         F1_critical = randi(size(unique(F1),1));
%         F2_critical = randi(size(unique(F2),1));
%         Y_critical = randi(size(unique(Y),1));
%         theta1 = marginal_Y(Y_critical);
%         theta2 = joints(F1_critical, F2_critical);
%         theta3 = trio(F1_critical, F2_critical, Y_critical);
%     end
%     temp = find(marginal_Y_ascend == theta1);
%     indices(2) = temp(1);            
%     temp = find(joints_ascend == theta2);
%     indices(3) = temp(1); 
%     temp = find(trio_ascend == theta3);
%     indices(4) = temp(1);     
%     
%     %Gradient Descent: be carefully here, we vdon't have the full landscape here, thus we find the best combination of variables
%     beforeupdate = mi(joint([F1,F2],[]), Y);  %objective using parameter before updated
%     afterupdate = mi(joint([F1,F2],[]), Y);  %objective using parameter afetr update, we set it to zero to enter while loop
%     loop = 1;
%     while afterupdate <= beforeupdate && loop < maxloop
%         beforeupdate = afterupdate;
%         %find the valid support for p(y), p(f1,f2) and p(f1,f2,y) based on gradients, also consider moveable or not based on current position
%         gradient1 = lambda*log((lambda*theta1)/(lambda*theta1 + 1 - lambda))/log(2);
%         gradient2 = lambda*log((lambda*theta2)/(lambda*theta2 + 1 - lambda))/log(2);
%         gradient3 = lambda*log((lambda*theta3 + 1 - lambda)/(lambda*theta3))/log(2);
%         flag = 0;
%         if abs(gradient3) >= abs(gradient2) && abs(gradient3) >= abs(gradient1) %theta3 has maximum gradient mag and decide updates
%             if  gradient3 < 0 && indices(4) < size(trio_ascend,2)   %find moving region of prmimary parameter
%                 theta3_support = indices(4)+1:size(trio_ascend,2);
%             elseif gradient3 > 0 && indices(4) > 1
%                 theta3_support = 1:indices(4)-1;
%                 theta3_support = fliplr(theta3_support);
%             else
%                 theta3_support = indices(4);
%             end            
%             for i = theta3_support
%                 shift = abs(trio_ascend(i) - trio_ascend(indices(4)));
%                 [d1,d2,d3] = ind2sub(size(trio), find(trio == trio_ascend(i))); %corresponding (theta1, theta2, theta3)'s
%                 for j = 1:size(d1,1)
%                     if marginal_Y(d3(j)) > 0 && joints(d1(j), d2(j)) > 0  %nonzero probabilities
%                         if abs(marginal_Y(d3(j)) - theta1) < shift && abs(joints(d1(j), d2(j)) - theta2) < shift % these shifts smaller than shift in deciding direction
%                             indices(4) = i;
%                             temp = find(joints_ascend == joints(d1(j), d2(j)));
%                             indices(3) = temp(1);
%                             temp = find(marginal_Y_ascend == marginal_Y(d3(j)));
%                             indices(2) = temp(1);
%                             F1_critical = d1(j);
%                             F2_critical = d2(j);
%                             Y_critical = d3(j);
%                             flag = 1;
%                             break;
%                         end
%                     end
%                 end
%                 if flag == 1
%                     break;
%                 end                
%             end
%         elseif abs(gradient2) >= abs(gradient1) && abs(gradient2) >= abs(gradient3) %theta2 has maximum gradient mag and decide updates
%             if  gradient2 < 0 && indices(3) < size(joints_ascend,2)  %find moving region of prmimary parameter
%                 theta2_support = indices(3)+1:size(joints_ascend,2);
%             elseif gradient2 > 0 && indices(3) > 1
%                 theta2_support = 1:indices(3)-1;
%                 theta2_support = fliplr(theta2_support);
%             else
%                 theta2_support = indices(3);
%             end            
%             for i = theta2_support
%                 shift = abs(joints_ascend(i) - joints_ascend(indices(3)));
%                 [d1,d2] = find(joints == joints_ascend(i));
%                 for j = 1:size(d1,1)
%                     for k = 1:size(unique(Y),1)
%                         if marginal_Y(k) > 0 && trio(d1(j), d2(j), k) > 0
%                             if abs(marginal_Y(k) - theta1) < shift && abs(trio(d1(j), d2(j), k) - theta3) < shift
%                                 temp = find(trio_ascend == trio(d1(j), d2(j), k));
%                                 indices(4) = temp(1);
%                                 indices(3) = i;
%                                 temp = find(marginal_Y_ascend == marginal_Y(k));
%                                 indices(2) = temp(1);
%                                 F1_critical = d1(j);
%                                 F2_critical = d2(j);
%                                 Y_critical = k;
%                                 flag = 1;
%                                 break;
%                             end
%                         end
%                     end
%                     if flag == 1
%                         break;
%                     end
%                 end
%                 if flag == 1
%                     break;
%                 end
%             end     
%         else  %theta1 has maximum gradient mag and decide updates
%             if  gradient1 < 0 && indices(2) < size(marginal_Y_ascend,2) %increase
%                 theta1_support = indices(2)+1:size(marginal_Y_ascend,2);
%             elseif gradient1 > 0 && indices(2) > 1 %decrease
%                 theta1_support = 1:indices(2)-1;
%                 theta1_support = fliplr(theta1_support);
%             else  %stay put
%                 theta1_support = indices(2);
%             end            
%             for i = theta1_support
%                 shift = abs(marginal_Y_ascend(i) - marginal_Y_ascend(indices(2)));
%                 d3 = find(marginal_Y == marginal_Y_ascend(i));
%                 for j = d3
%                     for k1 = 1:size(unique(F1),1)
%                         for k2 = 1:size(unique(F2),1)
%                             if joints(k1,k2) > 0 && trio(k1,k2,j) > 0
%                                 if abs(joints(k1,k2) - theta2) < shift && abs(trio(k1,k2,j) - theta3) < shift
%                                     temp = find(trio_ascend == trio(k1,k2,j));
%                                     indices(4) = temp(1);
%                                     temp = find(joints_ascend == joints(k1,k2));
%                                     indices(3) = temp(1);
%                                     indices(2) = i;
%                                     F1_critical = k1;
%                                     F2_critical = k2;
%                                     Y_critical = j;
%                                     flag = 1;
%                                     break;
%                                 end
%                             end
%                         end
%                         if flag == 1
%                             break;
%                         end
%                     end
%                     if flag == 1
%                         break;
%                     end
%                 end
%                 if flag == 1
%                     break;
%                 end
%             end
%         end
%         %update the variables
%         theta1 = marginal_Y_ascend(indices(2));
%         theta2 = joints_ascend(indices(3));
%         theta3 = trio_ascend(indices(4));
%         
%         F1_temp = [F1; F1_critical];
%         F2_temp = [F2; F2_critical];
%         Y_temp = [Y; Y_critical];
%         afterupdate = mi(joint([F1_temp,F2_temp],[]), Y_temp);
% 
%         if flag == 0
%             loop = 10;
%         else
%             loop = loop + 1;
%         end
%     end
%     
%     gradient0 = (theta1 + theta2 - theta3 + 1)*log(1)/log(2) + (h(Y) + theta1*log(theta1)/log(2) + h(joint([F1,F2],[])) + theta2*log(theta2)/log(2) - h(joint([joint([F1,F2],[]),Y],[])) - theta3*log(theta3)/log(2)) - (theta1 - 1)*log(1*theta1 + 1 - 1)/log(2) - (theta2 - 1)*log(1*theta2 + 1 - 1)/log(2) + (theta3 - 1)*log(1*theta3 + 1 - 1)/log(2);
%     for toappend = 1:(n + points - current_size)
%         lambda_temp = current_size/(current_size + toappend);
%         temp = (theta1 + theta2 - theta3 + 1)*log(lambda_temp)/log(2) + (h(Y) + theta1*log(theta1)/log(2) + h(joint([F1,F2],[])) + theta2*log(theta2)/log(2) - h(joint([joint([F1,F2],[]),Y],[])) - theta3*log(theta3)/log(2)) - (theta1 - 1)*log(lambda_temp*theta1 + 1 - lambda_temp)/log(2) - (theta2 - 1)*log(lambda_temp*theta2 + 1 - lambda_temp)/log(2) + (theta3 - 1)*log(lambda_temp*theta3 + 1 - lambda_temp)/log(2);
%         if temp < 0 || temp < abs(gradient0*sigma) 
%             break;
%         end
%     end
%     lambda = current_size/(current_size + toappend);
%     current_size = current_size + toappend; % update the length of vectors
%     F1 = [F1; F1_critical*ones(toappend,1)];
%     F2 = [F2; F2_critical*ones(toappend,1)];
%     Y = [Y; Y_critical*ones(toappend,1)];
%     benchmark = [benchmark mi(joint([F1,F2],[]), Y)];
% 
%     %we update the distributions based on above lambda and critical outcomes
%     for s = 1:size(unique(Y),1)
%         if s == Y_critical
%             marginal_Y(s) = lambda*marginal_Y(s)+1-lambda;
%         else
%             marginal_Y(s) = lambda*marginal_Y(s);
%         end
%     end
%     for s = 1:size(unique(F1),1)
%         for r = 1:size(unique(F2),1)
%             if (s == F1_critical) && (r == F2_critical)
%                 joints(s,r) = lambda*joints(s,r)+1-lambda;
%             else
%                 joints(s,r) = lambda*joints(s,r);
%             end
%         end
%     end
%     for s = 1:size(unique(F1),1)
%         for r = 1:size(unique(F2),1)
%             for v = 1:size(unique(Y),1)
%                 if (s == F1_critical) && (r == F2_critical) && (v == Y_critical)
%                     trio(s,r,v) = lambda*trio(s,r,v)+1-lambda;
%                 else
%                     trio(s,r,v) = lambda*trio(s,r,v);
%                 end
%             end
%         end
%     end
%     %sort the probabilities, get the available support
%     marginal_Y_ascend = sort(marginal_Y, 'ascend');
%     joints_ascend = joints(:);
%     joints_ascend = joints_ascend';
%     joints_ascend = sort(joints_ascend(joints_ascend > 0), 'ascend');
%     trio_ascend = trio(:);
%     trio_ascend = trio_ascend';
%     trio_ascend = sort(trio_ascend(trio_ascend > 0), 'ascend');
% end
% 
% plot(benchmark)
