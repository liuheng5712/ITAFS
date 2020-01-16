addpath /home/hengl/matlab/bin/scripts/ITAFS/source/
addpath /home/hengl/matlab/bin/MIToolbox/
addpath /home/hengl/matlab/bin/FEAST/
addpath /home/hengl/matlab/bin/scripts/ITAFS/

Parallelism = 3;
rounds = 4;
data_index = 33; %20,13,14,16,17,22,25,29,31,34,32,33,18,23,28,19,26
Toselect = 0.15;
TrTeRatio = 0.75;

parpool(Parallelism); %start the parallel pool and distributing the feature space
poolobj = gcp;
addAttachedFiles(poolobj,{'mi.m', 'feast.m', 'FSToolboxMex.mexa64', 'JonPoisingGenr.m', 'TwoMiOpt.m'});

parfor i = 31:38
   AttackAlternative(data_index, i, Toselect, TrTeRatio);
end

delete(gcp('nocreate'));
