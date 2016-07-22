% Early detection (toy dataset)
%
% Maxime Sangnier
% 21-Jul-2016

clear all;

% Libraries
addpath ../core/utils;
addpath ../core/nsvm-linear;
addpath ../core/simpleed;
addpath /home/maxime/matlab/toolboxes/lpsolve/; % Path to LPsolve
addpath(genpath('/home/maxime/matlab/toolboxes/mmed-release-0.1/')); % Path to MMED


%% Toy database
db = load('../data/toy_mfcc_21-Jul-2016');
disp(db);
fprintf('Number of samples: %d\n', size(db.data, 1));
fprintf('Number of variables: %d\n\n', size(db.data, 2));


%% Options
opt = default_opt();
% Set opt.forecast = false not to use the reliable proxy
opt.class_detection = 'chirp1';
opt.C = 2^8; % Trade-off parameter on the cost function
opt.D = 2; % Penalization parameter for the negative weights
opt.nsvm_high_weight = 8; % Highest weight of the L1-norm


%% Preprocess database
[db, opt.labels] = encode_classes(db, opt.class_detection, opt.class_vs);
[db_training, db_test] = create_partitions(db, opt.ratio, 1, ...
    opt.training_balance);
[db_training, opt] = preprocess_training(db_training, opt);


%% Training
model = edtrain(db_training, opt);


%% Preprocess test data
db_test = preprocess_test(db_test, model, opt); % Sparse preprocessing


%% Test
report = eval_earliness(db_test, model, opt);
disp(report);


%% Plot
plot_reports(report);
