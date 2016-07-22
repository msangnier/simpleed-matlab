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


%% Preprocess database
[db, opt.labels] = encode_classes(db, opt.class_detection, opt.class_vs);
[db_training, db_test] = create_partitions(db, opt.ratio, 1, ...
    opt.training_balance);
[db_training, opt] = preprocess_training(db_training, opt);


%% Loop on parameters
weights = [1 2 4 8]; % Highest weights of the L1-norm
penalizations = [0 1 2 Inf]; % Penalization parameters for the negative weights

reports = {};
idx = 1;
for w = weights
    fprintf('NSVM high weight: %0.2f\n', w);
    for pen = penalizations
        fprintf('   Penalization: %0.2f\n', pen);

        % Parameters
        opt.D = pen; % Penalization parameter for the negative weights
        opt.nsvm_high_weight = w; % Highest weight of the L1-norm
        % This step is normally done in preprocess_training
        opt.norm_weights = build_norm_weights(opt.nsvm_low_weight, ...
            opt.nsvm_high_weight, db_training.ind_file, ...
            opt.nsvm_weighting_mode, opt.dec_landmarks);
        
        % Training
        model = edtrain(db_training, opt);
        
        % Preprocess test data
        db_test = preprocess_test(db_test, model, opt); % Sparse preprocessing
        
        % Test model
        reports{idx} = eval_earliness(db_test, model, opt);
        idx = idx + 1;
    end
end


%% MMED
opt.kernel = 'mmed';
[db_training, opt] = preprocess_training(db_training, opt);
model = edtrain(db_training, opt); % Training
db_test = preprocess_test(db_test, model, opt); % Non-sparse preprocessing
reports{idx} = eval_earliness(db_test, model, opt); % Test model


%% Plot
plot_reports(reports);
