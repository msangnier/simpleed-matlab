function opt = default_opt()
    opt = struct;
    opt.verbose = false;
    
    % Detection problem
    opt.class_vs = ''; % All other classes

    % General options
    opt.kernel = 'simfull'; % SVM kernel (or mmed)
    opt.alg = 'penweightbiasl1svmAS'; % Nsvm algorithm (or penweightbiasl1svm)
    opt.forecast = true; % Use the reliable proxy
    opt.C = 1; % Trade-off parameter on the cost function
    opt.D = 0; % Penalization parameter for the negative weights
    opt.gamma = 0.1233; % Kernel parameter for simfull.
    opt.nsvm_high_weight = 1; % Highest weight of the L1-norm
    opt.dec_landmarks = 2; % Decimation training instances -> landmarks

    % Advanced options (preferably do not change)
    opt.nsvm_low_weight = 1; % Lowest weight of the L1-norm
    opt.nsvm_weighting_mode = 'linear'; % How to build the weights of the L1-norm (linear, killer, log)
    opt.n_it_max_out = 5000; % Max number of active set iterations
    opt.n_it_max = []; % Max number of iteration in the inner incremental loop
    opt.n_it_check = 5; % Iteration at which we check if the active set stratefy works
    opt.n_var_max = 100; % Max number of variables in the inner problems (active set strategy)
    opt.n_var_add = 10; % Number of variable to add (active set strategy)
    opt.act_set_tol=1e-5;
    opt.simfun = @dist_matrix_square; % Similarity base function
    opt.incfun = @(x) min(x, [], 1); % Similarity increase function
    opt.incfunstep = @(x, y) min(x, y); % Similarity inscrease (step by step) function
    opt.highsimfun = @(x, p) exp(-p * x); % Similarity final function
    opt.dec_time = true; % True to decrease the time resolution in the constraint
    opt.mmed_train_batch = true; % False to train as in Hoai's paper

    % Random splits
    opt.ratio = 0.5; % Training set ratio (split the database in a training and a test set)
    opt.training_balance = true; % Are the training classes balanced (ie same number of instances in each class) ?

    % Accuracy measure
    opt.perf = @(scores, labels) aucscore(scores, labels, max(labels)); % Performance measure.
end