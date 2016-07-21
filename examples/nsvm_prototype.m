% Prototype non-negative SVM
%
% Maxime Sangnier
% 27-Jun-2016

clear all;
close all;

% Libraries
addpath ../core/utils;
addpath ../core/nsvm-linear;
addpath /home/maxime/matlab/toolboxes/lpsolve/; % Path to LPsolve


%% Options
% Data parameters
N = 100; % Number of points in each class
p = 2000; % Number of random features

% SVM and active set parameters
options = struct('C', 1, ... % SVM cost parameter
    'D', 3.5, ... % Negative weights penalization
    'lambda', 0.1, ... % Regularization tradeoff ((1-l)*||w||_1 + l*||b||_1)
    'norm_weights', [1; 2], ... % Norm weights
    'n_var_max', 50, ... % Threshold beyond which the active set strategy is not used
    'n_var_add', 10, ... % Number of variable to add at each iteration
    'n_it_max_out', 500, ... % Max number of iterations
    'n_it_check', 10, ... % Number of iterations to which we check if the active set stratey works well
    'act_set_tol', 1e-5, ... % Tolerance on the otpimility condition
    'verbose', true);


%% Toy dataset
x1 = rand(N, 2)*0.5 + kron([0.5, 0.5], ones(N, 1)); % Class 1
x2 = rand(N, 2)*0.5 + kron([0, 0], ones(N, 1)); % Class 2
x1 = rand(N, 2)*0.5 + kron([0, 0.5], ones(N, 1)); % Class 1
x2 = rand(N, 2)*0.5 + kron([0.5, 0], ones(N, 1)); % Class 2
x = [x1; x2]; % Training set
y = [ones(N, 1); -ones(N, 1)]; % Labels
Y = sparse(diag(y)); % Matrix of labels

% Add random features
random_features = rand(2*N, p)*0.2;
random_features(:, 1:2:end) = 0.1;
x = [x, random_features];
options.norm_weights = [options.norm_weights; ones(p, 1)];
n = size(x, 2);


%% Solve the dual problem without and with active set
% Without active set
tic;
options.alg = 'penweightbiasl1svm';
model = nsvmtrain(y, x, options);
training_time = toc;

% With active set algorithm
tic;
options.alg = 'penweightbiasl1svmAS';
model_as = nsvmtrain(y, x, options);
training_time_as = toc;


%% Print a report
fprintf('\n\n========== REPORT (direct vs active set) ==========\n');

disp('Hyperplan normal vector (non-zero components):');
ind = find(abs(model.w(1:end-1)) + abs(model.w(1:end-1)) > 1e-6);
disp(full([model.w(ind)', model_as.w(ind)']));

disp('Hyperplan bias:');
disp(full([model.w(end), model_as.w(end)]));

disp('Training time:');
disp([training_time, training_time_as]);

% Check the prediction
[pred_lbl, acc, pred_val] = nsvmpredict(y, x, model);
[pred_lbl_as, acc_as, pred_val_as] = nsvmpredict(y, x, model_as);
fprintf('Accuracy: %d%%\n\n', floor(acc(1)));
fprintf('Difference between predictions: %0.2f\n', norm(pred_val_as-pred_val));
