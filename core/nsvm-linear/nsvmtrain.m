function out = nsvmtrain(y, x, varargin)
%NSVMTRAIN Non-negative Support Vector Machine
%   model    = NSVMTRAIN(labels, data, options)
%   accuracy = NSVMTRAIN(labels, data, options) if options.cv > 0
%   perf     = NSVMTRAIN(labels, data, options) if options.cv > 0 and
%   options.perf exists
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - options: structure with SVM options
%       - alg: (default: l1dual)
%           - l1l2primal: L1-regularized L2-loss non-negative support vector classification (active-set in the primal)
%           - l1dual: L1-regularized L1-loss non-negative support vector classification (dual)
%           - biasl1dual: L1-regularized (weights and bias) L1-loss non-negative support vector classification (dual)
%           - biasl1partdual: L1-regularized (weights and bias) L1-loss partition (non-negative, non-positive) support vector classification (dual)
%           - biasl1dualAS: biasl1dual with an active set strategy
%           - biasl1svm: L1-regularized (weights and bias) L1-loss support vector classification (dual)
%           - weightbiasl1svm: L1-regularized (weights and bias, with bounded average weight) L1-loss support vector classification (dual)
%           - penweightbiasl1svm: L1-regularized (weights and bias, with a penalization on negative weights) L1-loss support vector classification (dual)
%           - penweightbiasl1svmAS: penweightbiasl1svm with an active set strategy
%           - weightbiasl1l2svm: Elastic-net regularized L2-loss SVM
%       - C: cost parameter (default: 1)
%       - b_tol: minimum bias that is authorized (only for l1l2primal and
%       l1dual, default: 1e-1)
%       - cv: number of folds in the cross-validation. If 0, no
%       cross-validation (default: 0)
%       - perf: lambda function to compute a performance measure. For
%       instance: @(scores, labels) mean(sign(scores) == labels)
%       - verbose: verbosity (default: false)
% 
%       Options particular to biasl1dualAS and penweightbiasl1svmAS:
%       - n_var_max: number of variables above which the active set
%       strategy is used (default: 50)
%       - n_var_add: number of variables to add at each active set
%       iteration (default: 10)
%       - n_it_max_out: maximum number of active set iterations  (default: 500)
%       - n_it_check: number of iterations to which we check if the active
%       set stratey works well (default: 10)
%       - act_set_tol: tolerance on the otpimility condition of the active
%       set algorithm (default: 1e-5)
%
%       Options particular to l1l2primal (see nsvml1l2primalactivesettrain
%       for default values):
%       - lambda: tradeoff of the elastic net regularization
%       - n_var_max: number of variables above which the active set
%       strategy is used
%       - n_var_add: number of variables to add at each active set
%       iteration
%       - n_it_max: maximum number of iterations in the inner loop 
%       - n_it_max_out: maximum number of iterations in the outer loop 
%       - beta: evolution of the gradient step t <- beta*t 
%       - step: first step 
%       - w: starting weight vector 
%       - b: starting bias 
%       - stop_criterion: stopping criterion 
%           - grad_rel: |grad| < grad_tol * |grad_init|
%           - grad_inf: max(|grad|) < grad_tol
%           - grad_2:   ||grad||_2 < grad_tol
%       - grad_tol: tolerance on zero gradient for the stopping criterion
%       - act_set_tol: tolerance on the first otpimility condition of the
%       active set algorithm
%
%       Options particular to biasl1dual and biasl1svm, weightbiasl1svm,
%       penweightbiasl1svm:
%       - lambda: tradeoff of the regularization (1-lambda)*||w||_1 + 
%       lambda*||b||_1 (default: 0)
%       - norm_weights: weights of the 1-norm (default: ones(dim data, 1))
%
%       Options particular to penweightbiasl1svm:
%       - D: penalization parameter D*max(0, -w) (default: 0)
%
%       Options particular to weightbiasl1svm:
%       - weight_bound: lower bound of the average weight weight_bound <= sum(w) (default: 0)
%
%       Options particular to weightbiasl1l2svm:
%       - lambdaEN: tradeoff of the elastic net regularization (1-lambdaEN)*||.||_1
%       + lambdaEN/2*||.||_2^2 (default: 0 -> 1-norm SVM)
%       - norm_weights: weights of the 1-norm (default: ones(dim data, 1))
%       - lambda: tradeoff of the regularization on the bias
%       (1-lambda)*||w||_elastic-net + lambda*||b||_elastic-net (default: 0)
%       - pos_cons: add the constraint w >= 0 if true (default: false)
%       - bound_cons: add the constraint sum(cons_weights .* w) >= B if
%       true (default: false)
%       - cons_weights: weights of the previous constraint (default:
%       ones(dim data, 1))
%       - B: bound of the previous constraint (default: 0)
%       - obj_tol: stop when (obj_t - obj_{t-1}) < obj_tol (default: 1e-7)
%       - n_it_max: maximum number of iterations (default: 5000)
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%   if options.cv = 0 or the accuracy if options.cv > 0
%
% See also nsvmpredict, nsvml1dualtrain,
% nsvml1l2primaltrain

% Maxime Sangnier (CEA)
% Revision: 0.1     28-Mar-2014
%           0.2     14-Nov-2014
%           0.3     17-Nov-2014

% Default values
options = check_options(check_argin(varargin, struct), ...
    'alg', 'l1dual', ... % Algorithm to use
    'C', 1, ... % Cost parameter
    'b_tol', 1e-1, ... % Minimum bias that is authorized
    'cv', 0, ... % Cross-validation
    'lambda', 0, ... % Tradeoff of the elastic net regularization
    'norm_weights', ones(size(x, 2), 1), ... % Weights of the 1-norm
    'ind_neg', []); % Indexes of non-positive weights

% Algorithm to use
switch (options.alg)
    case 'l1dual'
        lambda_train = @(y, x, options) ...
            nsvml1dualtrain(y, x, options.C, options.b_tol);
    case 'biasl1dual'
        lambda_train = @(y, x, options) ...
            nsvmbl1dualtrain(y, x, options.C, options.lambda, ...
            options.norm_weights);
    case 'biasl1partdual'
        lambda_train = @(y, x, options) ...
            nsvmbl1partdualtrain(y, x, options.C, options.lambda, ...
            options.norm_weights, options.ind_neg);
    case 'biasl1dualAS'
        lambda_train = @(y, x, options) ...
            nsvmbl1dualactivesettrain(y, x, options.C, options.lambda, ...
            options);
    case 'biasl1svm'
        lambda_train = @(y, x, options) ...
            svmbl1dualtrain(y, x, options.C, options.lambda);
    case 'weightbiasl1svm'
        lambda_train = @(y, x, options) ...
            svmbl1wdualtrain(y, x, options.C, options.lambda, ...
            options.weight_bound, options.norm_weights);
    case 'penweightbiasl1svm'
        lambda_train = @(y, x, options) ...
            svmbl1wpdualtrain(y, x, options.C, options.lambda, ...
            options.D, options.norm_weights);
    case 'penweightbiasl1svmAS'
        lambda_train = @(y, x, options) ...
            nsvmbl1wpdualactivesettrain(y, x, options.C, options.lambda, ...
            options.D, options);
    case 'l1l2primal'
        lambda_train = @(y, x, options) ...
            nsvml1l2primalactivesettrain(y, x, options.C, ...
            options.b_tol, options.lambda, options);
    case 'weightbiasl1l2svm'
        options.A = options.lambda;
        lambda_train = @(y, x, options) ...
            svmbl1l2wprimaltrainwrapper(y, x, options.C, options.lambdaEN, ...
            options);
    otherwise
        error('Unknown algorithm. Use either l1l2primal or l1dual.');
end

% No cross-validation
if (options.cv < 1)
    % Count the number of classes
    lbl_range = min(y):max(y); % Range of potential labels
%     lbl = lbl_range(find(histc(y, lbl_range))); % Existing labels
    lbl = lbl_range(histc(y, lbl_range) > 0); % Existing labels
    
    % Some info
    nr_class = numel(lbl); % Number of classes
    [N, nr_feature] = size(x); % Number of instances x number of features
    
    % If this is a two class problem
    if (nr_class < 3)
        % Build the model by solving the optimization problem
%         out = nsvml1dualtrain(y, x, options.C, options.b_tol);
        out = lambda_train(y, x, options);
    % If there are more than 2 classes
    else
        % Matrix with all SVM weights
        w = zeros(nr_class, nr_feature+1);
        
        % Build one-vs-rest models
        for ic = 1:nr_class % For each class
            % Build the new labels
            new_y = -ones(N, 1);
            new_y(y == lbl(ic)) = 1;
            % Solve the SVM problem
%             model = nsvml1dualtrain(new_y, x, options.C, options.b_tol);
            model = lambda_train(new_y, x, options);
            % Save the weight vector
            w(ic, :) = model.w;
        end
        
        % Build the model
        out = struct;
%         out.Parameters = model.Parameters;
        out.nr_class = nr_class;
        out.nr_feature = nr_feature;
        out.bias = model.bias;
        out.Label = lbl';
        out.w = sparse(w);
    end

% Cross-validation
else
    % Info
%     [N, n] = size(x); % Number of instances x length of an instance
    N = size(x, 1); % Number of instances
    Ncv = floor(N / options.cv); % Number of instances in a fold
    cva = zeros(options.cv, 1); % CV accuracies
    seed = rng; rng(1); % Reproducible randomness
    ind = randperm(N); % Indexes of the instances randomly permuted
    rng(seed); clear seed; % Restore the generator settings
    
    % Copy options structure without cv
    opt = options;
    opt = rmfield(opt, 'cv');
    
    % Do the cross-validation
    for it = 1:options.cv
        ind_eval = ind((it-1)*Ncv+1 : it*Ncv); % Indexes of the validation instances
        ind_train = compl(ind_eval, N); % Indexes of the training instances
        
        % Training
%         model = nsvml1dualtrain(y(ind_train), x(ind_train, :), ...
%             options.C, options.b_tol);
        model = nsvmtrain(y(ind_train), x(ind_train, :), opt);
        % Evaluation
%         [~, lib_acc, ~] = predict(y(ind_eval), sparse(x(ind_eval, :)), ...
%             model, libsvm_options('verbose', false));
        [~, lib_acc, scores] = nsvmpredict(y(ind_eval), ...
            x(ind_eval, :), model);
        if (isfield(options, 'perf') && ~isempty(options.perf))
            cva(it) = options.perf(scores, y(ind_eval));
        else
        	cva(it) = lib_acc(1);
        end
    end
    
    % Average of the accuracies
    out = mean(cva);
end
end

% Give the compatary set of lower_set in 1:upper_bound
function compl_set = compl(lower_set, upper_bound)
    compl_set = 1:upper_bound;
    compl_set(lower_set) = [];
end
