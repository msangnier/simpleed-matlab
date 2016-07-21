function model = nsvmbl1wpdualactivesettrain(y, x, varargin)
%NSVMBL1WPDUALACTIVESETTRAIN bias-regularized SVM with a penalization on negative
%weights trained in the dual with an active set method.
%   model = NSVMBL1WPDUALACTIVESETTRAIN(labels, data, C, lambda, D, options)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - C: cost parameter (default: 1)
%   - lambda: tradeoff of the regularization (1-lambda)*||w||_1 + 
%   lambda*||b||_1 (default: 0)
%   - D: penalization parameter D*max(0, -w) (default: 0)
%   - options: options of the active set algorithm as a structure that can
%   contain:
%       - norm_weights: weights of the L1-norm (default: ones(dim data, 1))
%       - n_var_max: number of variables above which the active set
%       strategy is used (default: 50)
%       - n_var_add: number of variables to add at each active set
%       iteration (default: 10)
%       - n_it_max_out: maximum number of active set iterations  (default: 500)
%       - n_it_check: number of iterations to which we check if the active
%       set stratey works well (default: 10)
%       - act_set_tol: tolerance on the otpimility condition of the active
%       set algorithm (default: 1e-5)
%       - verbose: verbosity (default: false)
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%
% See also svmbl1wpdualtrain, nsvmtrain.

% Maxime Sangnier (Télécom ParisTech)
% Revision: 0.1
% Date: 2016/01/01

% Default values
[C, l, D, options] = check_argin(varargin, ...
    1, ... % C
    0, ... % l
    0, ... % D
    struct()); % Options
options = check_options(options, ...
    'norm_weights', ones(size(x, 2), 1), ... % Weights of the weighting L1-norm
    'n_var_max', 50, ... % Threshold beyond which the active set strategy is not used
    'n_var_add', 10, ... % Number of variable to add at each iteration
    'n_it_max_out', 500, ... % Max number of iterations
    'n_it_check', 10, ... % Number of iterations to which we check if the active set stratey works well
    'act_set_tol', 1e-5, ... % Tolerance on the otpimility condition
    'verbose', false);

% Check point
maxy = max(y);
miny = min(y);
if ~all(y == maxy | y == miny)
    error('Use only two classes.');
else
    % Save the true labels
    labels = [maxy; miny]; % Labels
    % Use 1 and -1 for the labels
    ytemp = zeros(numel(y), 1);
    ytemp(y == maxy) = 1;
    ytemp(y == miny) = -1;
    y = ytemp;
    clear ytemp;
end
if (size(y, 2) > 1)
    error('Labels should be in a column vector.');
end
if (size(y, 1) ~= size(x, 1))
    error('The number of labels does not match the number of instances.');
end
if (size(options.norm_weights, 2) ~=1 || length(options.norm_weights) ~= size(x, 2))
    error('The size of the 1-norm weighted vector (norm_weights) does not match the data dimension.');
end
if (C <= 0)
    error('The cost parameter must be positive.');
end
if (D < 0)
    error('The penalization parameter must be non-negative.');
end
if (l < 0 || l > 1)
    error('The tradeoff parameter lambda must be between 0 and 1.');
end

% First case: hard penalization (D = infty)
if isinf(D)
    model = nsvmbl1dualactivesettrain(y, x, C, l, options);
    model.Label = labels; % Labels
% Second case: 0 <= D < infty
else
    % Init the LibLINEAR-like model
    model = struct();

    % Shortcut and info
    obj_tol = 1e-3; % Tolerance on the objective function (to check if it improves)
    verb = options.verbose; % Verbosity
    n = size(x, 2); % Number of variables
    n_var_add = min(options.n_var_add, floor(n/2)); % Number of variables to add at each step
    everything_s_allright = true; % The active set strategy works well (until now)

    % Print options
    vprint(verb, 'Parameters:');
    vprint(verb, '                C: %f\n', C);
    vprint(verb, '           lambda: %f\n\n', l);
    vprint(verb, '                D: %f\n\n', D);
    vprint(verb, '     norm_weights: [%2.2f  %2.2f  %2.2f  %2.2f  ...]\n', ...
        full(options.norm_weights(1:4)));
    vprint(verb, 'Active set options:');
    vprint(verb, '        n_var_max: %d\n', options.n_var_max);
    vprint(verb, '        n_var_add: %d\n', n_var_add);
    vprint(verb, '     n_it_max_out: %d\n', options.n_it_max_out);
    vprint(verb, '      act_set_tol: %f\n', options.act_set_tol);
    vprint(verb, '          verbose: %d\n\n', options.verbose);
    vprint(verb, '  it       obj      viol    #active');
    vprint(verb, '-------------------------------------');

    % Decide to use the active set strategy or not
    % If there are few variables
    if (n <= options.n_var_max)
        model = nsvmbl1dualtrain(y, x, C, l, options.norm_weights);
        model.Label = labels; % Labels

    % If there are many variables
    else
        % Matrix labels * data (#dim x #points)
        Q = bsxfun(@times, x.', y.');
        % Initialization: take most violating variables for the initial values
        % of the weights w and of the bias b
        w = sparse(n, 1); % Initial SVM weights
    %     w = zeros(n, 1); % Initial SVM weights
        b = 0; % Initial bias
        const_rhs = (1 - l) * options.norm_weights; % RHS of the constraint to violate
    %     dual = C * y; % SVM dual variables
    %     viol = Q * dual; % LHS of the constraint to violate
    %     [~, viol_ind] = sort(viol, 'descend'); % Sorting in descending order
    %     ind = viol_ind(1:options.n_var_max); % Most violating variables
        ind = randperm(n);
        ind = ind(1:options.n_var_max)'; % Take random variables

        % Active set loop
        objective = zeros(options.n_it_max_out, 1); % Objective value
        for n_it = 1:options.n_it_max_out
            % Solve the small problem
            model = svmbl1wpdualtrain(y, x(:, ind), C, l, D, ...
                options.norm_weights(ind));

            % Check if the problem has been correctly solved
            if ~isfield(model, 'alpha')
                everything_s_allright = false;
                vprint(verb, 'nsvmbl1wpdualactivesettrain does not use the active set strategy because the subproblem has not been solved correctly.\n', n_it);
                break;
            end

            % Get the result back
            w(ind) = model.w(1:model.nr_feature).';
            b = - model.w(end);

            % Get rid of zero weights
            ind(w(ind) == 0) = [];

            % Find the most violating upper constraints
            Qalpha = Q * model.alpha; % Temporary variable
            viol = Qalpha - const_rhs; % Constraint to violate
            [viol_val, viol_ind] = sort(viol, 'descend'); % Sorting in descending orderrder
            n_viol = numel(find(viol > 0)); % Number of violating (ie positive) constraints
            ind = unique([ind; viol_ind(1:min(n_var_add, n_viol))]); % Add the most violating variables
            max_viol = viol_val(1); % Maximum violation of the constraint

            % Find the most violating lower constraints
            viol = -Qalpha - const_rhs - D; % Constraint to violate
            [viol_val, viol_ind] = sort(viol, 'descend'); % Sorting in descending orderrder
            n_viol = numel(find(viol > 0)); % Number of violating (ie positive) constraints
            ind = unique([ind; viol_ind(1:min(n_var_add, n_viol))]); % Add the most violating variables
            max_viol = max(max_viol, viol_val(1)); % Maximum violation of the constraint

            % Objective value
            objective(n_it) = model.obj(end);

            % Check that the active set strategy does improve the objective
            % value
            if(n_it == options.n_it_check && ...
                    abs(objective(1) - objective(n_it)) < obj_tol)
                everything_s_allright = false;
    %             warning('nsvmbl1dualactivesettrain does not use the active set strategy.\n', n_it);
                vprint(verb, 'nsvmbl1wpdualactivesettrain does not use the active set strategy any longer.\n', n_it);
                break;
            end

            % Display
            vprint(verb, '%5d   %7.2f   %6.3f     %3d\n', ...
                n_it, model.obj(end), max_viol, numel(ind));

            % Stopping criterion (if max_viol is non negative)
            if (max_viol <= options.act_set_tol)
                break;
            end
        end
        if (everything_s_allright && n_it == options.n_it_max_out)
            warning('nsvmbl1wpdualactivesettrain did NOT converge after %d iterations.\n', n_it);
        elseif (everything_s_allright)
            vprint(verb, 'nsvmbl1wpdualactivesettrain converged after %d iterations.\n', n_it);
        end

        if (everything_s_allright)
            % Build the LibLINEAR-like model
            model.nr_class = 2; % Number of class
            model.nr_feature = size(x, 2); % Dimension of the instances
            model.bias = 1; % Enable the SVM bias
            model.Label = labels; % Labels
            model.w = [w; -b]'; % Weight vector + SVM bias at the end
            model.obj = objective(1:n_it); % Objective values
        else
            % If the active set strategy does not work, solve the full problem
            model = nsvmbl1dualtrain(y, x, C, l, options.norm_weights);
            model.Label = labels; % Labels
        end
    end
end
end

