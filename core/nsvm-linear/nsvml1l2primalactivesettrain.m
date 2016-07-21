function model = nsvml1l2primalactivesettrain(y, x, varargin)
%NSVML1L2PRIMALACTIVESETTRAIN Non-negative elastic net and L2-loss SVM
%trained in the primal with an active set strategy.
%   model = NSVML1L2PRIMALACTIVESETTRAIN(labels, data, C, min_bias, lambda, options)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - C: cost parameter (default: 1)
%   - min_bias: minimum bias that is authorized (default: 1e-1)
%   - lambda: tradeoff of the elastic net regularization (1-lambda)*||.||_1
%   + lambda*||.||_2^2 (default: 0)
%   - options: options of the gradient algorithm as a structure that can
%   contain:
%       - n_var_max: number of variables above which the active set
%       strategy is used (default: 50)
%       - n_var_add: number of variables to add at each active set
%       iteration (default: 1)
%       - n_it_max: maximum number of iterations in the inner loop (default: 5000)
%       - n_it_max_out: maximum number of iterations in the outer loop (default: 500)
%       - beta: evolution of the gradient step t <- beta*t (default: 0.5)
%       - step: first step (default: 1)
%       - w: starting weight vector (default: zeros(#instances, 1))
%       - b: starting bias (default: 1)
%       - stop_criterion: stopping criterion (default: grad_rel)
%           - grad_rel: |grad| < grad_tol * |grad_init|
%           - grad_inf: max(|grad|) < grad_tol
%           - grad_2:   ||grad||_2 < grad_tol
%       - grad_tol: tolerance on zero gradient for the stopping criterion
%       (default: 1e-5)
%       - act_set_tol: tolerance on the first otpimility condition of the
%       active set algorithm (default: 1e-3)
%       - verbose: verbosity (default: false)
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%
% See also nsvml1l2primaltrain, nsvml1dualtrain, nsvmtrain.

% Maxime Sangnier (CEA)
% Revision: 0.1
% Date: 2014/04/18

% Default values
[C, b_tol, l, options] = check_argin(varargin, ...
    1, ... % C
    1e-1, ... % b_tol
    0, ... % l
    struct()); % Options
options = check_options(options, ...
    'n_var_max', 50, ...
    'n_var_add', 1, ...
    'n_it_max', 5000, ...
    'n_it_max_out', 500, ...
    'beta', 0.5, ...
    'step', 1, ...
    'w', sparse(size(x, 2), 1), ...
    'b', 1, ...
    'stop_criterion', 'grad_rel', ...
    'grad_tol', 1e-5, ...
    'act_set_tol', 1e-3, ...
    'verbose', false);

% Check point
maxy = max(y);
miny = min(y);
if ~all(y == maxy | y == miny)
    error('Use only two classes.');
else
    % Save the true labels
    labels = [maxy; miny]; % Labels
    % Make the two labels being 1 and -1
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
if (C <= 0)
    error('The cost parameter must be positive.');
end
if (b_tol <= 0)
    error('The minimum bias authorized must be positive.');
end
if (l < 0 || l > 1)
    error('The elastic net tradeoff must lie between 0 and 1.');
end

% Shortcut and info
verb = options.verbose; % Verbosity
n_max = options.n_var_max; % Number of variables in the non active set algo
n = size(x, 2); % Number of variables
n_var_add = min(options.n_var_add, floor(n/2)); % Number of variables to add at each step

% Print options
print(verb, 'Parameters:');
print(verb, '                C: %f\n', C);
print(verb, '         min_bias: %f\n', b_tol);
print(verb, '           lambda: %f\n\n', l);
print(verb, 'Gradient options:');
% print(verb, options);
print(verb, '        n_var_max: %d\n', options.n_var_max);
print(verb, '        n_var_add: %d\n', options.n_var_add);
print(verb, '         n_it_max: %d\n', options.n_it_max);
print(verb, '     n_it_max_out: %d\n', options.n_it_max_out);
print(verb, '             beta: %f\n', options.beta);
print(verb, '             step: %f\n', options.step);
if (numel(options.w) > 1)
    print(verb, '                w: [%2.2f  %2.2f  ...]\n', full(options.w(1:2)));
else
    print(verb, '                w: %f\n', full(options.w));
end
print(verb, '                b: %f\n', options.b);
print(verb, '   stop_criterion: %s\n', options.stop_criterion);
print(verb, '         grad_tol: %f\n', options.grad_tol);
print(verb, '      act_set_tol: %f\n', options.act_set_tol);
print(verb, '          verbose: %d\n\n', options.verbose);
print(verb, '  it       obj      viol    #active');
print(verb, '-------------------------------------');

% Loss function and its gradient
% loss = @(u, v) shinge_loss(u, v);
grad_loss = @(u, v) grad_shinge_loss(u, v);

% Determine the stopping critetion function
switch lower(options.stop_criterion)
    case 'grad_rel'
    case 'grad_inf'
    case 'grad_2'
    otherwise
        error('Unknown stopping criterion (see the manual).');
end

% Decide to use the active set strategy or not
% If there are few variables
if (n <= n_max)
    model = nsvml1l2primaltrain(y, x, C, b_tol, l, options);

% If there are many variables
else
    % Initialization: take most violating variables for the initial values
    % of the weights w and of the bias b
    w = sparse(n, 1); % Initial SVM weights
    b = options.b; % Initial bias
    grad_w = grad_obj(y, x, w, b, C, l, grad_loss); % Gradient
    [~, grad_w_ind] = sort(grad_w); % Sorting in ascending order
    ind = grad_w_ind(1:n_max); % Most violating variables
%     max_viol = grad_w(1); % Maximum violation of the constraint
    
    % Disable the verbosity of the inner loop
    options.verbose = false;
    warning('off', 'nsvml1l2primaltrain:earlyStopping');
    
    % Active set loop
    objective = zeros(options.n_it_max_out, 1); % Objective value
    for n_it = 1:options.n_it_max_out
        % Solve the small problem with warm start
        options.w = w(ind); % Weights
        options.b = b; % Bias
        model = nsvml1l2primaltrain(y, x(:, ind), C, b_tol, l, options);
        
        % Get the result back
        w(ind) = model.w(1:model.nr_feature).';
        b = - model.w(end);
        
        % Get rid of zero weights
        ind(w(ind) == 0) = [];
        
        % Find the most violating constraints
        grad_w = grad_obj(y, x, w, b, C, l, grad_loss); % Gradient
        [grad_w, grad_w_ind] = sort(grad_w); % Sorting in ascending order
        n_var_neg = numel(find(grad_w <= 0)); % Number of violating (ie negative) constraints
        ind = unique([ind; grad_w_ind(1:min(n_var_add, n_var_neg))]); % Add the most violating variables
        max_viol = grad_w(1); % Maximum violation of the constraint
        
        % Objective value
        objective(n_it) = model.obj(end);
        
        % Display
        print(verb, '%5d   %7.2f   %6.3f     %3d\n', ...
            n_it, model.obj(end), -max_viol, numel(ind));
    
        % Stopping criterion (if max_viol is non negative)
        if (max_viol >= -options.act_set_tol)
            break;
        end
    end
    if (n_it == options.n_it_max_out)
    %     print(verb, 'Did not converge after %d iterations.\n', n_it);
        warning('nsvml1l2primalactivesettrain did NOT converge after %d iterations.\n', n_it);
    else
        print(verb, 'nsvml1l2primalactivesettrain converged after %d iterations.\n', n_it);
    end
    
    % Re-enable the verbosity of the inner loop
    warning('on', 'nsvml1l2primaltrain:earlyStopping');

    % Build the LibLINEAR-like model
    model.Flip = false; % Did not flip the data
    model.nr_class = 2; % Number of class
    model.nr_feature = size(x, 2); % Dimension of the instances
    model.bias = 1; % Enable the SVM bias
    model.Label = labels; % Labels
    model.w = [w; -b]'; % Weight vector + SVM bias at the end
    model.obj = objective(1:n_it); % Objective values
end
end

% Sparse diagonal matrix
function X = spdiag(vec)
    l = numel(vec);
    X = sparse(1:l, 1:l, vec, l, l);
end

% % Squared hinge loss
% function res = shinge_loss(u, v)
%     res = sum(max(0, 1 - u.*v).^2);
% end

% Gradient of the squared hinge loss
function res = grad_shinge_loss(u, v)
    res = -2 * v .* max(0, 1 - u.*v);
end

% % Objective function (minus (1-lambda)||.||_1)
% function res = obj(y, x, w, b, C, l, loss)
% %     res = (1-l)*w + l/2 * norm(w, 2)^2 + C * loss(x*w - b, y);
%     res = (1-l)*w + l/2 * sum(w.^2) + C * loss(x*w - b, y);
% end

% Gradient of the objective function
function grad_w = grad_obj(y, x, w, b, C, l, grad_loss)
    current_grad_loss = grad_loss(x*w - b, y);
    grad_w = 1-l + l*w + C * sum(spdiag(current_grad_loss) * x).';
end
