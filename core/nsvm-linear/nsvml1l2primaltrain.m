function model = nsvml1l2primaltrain(y, x, varargin)
%NSVML1L2PRIMALTRAIN Non-negative elastic net and L2-loss SVM trained in
%the primal.
%   model = NSVML1L2PRIMALTRAIN(labels, data, C, min_bias, lambda, options)
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
%       - n_it_max: maximum number of iterations (default: 5000)
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
%       - verbose: verbosity (default: false)
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%
% See also nsvml1dualtrain, nsvmtrain.

% Maxime Sangnier (CEA)
% Revision: 0.1
% Date: 2014/03/26

% Default values
[C, b_tol, l, options] = check_argin(varargin, ...
    1, ... % C
    1e-1, ... % b_tol
    0, ... % l
    struct()); % Options
options = check_options(options, ...
    'n_it_max', 5000, ...
    'beta', 0.5, ...
    'step', 1, ...
    'w', sparse(size(x, 2), 1), ...
    'b', 1, ...
    'stop_criterion', 'grad_rel', ...
    'grad_tol', 1e-5, ...
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

% Verbosity option and shortcut
verb_it = 10; % Display the objective value all the verb_it iterations
verb = options.verbose; % Shortcut

% Print options
print(verb, 'Parameters:');
print(verb, '                C: %f\n', C);
print(verb, '         min_bias: %f\n', b_tol);
print(verb, '           lambda: %f\n\n', l);
print(verb, 'Gradient options:');
% print(verb, options);
print(verb, '         n_it_max: %d\n', options.n_it_max);
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
print(verb, '          verbose: %d\n\n', options.verbose);
print(verb, '  it     obj');
print(verb, '---------------');

% Loss function and its gradient
loss = @(u, v) shinge_loss(u, v);
grad_loss = @(u, v) grad_shinge_loss(u, v);

% Determine the stopping critetion function
switch lower(options.stop_criterion)
    case 'grad_rel'
        stop_crit = @(w, b, grad_w, grad_b,  first_grad_w, first_grad_b, ...
            b_tol, tol) stop_grad_rel(w, b, grad_w, grad_b, first_grad_w, ...
            first_grad_b, b_tol, tol);
    case 'grad_inf'
        stop_crit = @(w, b, grad_w, grad_b,  first_grad_w, first_grad_b, ...
            b_tol, tol) stop_grad_inf(w, b, grad_w, grad_b, b_tol, tol);
    case 'grad_2'
        stop_crit = @(w, b, grad_w, grad_b,  first_grad_w, first_grad_b, ...
            b_tol, tol) stop_grad_2(w, b, grad_w, grad_b, b_tol, tol);
    otherwise
        error('Unknown stopping criterion (see the manual).');
end

% Init the LibLINEAR-like model
model = struct();

% Starting values
w = options.w; options.w = []; % Weight vector
b = options.b; options.b = []; % Bias
magic_f = 1; % FISTA magic factor
magic_w = w; % FISTA intermediate weight vector
magic_b = b; % FISTA intermediate bias
step = options.step; % Gradient step

objective = zeros(options.n_it_max, 1);

% Big loop
for n_it = 1:options.n_it_max
    % Objective function and gradients at the magic point
    cur_obj = obj(y, x, magic_w, magic_b, C, l, loss);
    [grad_w, grad_b] = grad_obj(y, x, magic_w, magic_b, C, l, grad_loss);
    
    % Useful for the stopping criterion (we also consider the 2nd
    % gradient, because the first one is to big
    if (n_it == 1 || n_it == 2)
        first_grad_w = abs(grad_w);
        first_grad_b = abs(grad_b);
    end
    
    % Backtracking
    new_w = prox(magic_w - step*grad_w, step*(1-l));
    new_b = prox(magic_b - step*grad_b, b_tol) + b_tol; % Projected gradient step
    new_obj = obj(y, x, new_w, new_b, C, l, loss);
    while (new_obj > cur_obj + ...
            (new_w-magic_w)'*grad_w + (new_b-magic_b)*grad_b + ...
            norm(new_w-magic_w, 2)^2 / (2*step) + (new_b-magic_b)^2 / (2*step))
        step = step * options.beta;
        new_w = prox(magic_w - step*grad_w, step*(1-l));
        new_b = prox(magic_b - step*grad_b, b_tol) + b_tol;
        new_obj = obj(y, x, new_w, new_b, C, l, loss);
    end
    
    % Update variables (FISTA)
    new_f = next_magic_f(magic_f);
    magic_w = next_magic_point(new_w, w, magic_f, new_f);
    magic_b = next_magic_point(new_b, b, magic_f, new_f);
    magic_f = new_f;
    w = new_w;
    b = new_b;
%     magic_b = b; % Disable FISTA
%     magic_w = w; % Disable FISTA
    
    % The gradient of the whole objective function (with (1-lambda)||.||_1)
    grad_w = grad_w + (1-l);
    
    objective(n_it) = (1-l)*sum(w) + new_obj;
    
    % Display
    if (mod(n_it, verb_it) == 1)
        print(verb, '%5d   %5.2f\n', n_it, (1-l)*sum(w) + new_obj);
        % Distance to the stopping point
%         disp(norm([abs(grad_w(w > 0)) - options.grad_tol * first_grad_w(w > 0); ...
%             abs(grad_b(b > b_tol)) - options.grad_tol * first_grad_b(b > b_tol)], 2));
%         disp(step);
%         hold on, plot(abs(grad_w(w>0))); hold on, plot(options.grad_tol * first_grad_w(w>0), 'r'); pause;
    end
    
    % Stopping criterion
    stop = stop_crit(w, b, grad_w, grad_b,  first_grad_w, first_grad_b, ...
            b_tol, options.grad_tol);
    if (stop)
        break;
    end
end
if (n_it == options.n_it_max)
%     print(verb, 'Did not converge after %d iterations.\n', n_it);
    warning('nsvml1l2primaltrain:earlyStopping', ...
        'nsvml1l2primaltrain did NOT converge after %d iterations.\n', n_it);
else
    print(verb, 'nsvml1l2primaltrain converged after %d iterations.\n', n_it);
end

% Build the LibLINEAR-like model
model.Flip = false; % Did not flip the data
model.nr_class = 2; % Number of class
model.nr_feature = size(x, 2); % Dimension of the instances
model.bias = 1; % Enable the SVM bias
model.Label = labels; % Labels
model.w = [w; -b]'; % Weight vector + SVM bias at the end

model.obj = objective(1:n_it);
end

% Sparse diagonal matrix
function X = spdiag(vec)
    l = numel(vec);
    X = sparse(1:l, 1:l, vec, l, l);
end

% Squared hinge loss
function res = shinge_loss(u, v)
    res = sum(max(0, 1 - u.*v).^2);
end

% Gradient of the squared hinge loss
function res = grad_shinge_loss(u, v)
    res = -2 * v .* max(0, 1 - u.*v);
end

% Objective function (minus (1-lambda)||.||_1)
function res = obj(y, x, w, b, C, l, loss)
%     res = l/2 * norm(w, 2)^2 + C * loss(x*w - b, y);
    res = l/2 * sum(w.^2) + C * loss(x*w - b, y);
end

% Gradient of the objective function
function [grad_w, grad_b] = grad_obj(y, x, w, b, C, l, grad_loss)
    current_grad_loss = grad_loss(x*w - b, y);
    grad_w = l*w + C * sum(spdiag(current_grad_loss) * x).';    
    grad_b = -C * sum(current_grad_loss);
end

% Proximal operator
function res = prox(x, nu)
    res = max(0, x - nu);
end

% Next magic factor of FISTA
function res = next_magic_f(f)
    res = (1 + sqrt(1 + 4*f^2)) / 2;
end

% Next magic point of FIST
function res = next_magic_point(x_current, x_previous, f, new_f)
    res = x_current + (f-1)/new_f * (x_current - x_previous);
end

% Stopping criteria
function stop = stop_grad_rel(w, b, grad_w, grad_b, ...
    first_grad_w, first_grad_b, b_tol, tol)
    % We separate the cases else Matlab sends an error (although they are
    % mathematically identical)
    if (all(w == 0))
        stop = all(grad_w >= 0) && ...
            (abs(grad_b) < tol * first_grad_b || ...
            (b == b_tol && grad_b >= 0)); % Boolean: true if we must stop here
    else
        stop = all(abs(grad_w(w > 0)) < tol * first_grad_w(w > 0)) && ...
            all(grad_w(w == 0) >= 0) && ...
            (abs(grad_b) < tol * first_grad_b || ...
            (b == b_tol && grad_b >= 0)); % Boolean: true if we must stop here
    end
end

function stop = stop_grad_inf(w, b, grad_w, grad_b, b_tol, tol)
    if (all(w == 0))
        stop = all(grad_w >= 0) && ...
            (abs(grad_b) < tol || (b == b_tol && grad_b >= 0));
    else
        stop = max(abs(grad_w(w > 0))) < tol && ...
            all(grad_w(w == 0) >= 0) && ...
            (abs(grad_b) < tol || (b == b_tol && grad_b >= 0));
    end
end

function stop = stop_grad_2(w, b, grad_w, grad_b, b_tol, tol)
    if (all(w == 0))
        stop = all(grad_w >= 0) && ...
            (norm(grad_b) < tol || (b == b_tol && grad_b >= 0));
    else
        stop = norm(grad_w(w > 0)) < tol && ...
            all(grad_w(w == 0) >= 0) && ...
            (norm(grad_b) < tol || (b == b_tol && grad_b >= 0));
    end
end