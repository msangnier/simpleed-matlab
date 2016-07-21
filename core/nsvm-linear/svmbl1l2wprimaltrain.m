function model = svmbl1l2wprimaltrain(y, x, varargin)
%SVMBL1L2WPRIMALTRAIN Elastic net and L2-loss SVM trained in the primal
%(generalized forward backward splitting)
%   model = SVMBL1L2WPRIMALTRAIN(labels, data, C, lambda, options)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - C: cost parameter (default: 1)
%   - lambda: tradeoff of the elastic net regularization (1-lambda)*||.||_1
%   + lambda/2*||.||_2^2 (default: 0 -> 1-norm SVM)
%   - options: structure that can contain:
%       - norm_weights: weights of the 1-norm (default: ones(dim data, 1))
%       - A: tradeoff of the regularization on the bias
%       (1-A)*||w||_elastic-net + A*||b||_elastic-net (default: 0)
%       - pos_cons: add the constraint w >= 0 if true (default: false)
%       - bound_cons: add the constraint sum(cons_weights .* w) >= B if
%       true (default: false)
%       - cons_weights: weights of the previous constraint (default:
%       ones(dim data, 1))
%       - B: bound of the previous constraint (default: 0)
%       - obj_tol: stop when (obj_t - obj_{t-1}) < obj_tol (default: 1e-7)
%       - n_it_max: maximum number of iterations (default: 5000)
%       - verbose: verbosity (default: false)
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%
%   REFERENCES:
%   H. Raguet, J. Fadili, and G. Peyre. Generalized forward-backward splitting.
%   SIAM Journal on Imaging Sciences, 6(3):1199–1226, July 2013. arXiv: 1108.4404.
%   E. Richard, P.-A. Savalle, and N. Vayatis. Estimation of simultaneously
%   sparse and low rank matrices. In International Conference on Machine Learning, 2013.
%
% See also nsvml1dualtrain, nsvmtrain.

% Maxime Sangnier (University of Rouen)
% Revision: 0.1 28-Nov-2014

% Default values
[C, l, options] = check_argin(varargin, ...
    1, ... % C
    0, ... % l
    struct()); % Options
options = check_options(options, ...
    'norm_weights', ones(size(x, 2), 1), ...
    'A', 0, ...
    'pos_cons', false, ...
    'bound_cons', false, ...
    'cons_weights', ones(size(x, 2), 1), ...
    'B', 0, ...
    'n_it_max', 5000, ...
    'obj_tol', 1e-7, ...
    'verbose', false, ...
    'w', sparse(size(x, 2)+1, 1));
%     'beta', 0.5, ...
%     'step', 1, ...
%     'w', sparse(size(x, 2), 1), ...
%     'b', 1, ...
%     'stop_criterion', 'grad_rel', ...
%     'grad_tol', 1e-5, ...

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
if (l < 0 || l > 1)
    error('The elastic net tradeoff must lie between 0 and 1.');
end
if (size(options.norm_weights, 1) ~= size(x, 2))
    error('The size of norm_weights does not match the number of variables.');
end
if (size(options.cons_weights, 1) ~= size(x, 2))
    error('The size of cons_weights does not match the number of variables.');
end
if all(options.cons_weights == 0)
    error('If you do not want to activate the linear constraint, set bound_cons to false and leave cons_weights to the default value.');
end
if (options.pos_cons && options.bound_cons)
    error('Please, do not active both constraints. Chose either the box or the linear one.');
end

% Verbosity option and shortcut
verb_it = 100; % Display the objective value all the verb_it iterations
verb = options.verbose; % Shortcut

% Map the data to a new space so that we can drop the bias
[n, r] = size(x); % #points x dimension
r = r + 1;
x = bsxfun(@times, [x, ones(n, 1)], y); % labels x data
norm_weights = [(1 - options.A) * options.norm_weights; options.A]; % Weights of the weighted norm + tradeoff regularization w and b
cons_weights = [options.cons_weights; 0];

% Print parameters
print(verb, 'Parameters:');
print(verb, '                    C: %f\n', C);
print(verb, '   lambda Elastic-net: %f\n\n', l);
print(verb, '0ptions:');
if (numel(options.norm_weights) > 2)
    print(verb, '    norm_weights: [%0.2f %0.2f ... %0.2f]\n', ...
        options.norm_weights(1), options.norm_weights(2), ...
        options.norm_weights(end));
else
    print(verb, '    norm_weights: [');
    print(verb, '%0.2f ', options.norm_weights);
    print(verb, ']\n');
end
print(verb, '    A: %0.2f\n', options.A);
if (options.bound_cons)
    print(verb, '    positive constraint: active\n');
else
    print(verb, '    positive constraint: inactive\n');
end
if (options.pos_cons)
    print(verb, '    bound constraint: active');
else
    print(verb, '    bound constraint: inactive');
end
if (numel(options.cons_weights) > 2)
    print(verb, '    cons_weights: [%0.2f %0.2f ... %0.2f]\n', ...
        options.cons_weights(1), options.cons_weights(2), ...
        options.cons_weights(end));
else
    print(verb, '    cons_weights: [');
    print(verb, '%0.2f ', options.cons_weights);
    print(verb, ']\n');
end
print(verb, '    bound: %0.2f\n', options.B);
print(verb, '    max it: %d\n', options.n_it_max);
print(verb, '    obj tol: %e\n', options.obj_tol);
% print(verb, options);
print(verb, '  it     obj');
print(verb, '---------------');

% Starting value
% w = sparse(r, 1); % w = [weights; bias] 
w = options.w; % w = [weights; bias]
if (options.pos_cons)
    w(1:end-1) = max(0, w(1:end-1)); % Projection on the positive orthant
end
if (options.bound_cons)
    w = proj(w, cons_weights, options.B); % Projection to cons_weights'*w >= options.B
end
zr = w; %sparse(r, 1); % Temporary variable in the GBF algorithm (regularizaiton)
nr = 1; % 1 regularization function: elastic-net
if (options.pos_cons || options.bound_cons)
    zc = w; %sparse(r, 1); % Temporary variable in the GBF algorithm (constraint)
    nr = 2; % 2 regularization functions: elastic-net + constraint
end
step = 1; % Gradient step

% Initial information
dec = max(0, 1 - x*w); % max(0, 1 - label * (w'*x + b))
cur_loss = C*dec'*dec; % Current loss value
cur_obj = ... % Current objective value
    cur_loss + ... % Loss
    (1-l)*sum(norm_weights .* abs(w)) + ... % L1-norm
    l/2*sum(norm_weights .* w.^2); % L2-norm
grad = -2*C * x' * dec; % Loss gradient
print(verb, '%5d   %5.2f\n', 0, cur_obj); % Display

% Big loop
objective = zeros(options.n_it_max, 1); % Store the objective values
for n_it = 1:options.n_it_max
    % Backtracking (determine the step value / local Lipschitz coef)
    backtracking = true;
    while (backtracking)
        new_w = w - step*grad; % Update variables
        dec = max(0, 1 - x*new_w);
        new_loss = C*dec'*dec;
        
        % Check the decrease
        dw = new_w - w;
        gap = new_loss - ( cur_loss + dw'*grad + 1/(2*step) * dw'*dw );
%         gap = new_loss - ( cur_loss + 1e-3 * dw'*grad );
        if (gap > 0 && step > 1e-12)
            step = step / 2;
        else
            backtracking = false;
        end
    end
%     step = step * 2; % We are allowed to have step \in ]0, 2*Lipschitz coef ^-1[
    
    % Variables update
    prox_step = 1;
    zr = zr + prox_step*(prox(nr*step, l, norm_weights,  2*w - zr - step*grad) - w); % Elastic-net step
    if (options.pos_cons) % Positive constraint
        bufw = 2*w - zc - step*grad;
        zc = zc + prox_step*([max(0, bufw(1:end-1)); bufw(end)] - w);
    elseif (options.bound_cons) % Linear constraint
        zc = zc + prox_step*(proj(2*w - zc - step*grad, cons_weights, options.B) - w);
    end
    if (options.pos_cons || options.bound_cons)
        w = (zr + zc) / 2;
    else
        w = zr;
    end

    % New objective value
    dec = max(0, 1 - x*w);
    cur_loss = C*dec'*dec;
    new_obj = cur_loss + ... % Loss
        (1-l)*sum(norm_weights .* abs(w)) + ... % L1-norm
        l/2*sum(norm_weights .* w.^2); % L2-norm
    
    % Display
    if (mod(n_it, verb_it) == 0)
        print(verb, '%5d   %5.2f\n', n_it, new_obj);
    end
    
    % Store the objective value
    objective(n_it) = new_obj;
    
    % Stopping criterion
%     if (abs(new_obj - cur_obj) < options.obj_tol)
    ref_obj = objective(max(1, n_it-100));
    ref_obj = min( objective(max(1, n_it-50):max(1, n_it-50)) );
    if (n_it == 1)
        ref_obj = ref_obj + 1;
    end
    if (abs(new_obj - ref_obj) < options.obj_tol)
        break;
    end
    
    % New local information
    cur_obj = new_obj; % Current objective value
    grad = -2*C * x' * dec; % Loss gradient
end
if (n_it == options.n_it_max)
%     print(verb, 'Did not converge after %d iterations.\n', n_it);
    warning('svmbl1l2wprimaltrain:earlyStopping', ...
        'svmbl1l2wprimaltrain did NOT converge after %d iterations.\n', n_it);
else
    print(verb, 'svmbl1l2wprimaltrain converged after %d iterations.\n', n_it);
end

% Build the LibLINEAR-like model
model = struct();
model.nr_class = 2; % Number of class
model.nr_feature = r-1; % Dimension of the instances
model.bias = 1; % Enable the SVM bias
model.Label = labels; % Labels
model.w = w'; % Weight vector + SVM bias at the end

model.obj = objective(1:n_it);
end

%% Subfunctions
% Projection to v'*x >= d
function res = proj(x, v, d)
    if (v'*x >= d)
        res = x;
    else
        res = x - (v'*x - d) / (v'*v) * v;
    end
end

% Proximal operator for the elastic-net regularization
function res = prox(nu, l, mu, x)
    res = max(0, abs(x) - nu*(1-l)*mu) ./ (1 + nu*l*mu) .*sign(x);
end
