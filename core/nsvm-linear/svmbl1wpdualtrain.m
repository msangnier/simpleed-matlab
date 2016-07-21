function model = svmbl1wpdualtrain(y, x, varargin)
%SVMBL1WPDUALTRAIN bias-regularized SVM with a penalization on negative
%weights trained in the dual.
%   model = SVMBL1WPDUALTRAIN(labels, data, C, lambda, D, norm_weights)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - C: cost parameter (default: 1)
%   - lambda: tradeoff of the regularization (1-lambda)*||w||_1 + 
%   lambda*||b||_1 (default: 0)
%   - D: penalization parameter D*max(0, -w) (default: 0)
%   - norm_weights: weights of the 1-norm (default: ones(dim data, 1))
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%
% See also nsvmbl1wdualtrain, nsvmtrain.

% Maxime Sangnier (Télécom ParisTech)
% Revision:
%     0.2 3-Feb-2016    New (smaller) dual problem
%     0.1 30-Apr-2015

% Default values
[C, l, D, norm_weights] = check_argin(varargin, 1, 0, 0, ones(size(x, 2), 1));

% Some inner options
% zero_tol = 1e-5; % Tolerance on null weight vector
scalemode = 3; % Prevent from numerical failures

% Init the LibLINEAR-like model (by default, data is not flipped)
model = struct();

% Check point
maxy = max(y);
miny = min(y);
if ~all(y == maxy | y == miny)
    error('Use only two classes.');
else
    labels = [maxy; miny]; % Labels
    % Use 1 and -1 for the labels
    ytemp = zeros(numel(y), 1);
    ytemp(y == maxy) = 1;
    ytemp(y == miny) = -1;
    y = ytemp;
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
if (D < 0)
    error('The penalization parameter must be non-negative.');
end
if (l < 0 || l > 1)
    error('The tradeoff parameter lambda must be between 0 and 1.');
end

% Useful data
[N, n] = size(x); % N: number of points, n: points dimension
Y = sparse(diag(y)); % Matrix of labels

% Prepare the data for LPsolve (2 cases)
if isinf(D)
    % First case: hard penalization (D = infty)
    model = nsvmbl1dualtrain(y, x, C, l, norm_weights);
    model.Label = labels; % Labels
else
    % Second case: 0 <= D < infty
    lpf = ones(N, 1); % Linear application of the objective function
    A = [(x' * Y); y'; (x' * Y); y']; % Linear application of the constraint
    lpb = [(1-l) * norm_weights; l; -(1-l) * norm_weights - D; -l]; % RHS of the constraint
    lpe = [-ones(n, 1); -1; ones(n, 1); 1]; % Constraint type (-1 = le, 0 = equal, 1 = ge)
    lpl = zeros(N, 1); % Lower bound
    lpu = C*ones(N, 1); % Upper bound
end

% Solve the problem only if D < infty (else it has already be done)
if isempty(fieldnames(model))
    % Solve dual problem with LPsolve
    % Solve     max lpf'*x
    %           A*x <> lpb
    %           lpl <= x <= lpu
    [obj, primals, duals, stat] = lp_solve(lpf, A, lpb, lpe, lpl, lpu);

    % Scale the problem if no solution found
    if isempty(obj)
        [obj, primals, duals, stat] = lp_solve(lpf, A, lpb, lpe, lpl, lpu, [], scalemode);
    end
    % Check the result and save the SVM model
    if ~isempty(obj)
            % Build the LibLINEAR-like model
            model.nr_class = 2; % Number of class
            model.nr_feature = n; % Dimension of the instances
            model.bias = 1; % Enable the SVM bias
            model.Label = labels; % Labels
            model.w = sparse(duals(1:n+1)'); % Weight vector + SVM bias at the end
            model.w = model.w + duals(n+2:end)'; % The weights and the bias are differences of non-negative variables (here it is a sum because for ge constraints, the duals are already negative)
            model.obj = obj; % Optimal objective value
            model.alpha = primals; % Usual SVM dual variables
    end

    % Impossible to solve the problem
    if isempty(obj)
        save; % For debugging
        warning('Unable to solve the problem (parameters C=%f, lambda=%f). %s', ...
            C, l, get_lpsolve_error(stat));

        % Build the LibLINEAR-like model (with zero weights)
        model.nr_class = 2; % Number of class
        model.nr_feature = n; % Dimension of the instances
        model.bias = 1; % Enable the SVM bias
        model.Label = labels; % Labels
        model.w = sparse(1, n+1); % Weight vector + SVM bias at the end
        model.obj = []; % Optimal objective value
    end
end
end

