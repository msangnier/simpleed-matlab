function model = nsvml1dualtrain(y, x, varargin)
%NSVML1DUALTRAIN Non-negative 1-norm SVM trained in the dual.
%   model = NSVML1DUALTRAIN(labels, data, C, min_bias)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - C: cost parameter (default: 1)
%   - min_bias: minimum bias that is authorized (default: 1e-1)
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%
% See also nsvml1dualtrain, nsvmtrain.

% Maxime Sangnier (CEA)
% Revision: 0.2
% Date: 2014/04/14

% Default values
[C, b_tol] = check_argin(varargin, 1, 1e-1);

% Some inner options
zero_tol = 1e-5; % Tolerance on null weight vector
scalemode = 3; % Prevent from numerical failures

% Init the LibLINEAR-like model (by default, data is not flipped)
model = struct('Flip', false);

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
if (b_tol <= 0)
    error('The minimum bias authorized must be positive.');
end

% Useful data
[N, n] = size(x); % N: number of points, n: points dimension
Y = sparse(diag(y)); % Matrix of labels

% Prepare the data for LPsolve
lpf = ones(N, 1) + b_tol*y; % Linear application of the objective function
A = [(x' * Y); y']; % Linear application of the constraint
lpb = [ones(n, 1); 0]; % RHS of the constraint
lpe = [-ones(n, 1); 1]; % Constraint type (-1 = le, 0 = equal, 1 = ge)
lpl = zeros(N, 1); % Lower bound
lpu = C*ones(N, 1); % Upper bound

% Solve dual problem with LPsolve
% Solve     max lpf'*x
%           A*x <> lpb
%           lpl <= x <= lpu
[obj, ~, duals, stat] = lp_solve(lpf, A, lpb, lpe, lpl, lpu);

% Scale the problem if no solution found
if isempty(obj)
    [obj, ~, duals, stat] = lp_solve(lpf, A, lpb, lpe, lpl, lpu, [], scalemode);
end
% Check the result and save the SVM model
if ~isempty(obj)
    % Check that the solution is not null (it may occur if the positive
    % label has not been well chosen) else flip the data and solve the
    % problem again
%     if (sum(duals(1:end-1)) < zero_tol) % if the weight vector is null
% %         warning('flip the data and solve the problem again');
%         model.Flip = true;
%         % Correspond to x <- 1-x
%         A(1:end-1, :) = repmat(y', n, 1) - A(1:end-1, :);
%         
%         % Solve the problem again with flipped labels
%         [obj, ~, duals, stat] = lp_solve(lpf, A, lpb, lpe, lpl, lpu);
%         
%         % Check the result
%         if isempty(obj)
%             % Solve again with constraint scaling
%             [obj, ~, duals, stat] = lp_solve(lpf, A, lpb, lpe, lpl, lpu, [], scalemode);
%         end
%     end
    
    % Check the result and save the SVM model
%     if ~isempty(obj)
        % Build the LibLINEAR-like model
        model.nr_class = 2; % Number of class
        model.nr_feature = n; % Dimension of the instances
        model.bias = 1; % Enable the SVM bias
        model.Label = labels; % Labels
        model.w = sparse(duals'); % Weight vector + SVM bias at the end
        model.w(end) = model.w(end) - b_tol; % SVM bias at the end
        model.obj = obj; % Optimal objective value
%     end
end

% Impossible to solve the problem
if isempty(obj)
    save; % For debugging
    warning('Unable to solve the problem (parameters C=%f, b_tol=%f). %s', ...
        C, b_tol, get_lpsolve_error(stat));
    
%     model = []; % Empy SVM model
    
    % Build the LibLINEAR-like model (with zero weights)
    model.nr_class = 2; % Number of class
    model.nr_feature = n; % Dimension of the instances
    model.bias = 1; % Enable the SVM bias
    model.Label = labels; % Labels
    model.w = [sparse(1, n), -1]; % Weight vector + SVM bias at the end
    model.obj = []; % Optimal objective value
end
end

