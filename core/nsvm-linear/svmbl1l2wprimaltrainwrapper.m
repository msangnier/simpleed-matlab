function model = svmbl1l2wprimaltrainwrapper(y, x, varargin)
%SVMBL1L2WPRIMALTRAINWRAPPER Wrapper for svmbl1l2wprimaltrain. Solve an LP
%to get a good initialization.
%
%   See svmbl1l2wprimaltrain.
%
% See also svmbl1l2wprimaltrain, nsvmtrain.

% Maxime Sangnier (University of Rouen)
% Revision: 0.1 28-Nov-2014

% Default values
[C, l, options] = check_argin(varargin, ...
    1, ... % C
    0, ... % l
    struct()); % Options

% Tradeoff of the regularization (1-lambda)*||w||_1 + lambda*||b||_1 (default: 0)
if isfield(options, 'A')
    ll = options.A;
else
    ll = [];
end
% Lower bound of the average weight B <= sum(w)
if isfield(options, 'B')
    B = options.B;
else
    B = [];
end
% Weights of the 1-norm
if isfield(options, 'norm_weights')
    norm_weights = options.norm_weights;
else
    norm_weights = [];
end

% Solve the LP (1-norm SVM)
if (isfield(options, 'bound_cons') && options.bound_cons)
    model = svmbl1wdualtrain(y, x, C, ll, B, norm_weights);
elseif (isfield(options, 'pos_cons') && options.pos_cons)
    model = nsvmbl1dualtrain(y, x, C, ll, norm_weights);
else
    model = svmbl1dualtrain(y, x, C, ll);
end

% Get the weights for the initialization
options.w = model.w';

% Solve the true problem
model = svmbl1l2wprimaltrain(y, x, C, l, options);
end
