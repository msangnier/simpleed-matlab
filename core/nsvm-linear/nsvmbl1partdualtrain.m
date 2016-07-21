function model = nsvmbl1partdualtrain(y, x, varargin)
%NSVMBL1PARTPDUALTRAIN bias-regularized SVM with constraints on positive
%and negative weights trained in the dual.
%   model = NSVMBL1PARTPDUALTRAIN(labels, data, C, lambda, norm_weights,
%   ind_neg)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - C: cost parameter (default: 1)
%   - lambda: tradeoff of the regularization (1-lambda)*||w||_1 + 
%   lambda*||b||_1 (default: 0)
%   - norm_weights: weights of the 1-norm (default: ones(dim data, 1))
%   - ind_neg: indexes of non-positive weights
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%
% See also nsvmbl1wdualtrain, nsvmtrain.

% Maxime Sangnier (Télécom ParisTech)
% Revision: 0.1 19-Oct-2015

% Default values
[C, l, norm_weights, ind_neg] = ...
    check_argin(varargin, 1, 0, ones(size(x, 2), 1), []);

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
if (l < 0 || l > 1)
    error('The tradeoff parameter lambda must be between 0 and 1.');
end
if (min(ind_neg) < 1 || max(ind_neg) > size(x, 2))
    error('Indexes of negative weights out of range.');
end
if (numel(unique(ind_neg)) ~= numel(ind_neg))
    error('Multiple indexes of negative weights.');
end

% Useful data
[N, n] = size(x); % N: number of points, n: points dimension
Y = sparse(diag(y)); % Matrix of labels

% Prepare the data:
%
mask = ones(size(x));
mask(y==1, ind_neg) = -1;
mask(y==-1, ind_neg) = 0;
x = x.*mask;
clear mask;

% Train the model using the non-negative weights method
model = nsvmbl1dualtrain(y, x, C, l, norm_weights);
model.w(ind_neg) = -model.w(ind_neg);
model.Label = labels; % Labels
end

