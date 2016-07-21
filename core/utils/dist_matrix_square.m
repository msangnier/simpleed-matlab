function D = dist_matrix_square(x, varargin)
%DIST_MATRIX_SQUARE Matrix of the square of the distances between rows of
%one or two matrices
%   D = DIST_MATRIX_SQUARE(x, y)
%
%   INPUT:
%   - x: first dataset as a 2D-matrix. Each row is an instance and each
%   column embodies a feature
%   - y: second dataset as a 2D-matrix (optional, default: x)
%
%   OUTPUT:
%   - D: matrix of the square of distances ||x_i - y_j||^2
%
% See also svmtrain (LIBSVM), train (LIBLINEAR), libsvm_options

% Maxime Sangnier (CEA)
% Revision: 0.1
% Date: 2014/02/14

% Input
y = check_argin(varargin, []);

% Matrix size
s1 = size(x, 1); % First dimension
if isempty(y)
    s2 = s1;
else
    s2 = size(y, 1);
end

% Square
x2 = repmat(sum(abs(x).^2, 2), 1, s2);
if ~isempty(y)
    y2 = repmat(sum(abs(y).^2, 2).', s1, 1);
end

% Distance matrix
if isempty(y)
    D = x2 + x2.' - 2*x*x.';
else
    D = x2 + y2 - 2*x*y.';
end
end

