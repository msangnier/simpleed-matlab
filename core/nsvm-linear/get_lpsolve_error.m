function mes = get_lpsolve_error(status)
%NSVML1DUALTRAIN Non-negative 1-norm SVM trained in the dual
%   model = NSVML1DUALTRAIN(labels, data, C, min_bias)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - C: cost parameter (default: 1)
%   - min_bias: minimum bias that is authorized (default: 1e-1)
%
%   OUTPUT:
%   - LibLINEAR-like model (use predict from LibLINEAR for the prediction)
%
% See also lp_solve

% Maxime Sangnier (CEA)
% Revision: 0.1
% Date: 2014/03/28

switch(status)
    case -2
        mes = 'Out of memory.';
    case 1
        mes = 'The model is suboptimal.';
    case 2
        mes = 'The model is infeasible.';
    case 3
        mes = 'The model is unbounded.';
    case 4
        mes = 'The model is degenerative.';
    case 5
        mes = 'Numerical failure encountered.';
    case 6
        mes = 'The abort routine returned true.';
    case 7
        mes = 'A time out occurred.';
    case 9
        mes = 'The model could be solved by presolved.';
    case 10
        mes = 'The B&B routine failed.';
    case 11
        mes = 'The B&B was stopped because of a break-at-first or a break-at-value.';
    case 12
        mes = 'A feasible B&B solution was found.';
    case 13
        mes = 'No feasible B&B solution found.';
    otherwise
        mes = sprintf('LPsolve status: %d.', status);
end
end