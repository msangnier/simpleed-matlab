function varargout = check_argin(varargin)
%CHECK_ARGIN Check the arguemnts of a function.
%   [arg1, arg2, ...] = check_argin(varargin, def_value1, def_value2, ...)
%
%   INPUT:
%   - varargin: arguments of a function as a cell array
%   - def_value1: default value for argument 1
%   - ...
%
%   OUTPUT:
%   - arg1: argument 1 is equal to the corresponding value given to the
%   funtion or is equal to the default value def_value1 if it has not been
%   given or if it is empty.
%
% See also check_options, parse_argin

% Maxime Sangnier (CEA)
% Revision: 0.1
% Date: 2014/01/16

% Cell array with the arguments
args_arr = varargin{1};
args = varargin(2:end);

len_arr = length(args_arr);
len_args = length(args);
if (len_arr > len_args)
    error('some default values are missing');
end

% Output
varargout = cell(1, len_args);

for i = 1:len_args
    % If the argument does not exist or is []
    if (len_arr < i || isempty(args_arr{i}))
        % Default value
        varargout{i} = args{i};
    else
        % Original value
        varargout{i} = args_arr{i};
    end
end
end