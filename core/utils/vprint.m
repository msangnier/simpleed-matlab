function vprint(varargin)
%PRINT Print formatted data to text file.
%   PRINT(fid, format, a, ...) does the same as fprintf (see the manual).
%   PRINT(verbose, fid, format, a, ...) does the same as fprintf if verbose
%   (Boolean of number) is equivalent to true. Else, does nothing.
%   PRINT(x) displays the array x like disp.
%   PRINT(verbose, x) displays the array x if verbose (Boolean of number)
%   is equivalent to true. Else, does nothing.
%
% See also fprintf

% Maxime Sangnier (CEA)
% Revision: 0.1
% Date: 2014/04/04

% Resolve the conflict print(message), print(verbose, message)
% If the first argument is a Boolean or a number
if ((islogical(varargin{1}) || isnumeric(varargin{1})) ...
        && numel(varargin{1}) == 1)
    verbose = logical(varargin{1}); % Verbosity
    varargin = varargin(2:end); % Message
    
% % If the first argument is a string, it is the message
% elseif ischar(varargin{1})
%     verbose = true; % Verbosity is true
%     
% % Else: error
% else
%     error('Wrong usage, see the manual.');

else
    verbose = true; % Verbosity is true
    
end

% Do the job
if (verbose)
    % If there is one element: use disp
    if (numel(varargin) == 1)
        disp(varargin{1});
        
    % Else, use fprintf
    else
        fprintf(varargin{:});
    end
end
end
