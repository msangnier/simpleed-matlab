function optout = check_options(optin, varargin)
%CHECK_OPTIONS Options check point.
%   options_out = check_options(options_in, 'option_name1', options_default_value1, ...)
%   checks if options are in the structure options_in and assigns the
%   default values if not.
%
%   options_out = check_options(options_in, options_default) does the same
%   with default options passed as a structure.
%
%   options_out = check_options(varargin, 'option_name1', options_default_value1, ...)
%   does the same with input options passed as a cell (varargin of the
%   form 'option_name1', options_input_value1, ... from another function)
%
%   options_out = check_options(varargin, options_default) does the same
%   with default options passed as a structure.
%
% See also check_argin, parse_argin

%   Author(s): Maxime Sangnier (CEA), 2013/01/07%
%   $Revision: 0.1 $  $Date: 2013/01/07 $

% # of input arguments as couples (except optin)
larg = (nargin - 1)/2;

% Check point
if (nargin < 1)
    error('optin is missing.');
end

if (larg - floor(larg) > 0) % # The length of varargin is odd
    if ~(nargin == 2 && isstruct(varargin{1})) % Arguments are not two structures
        error('Options must be couples (string, value) or a structure of default values.');
    end
end

% If optin is a cell, it may be a varargin from another function
if iscell(optin)
    optin = parse_argin(optin);
end

if ~isstruct(optin) % optin is not a structure
    error('optin must be a structure.');
end

% If arguments are two options structures, the second structure is
% flattened down to a fake varargin cell
if (nargin == 2 && isstruct(varargin{1}))
    optbuff = varargin{1}; % Buffering the options structure
    fnames = fieldnames(optbuff); % Getting field names
    larg = length(fnames);
    for iname = 0:larg-1 % Building the fake varargin
        varargin{2*iname+1} = fnames{iname+1};
        varargin{2*(iname+1)} = optbuff.(fnames{iname+1});
    end
end

% Building options structure
optout = optin;
for iopt = 0:larg-1
    if ~isfield(optin, varargin{2*iopt+1}) % Field does not already exist
        optout.(varargin{2*iopt+1}) = varargin{2*(iopt+1)};
    else % Verifying that classes match
        if ~isa(optin.(varargin{2*iopt+1}), class(varargin{2*(iopt+1)}))
            error('The option field %s is a %s but should be a %s.', ...
                varargin{2*iopt+1}, class(optin.(varargin{2*iopt+1})), ...
                class(varargin{2*(iopt+1)}));
        end
    end
end
end

