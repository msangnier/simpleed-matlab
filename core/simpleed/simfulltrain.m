function out = simfulltrain(y, x, varargin)
%SIMFULLTRAIN Proxy for NSVMTRAIN
%   model    = SIMFULLTRAIN(labels, data, options)
%   accuracy = SIMFULLTRAIN(labels, data, options) if options.cv > 0
%   perf     = SIMFULLTRAIN(labels, data, options) if options.cv > 0 and
%   options.perf exists
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: cell #points x 1. For each point: 2D matrix #features x #time step
%   - options: see nsvmtrain
%
%   OUTPUT:
%   - LibLINEAR-like model (DO NOT use predict from LibLINEAR for the prediction)
%   if options.cv = 0 or the accuracy if options.cv > 0
%
% See also nsvmtrain

% Maxime Sangnier (Télécom ParisTech)
% Revision: 0.1     26-May-2015

% Default values
options = check_options(check_argin(varargin, struct), ...
    'alg', 'l1dual', ... % Algorithm to use
    'C', 1, ... % Cost parameter
    'b_tol', 1e-1, ... % Minimum bias that is authorized
    'cv', 0, ... % Cross-validation
    'lambda', 0, ... % Tradeoff of the elastic net regularization
    'norm_weights', ones(size(x, 2), 1)); % Weights of the 1-norm

% Extract features at the last time step
data = zeros(length(x), size(x{1}, 1));
for ibag = 1:size(data, 1)
	data(ibag, :) = x{ibag}(:, end)';
end

% No cross-validation
if (options.cv < 1)
	out = nsvmtrain(y, data, options);

% Cross-validation
else
    % Info
%     [N, n] = size(x); % Number of instances x length of an instance
    N = size(data, 1); % Number of instances
    Ncv = floor(N / options.cv); % Number of instances in a fold
    cva = zeros(options.cv, 1); % CV accuracies
    seed = rng; rng(1); % Reproducible randomness
    ind = randperm(N); % Indexes of the instances randomly permuted
    rng(seed); clear seed; % Restore the generator settings
    
    % Copy options structure without cv
    opt = options;
    opt = rmfield(opt, 'cv');
    
    % Do the cross-validation
    for it = 1:options.cv
        ind_eval = ind((it-1)*Ncv+1 : it*Ncv); % Indexes of the validation instances
        ind_train = compl(ind_eval, N); % Indexes of the training instances
        
        % Training
        model = nsvmtrain(y(ind_train), data(ind_train, :), opt);
        % Evaluation
        [~, lib_acc, scores, scores_wrt_time] = ...
            simfullpredict(y(ind_eval), x(ind_eval), model);

	% Handcrafted evaluation
%        seq_lengths = cellfun(@(x) size(x, 1), x(ind_eval));
%        n_time = max(seq_lengths);
%        scores_wrt_time = zeros(length(ind_eval), n_time);
%        for itime = 1:n_time
%            temp = cellfun(@(x) x(min(itime, size(x, 1)), :), x(ind_eval), ...
%		'UniformOutput', false);
%            data_eval = cat(1, temp{:});
%
%            % Prediction
%	    [~, lib_acc, scores_wrt_time(:, itime)] = nsvmpredict(y(ind_eval), ...
%            	data_eval, model);
%        end
%        clear temp;
        
%	% AUAMOC
%        % Convert scores_wrt_time to a cell and restore the
%        % true lengths of the sequences
%        pred_cell = mat2cell(scores_wrt_time, ...
%            ones(1, numel(ind_eval)), n_time);
%        for ibag = 1:length(ind_eval)
%            pred_cell{ibag}(:, seq_lengths(ibag)+1:end) = [];
%        end
%        
%        [xamoc, yamoc, auamoc] = ...
%            timetodetection(pred_cell, y(ind_eval));

	% Score
%         if (isfield(options, 'perf') && ~isempty(options.perf))
        if (isfield(options, 'perf') && isa(options.perf, 'function_handle'))
            cva(it) = options.perf(scores, y(ind_eval));
        elseif (isfield(options, 'perf') && strcmpi(options.perf, 'auamoc'))
            [~, ~, auamoc] = timetodetection(scores_wrt_time, y(ind_eval));
            cva(it) = 1-auamoc;
        else
        	cva(it) = lib_acc(1);
        end
    end
    
    % Average of the accuracies
    out = mean(cva);
end
end

% Give the compatary set of lower_set in 1:upper_bound
function compl_set = compl(lower_set, upper_bound)
    compl_set = 1:upper_bound;
    compl_set(lower_set) = [];
end
