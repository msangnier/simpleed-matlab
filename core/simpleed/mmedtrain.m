% 14/05/28
% 21-Jan-2015   Change kOpt.type = 2 to kOpt.type = 3

function out = mmedtrain(y, x, mu, varargin)


options = check_options(check_argin(varargin, struct), ...
    'C', 1, ... % Cost parameter
    'cv', 0, ... % Cross-validation
    'Label', [1, -1]); % Labels to save in the model

%% MMED options
% Kernel option
% 0: 'Chi2', 1: 'Intersection', 2: 'Linear', 3:'Linear-unnormalized'
kOpt.type = 3;
kOpt.n = 5; % Useless for a linear kernel
kOpt.L = 0.38; % for good approx. use 0.38 for Chi2 and n=5, use 0.83 for Inter and n=5; - Useless for a linear kernel
kOpt.featType = 0; % 0: Bag, 1: Order
kOpt.sd = -1;

% Constraint option
cnstrOpt.minSegLen = 1;
cnstrOpt.maxSegLen = intmax('int32');
cnstrOpt.segStride  = 1;
cnstrOpt.trEvStride = 1;
cnstrOpt.shldCacheSegFeat  = 1;
cnstrOpt.shldCacheTrEvFeat = 1;

% Search option
sOpt.minSegLen = 1;
sOpt.maxSegLen = intmax('int32');
sOpt.segStride = 1;

% % Kernel option
% kOpt.type = 6; % linear with length normalization
% kOpt.n = 0;
% kOpt.L = 0; 
% kOpt.featType = 0; % 0: for BoW, 1: for ordered-sampling
% kOpt.sd = 0;
% kOpt.nSegDiv = 2; % segment is divided into two subsegments
% 
% % Constraint option
% cnstrOpt.minSegLen = 1;
% cnstrOpt.maxSegLen = intmax('int32');
% cnstrOpt.segStride  = 5;
% cnstrOpt.trEvStride = 5;
% cnstrOpt.shldCacheSegFeat  = 0;
% cnstrOpt.shldCacheTrEvFeat = 1;
% 
% % Search option
% sOpt.minSegLen = 1;
% sOpt.maxSegLen = intmax('int32');
% sOpt.segStride = 5;
            
%% Do the job
out = struct();
out.kOpt = kOpt;
out.sOpt = sOpt;
out.Label = options.Label;
out.nr_class = 2;

% No cross-validation
if (options.cv < 1)
%     % Count the number of classes
%     lbl_range = min(y):max(y); % Range of potential labels
%     lbl = lbl_range(histc(y, lbl_range) > 0); % Existing labels
    
    % Some info
%     nr_class = numel(lbl); % Number of classes
%     [N, nr_feature] = size(x); % Number of instances x number of features
    nr_features = size(x{1}, 1);
	fd = m_getFd(kOpt, nr_features);
	w_init = rand(fd, 1);
    
    % If this is a two class problem
%     if (nr_class < 3)
        % Build the model by solving the optimization problem
        out.nr_feature = nr_features;
        try
            [out.w, out.b] = m_mexMMED_ker(x, y, options.C, mu, w_init, ...
                kOpt, 'instant+extent', cnstrOpt);
        catch
            warning('Error in MMED: weights are set to 0 and bias to -1.');
            out.w = zeros(fd, 1);
            out.b = -1;
        end
        
    % If there are more than 2 classes
%     else
%         % Matrix with all SVM weights
%         w = zeros(nr_class, nr_feature+1);
%         
%         % Build one-vs-rest models
%         for ic = 1:nr_class % For each class
%             error('to do');
%             % Build the new labels
%             new_y = -ones(N, 1);
%             new_y(y == lbl(ic)) = 1;
%             % Solve the SVM problem
% %             model = nsvml1dualtrain(new_y, x, options.C, options.b_tol);
%             model = lambda_train(new_y, x, options);
%             % Save the weight vector
%             w(ic, :) = model.w;
%         end
%         
%         % Build the model
%         out = struct;
% %         out.Parameters = model.Parameters;
%         out.nr_class = nr_class;
%         out.nr_feature = nr_feature;
%         out.bias = model.bias;
%         out.Label = lbl';
%         out.w = sparse(w);
%     end

% Cross-validation
else
    % Info
%     [N, n] = size(x); % Number of instances x length of an instance
    N = length(x); % Number of instances
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
        model = mmedtrain(y(:, ind_train), x(1, ind_train), ...
            mu(1, ind_train), opt);
        
        % Evaluation
        [~, lib_acc, scores, ~, labels] = mmedpredict(y(:, ind_eval), ...
            x(1, ind_eval), model);

        if (isfield(options, 'perf') && ~isempty(options.perf))
            cva(it) = options.perf(scores, labels);
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