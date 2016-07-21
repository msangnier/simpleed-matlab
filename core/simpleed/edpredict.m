function varargout = edpredict(db, model, options)
%EDPREDICT Wrapper for early detection
%   [predicted_labels, accuracy, decision_values/prob_estimates, ...
%       decision_values_wrt_time, labels] = EDPREDICT(database, model, options)
%   [predicted_label] = EDPREDICT(database, model, options)
%
%   INPUT:
%   - database: structure with the fields data (data as a 2D-matrix in
%   which each row is a feature vector), labels (classification labels.
%   corresponding to the features in data), low_miles_features (min
%   squared distance between feature vectors and bags), bags_labels
%   model: LIBSVM/LIBLINEAR model
%   - options: structure with options. 
%
%   OUTPUT:
%   - predicted_labels
%   - accuracy
%   - decision_values/prob_estimates
%   - decision_values_wrt_time
%   - labels: for MMED, predicted_labels may be sized differently than the
%   number of bags in db. Thus, use labels as ground truth.
%
% See also edtrain

% Maxime Sangnier (Télécom ParisTech)
% Revision: 0.1
% Date: 20-Jul-2016

    % If no kernel in options: error
    if ~isfield(options, 'kernel')
        error('No kernel field in the options.');
    end
    
    % Switch among kernel types
    switch(options.kernel)
        case 'simfull'
            if (isfield(options, 'forecast'))
                forecast = options.forecast;
            else
                forecast = false;
            end
            % Compute similarity features
            simfun = @(x) options.highsimfun(x, options.gamma);
            Ds = cellfun(simfun, db.low_Ds, 'UniformOutput', false);
            % Prediction
            [varargout{1}, varargout{2}, varargout{3}, varargout{4}, ...
                varargout{5}] = simfullpredict(db.bags_labels, Ds, model, ...
                forecast);
        case 'mmed'
            % Compute similarity features
            simfun = @(x) options.highsimfun(x, options.gamma);
            Ds = cellfun(simfun, db.low_Ds, 'UniformOutput', false);
            % MMED prediction
            [varargout{1}, varargout{2}, varargout{3}, varargout{4}, ...
                varargout{5}] = mmedpredict(db.mmed_labels, ...
                Ds, model);
        case 'mmed_wlr'
            % MMED prediction
            [varargout{1}, varargout{2}, varargout{3}, varargout{4}, ...
                varargout{5}] = mmedpredict(db.mmed_labels, db.low_Ds, ...
                model);
        otherwise
            error('Unknown kernel');
    end
end
