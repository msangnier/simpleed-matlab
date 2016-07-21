function varargout = edtrain(db, options)
%EDTRAIN Wrapper for early detection
%   model = EDTRAIN(database, options);
%
%   INPUT:
%   - database: structure with the fields data (data as a 2D-matrix in
%   which each row is a feature vector), labels (classification labels.
%   corresponding to the features in data), low_miles_features (min
%   squared distance between feature vectors and bags), bags_labels
%   - options: structure with options. 
%
%   OUTPUT:
%   - model: LIBSVM/LIBLINEAR model
%
% See also edpredict

% Maxime Sangnier (Télécom ParisTech)
% Revision: 0.1
% Date: 20-Jul-2016
    
    % If no kernel in options: error
    if ~isfield(options, 'kernel')
        error('No kernel field in the options.');
    end
    
    t_start = cputime;

    % Switch among kernel types
    switch(options.kernel)
        case 'simfull'
            % Compute similarity features
            simfun = @(x) options.highsimfun(x, options.gamma);
            Ds = cellfun(simfun, db.low_Ds, 'UniformOutput', false);
            % Linear SVM in a similarity space
            varargout{1} = simfulltrain(db.bags_labels, Ds, options);
        case 'mmed'
            % Compute similarity features
            simfun = @(x) options.highsimfun(x, options.gamma);
            Ds = cellfun(simfun, db.low_Ds, 'UniformOutput', false);
            % Learn MMED
            options.Label = options.labels;
            varargout{1} = mmedtrain(db.mmed_labels, Ds, db.mu, options);
        case 'mmed_wlr'
            % Learn MMED
            options.Label = options.labels;
            varargout{1} = mmedtrain(db.mmed_labels, db.low_Ds, db.mu, options);
        otherwise
            error('Unknown kernel');
    end
    
    varargout{1}.train_time = cputime - t_start;
end
