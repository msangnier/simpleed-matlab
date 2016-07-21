function [db_training, opt] = preprocess_training(db_training, opt)
	if strcmpi(opt.kernel, 'simfull')
        [opt.landmarks, opt.norm_weights, landmarks_labels] = ...
            build_landmarks(db_training, opt);
        
        % Delete landmarks with infty as weight
        nw_infty = isinf(opt.norm_weights);
        opt.landmarks(nw_infty, :) = [];
        opt.norm_weights(nw_infty) = [];
        landmarks_labels(nw_infty) = [];
        
        if (isfield(opt, 'ind_neg') && isempty(opt.ind_neg))
            opt.ind_neg = find(landmarks_labels == -1);
        end
        
        db_training = compute_low_sim(db_training, opt.landmarks, ...
            opt.simfun, opt.incfun, opt.incfunstep, true, false);
    elseif strcmpi(opt.kernel, 'mmed')
        opt.landmarks = build_landmarks(db_training, opt);
        db_training = compute_low_sim(db_training, opt.landmarks, ...
            opt.simfun, opt.incfun, opt.incfunstep, true, false);
    elseif strcmpi(opt.kernel, 'mmed_wlr')
        db_training.bags_labels = cellfun(@(x) db_training.labels(x(1)), ...
            db_training.ind_file);
        db_training.low_sim_features = ...
            cellfun(@(x) db_training.data(x, :), db_training.ind_file, ...
            'UniformOutput', false);
    else
        error('Unknown kernel')
    end
    
    % Decrease time resolution for learning
    if (opt.dec_time)
        db_training.low_sim_features = ...
            cellfun(@(x) x(1:opt.dec_landmarks:end, :), ...
            db_training.low_sim_features, 'UniformOutput', false);
    end

    % Normalization
%     opt.training_mean = 0;
    opt.training_std = max(cellfun(@(x) max(abs(x(:))), ...
        db_training.low_sim_features));
    db_training.low_sim_features = cellfun(@(x) x / opt.training_std, ...
        db_training.low_sim_features, 'UniformOutput', false);
    
    % Format the data
    db_training.low_Ds = cellfun(@(x) x.', ...
        db_training.low_sim_features, 'UniformOutput', false)';
    db_training = rmfield(db_training, 'low_sim_features');
    if strcmpifirst(opt.kernel, 'mmed')
        db_training.mmed_labels = zeros(2, numel(db_training.bags_labels));
        db_training.mu = cell(1, numel(db_training.bags_labels));
        mu_alpha = 0;
        mu_beta = 1;
        for ibag = 1:numel(db_training.bags_labels)
            n_eff_time = size(db_training.low_Ds{ibag}, 2);
            if (db_training.bags_labels(ibag) ~= -1)
                db_training.mmed_labels(:, ibag) = [1; n_eff_time];
            end
            db_training.mu{ibag} = m_func_mu(n_eff_time, ...
                db_training.mmed_labels(:, ibag), mu_alpha, mu_beta);
        end
    end

    % Clear the data and the data labels
%     db_training = rmfield(db_training, 'data');
%     db_training = rmfield(db_training, 'labels');
%     db_training = rmfield(db_training, 'ind_file');
end