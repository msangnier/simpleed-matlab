function [db_test] = preprocess_test(db_test, model, opt)
    if strcmpifirst(opt.kernel, 'simfull')
        % Compute features corresponding to non-zero weights
        ind = find(model.w(1:model.nr_feature) ~= 0);
        db_test = compute_low_sim(db_test, ...
            opt.landmarks(ind, :), opt.simfun, opt.incfun, opt.incfunstep, ...
            true, false);
        for ibag = 1:numel(db_test.bags_labels)
            features = sparse(size(db_test.low_sim_features{ibag}, 1), ...
                size(opt.landmarks, 1));
            features(:, ind) = db_test.low_sim_features{ibag};
            db_test.low_sim_features{ibag} = features;
        end
        clear features;
    elseif strcmpi(opt.kernel, 'mmed')
        % Compute similarity features at each time
        % No selection of landmarks because MMED is not a dimensionally sparse classifier
        db_test = compute_low_sim(db_test, ...
            opt.landmarks, opt.simfun, opt.incfun, opt.incfunstep, ...
            true, false);
    elseif strcmpi(opt.kernel, 'mmed_wlr')
        db_test.bags_labels = cellfun(@(x) db_test.labels(x(1)), ...
            db_test.ind_file);
        db_test.low_sim_features = ...
            cellfun(@(x) db_test.data(x, :), db_test.ind_file, ...
            'UniformOutput', false);
    else
        error('Unknown kernel')
    end

    % Normalization
    db_test.low_sim_features = cellfun(@(x) x / opt.training_std, ...
        db_test.low_sim_features, 'UniformOutput', false);

    % Format the data for MMED
    db_test.low_Ds = cellfun(@(x) x.', ...
        db_test.low_sim_features, 'UniformOutput', false)';
    db_test = rmfield(db_test, 'low_sim_features');
    db_test.mmed_labels = zeros(2, numel(db_test.bags_labels));
    for ibag = 1:numel(db_test.bags_labels)
        n_eff_time = size(db_test.low_Ds{ibag}, 2);
        if (db_test.bags_labels(ibag) ~= -1)
            db_test.mmed_labels(:, ibag) = [1; n_eff_time];
        end
    end

    % Clear the data and the data labels
%     db_test = rmfield(db_test, 'data');
%     db_test = rmfield(db_test, 'labels');
end