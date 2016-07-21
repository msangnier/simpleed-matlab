function [landmarks, norm_weights, landmarks_labels] = build_landmarks(db, opt)
    %% Norm weights
    if (strcmpifirst(opt.kernel, 'sim') || ...
            strcmpifirst(opt.kernel, 'miles'))
        norm_weights = build_norm_weights(opt.nsvm_low_weight, ...
            opt.nsvm_high_weight, db.ind_file, ...
            opt.nsvm_weighting_mode, opt.dec_landmarks);
    else
        norm_weights = [];
    end
    
    %% Landmarks
    % Decimation
    ind_file = cellfun(@(x) x(1:opt.dec_landmarks:end), db.ind_file, ...
        'UniformOutput', false);
    % Get landmarks
    landmarks = zeros(sum(cellfun(@numel, ind_file)), size(db.data, 2));
    landmarks_labels = zeros(size(landmarks, 1), 1);
    indw = 1; % Index in landmarks matrix
    for ii = 1:length(ind_file)
        n = length(ind_file{ii});
        landmarks(indw:indw+n-1, :) = db.data(ind_file{ii}, :);
        landmarks_labels(indw:indw+n-1, :) = db.labels(ind_file{ii});
        indw = indw + n;
    end
end