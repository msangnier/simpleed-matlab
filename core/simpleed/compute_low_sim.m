function db = compute_low_sim(db, landmarks, simfun, incfun, ...
    incfunstep, full, verbose)
%COMPUTE_LOW_SIM Compute the landmarking representation before exp
%   db = compute_low_sim(db, landmarks, simfun, incfun, incfunstep, full, verbose)
%
%   INPUT:
%   - db: database. Structure with:
%       - data: n_points x n_features matrix
%       - labels: n_points x 1 matrix
%       - ind_file: n_bags x 1 cell, where each cell contains the range of
%       indexes of the corresponding instances in data. So
%       db.data(db.ind_file{1}, :) is the matrix of feature for the first
%       bag.
%   - landmarks: n_landmarks x n_features matrix
%   - simfun: function that computes a similarity between the instances of
%   a bag and the landmarks. The output is a n_instances x n_landmarks
%   matrix.
%   - incfun: an increasing function over instances.
%   n_instances x n_landmarks -> 1 x n_landmarks.
%   - incfunstep: an increasing function to run step by step
%   (1 x n_landmarks, 1 x n_landmarks) -> 1 x n_landmarks.
%   - full: true to compute the features at every time
%   - verbose: verbosity

% Maxime Sangnier (University of Rouen)
% Revision: 0.1   13-Nov-2014

%% Info
n_instances = size(db.data, 1);
n_bags = length(db.ind_file);
n_landmarks = size(landmarks, 1);

%% Display
if (verbose)
    h = waitbar(0, 'Computing features...');
end

%% Compute only the final features
if (~full)
    features = zeros(n_bags, n_landmarks);
    labels = zeros(n_bags, 1);
    
    tic;
    for ibag = 1:n_bags
        sim = simfun(db.data(db.ind_file{ibag}, :), landmarks);
        features(ibag, :) = incfun(sim);
        labels(ibag) = db.labels(db.ind_file{ibag}(1));
        
        if (verbose)
            remaining_time = toc/ibag*(n_bags-ibag)/60;
            h = waitbar(ibag/n_bags, h, ...
                sprintf('Computing features (remaining time: %0.2fm)...', ...
                remaining_time));
        end
            
    end
else
    %% Compute the features at each time
    features = cell(n_bags, 1);
    labels = zeros(n_bags, 1);
    
    tic;
    for ibag = 1:n_bags
        n_time = numel(db.ind_file{ibag});
        features{ibag} = zeros(n_time, n_landmarks);
        
        % First feature vector
        sim = simfun(db.data(db.ind_file{ibag}(1), :), landmarks);
        features{ibag}(1, :) = incfun(sim);
        
        % Next ones
        for itime = 2:n_time
            sim = simfun(db.data(db.ind_file{ibag}(itime), :), landmarks);
            features{ibag}(itime, :) = ...
                incfunstep(features{ibag}(itime-1, :), sim);
        end
        labels(ibag) = db.labels(db.ind_file{ibag}(1));
        
        if (verbose)
            remaining_time = toc/ibag*(n_bags-ibag)/60;
            h = waitbar(ibag/n_bags, h, ...
                sprintf('Computing features (remaining time: %0.2fm)...', ...
                remaining_time));
        end
    end
end

%% Display
if (verbose)
    close(h);
end

%% Save the features
db.low_sim_features = features;
db.bags_labels = labels;
end

