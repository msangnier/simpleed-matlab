function [db_training, db_test] = create_partitions(db, varargin)
    % Default values of the arguments
    [ratio, seed, training_balance] = ...
        check_argin(varargin, 0.8, 1, 1);
    
    % Initialize randomization
    if (exist('rng','file') ~= 2)
        rand('seed', seed);
        randn('seed', seed);
    else
        rng(seed); % Random seed
    end

    % Instead of a global random permutation, do a random permutation for
    % each class
    ind_training = []; % Indexes of the training files
    ind_test = []; % Indexes of the test files    
    % Label of each file
    files_labels = zeros(length(db.ind_file), 1);
    for ifile = 1:length(db.ind_file)
        files_labels(ifile) = db.labels(db.ind_file{ifile}(1));
    end
    
    % Compute the number of taining points in each class if the classes are
    % balanced
    balance_nb = Inf;
    if (training_balance)
        for iclass = min(files_labels):max(files_labels) % For each class
            ind_file = find(files_labels == iclass); % Find the corresponding files
            n_files_training = floor(ratio*length(ind_file)); % Number of training files
            if (n_files_training > 0) % It may be zero if labels are not consecutive
                balance_nb = min(balance_nb, n_files_training);
            end
        end
    end
    
    % Find the indexes of each class for the training and the test set
    for iclass = min(files_labels):max(files_labels) % For each class
        ind_file = find(files_labels == iclass); % Find the corresponding files
        n_files_training = floor(ratio*length(ind_file)); % Number of training files
        n_files_training = min(balance_nb, n_files_training); % Limit the number of instances in each class if needed
        n_files_test = length(ind_file) - n_files_training; % Number of test files
        ind_file = ind_file(randperm(n_files_training+n_files_test)); % Random permutation of files in a given class
        ind_training = [ind_training; ind_file(1:n_files_training)]; % Indexes of the training files
        ind_test = [ind_test; ind_file(n_files_training+1:end)]; % Indexes of the test files
    end
    n_files_training = length(ind_training); % Number of training files
    n_files_test = length(ind_test); % Number of test file
    ind_files = [ind_training; ind_test]; % Random permutation of file indexes
    
    % Build the training and the test databases
    % Training database
    db_training.ind_file = cell(n_files_training, 1);
    ind_training_tot = []; % Indexes of the training instances
    for ifile = 1:n_files_training
        ind_training = db.ind_file{ind_files(ifile)};
%         ind_training = ind_training(1:train_frame_dec:end);
        ind_training_tot = [ind_training_tot; ind_training'];
        db_training.ind_file{ifile} = length(ind_training_tot) - ...
            length(ind_training)+1:length(ind_training_tot);
    end
    db_training.data = db.data(ind_training_tot, :);
    db_training.labels = db.labels(ind_training_tot);
    
    % Test database
    db_test.ind_file = cell(n_files_test, 1);
    ind_test_tot = []; % Indexes of the test instances
    for ifile = 1:n_files_test
        ind_test = db.ind_file{ind_files(n_files_training+ifile)};
%         ind_test = ind_test(1:test_frame_dec:end);
        ind_test_tot = [ind_test_tot; ind_test'];
        db_test.ind_file{ifile} = length(ind_test_tot) - ...
            length(ind_test)+1:length(ind_test_tot);
    end
    db_test.data = db.data(ind_test_tot, :);
    db_test.labels = db.labels(ind_test_tot);
    
    db_test.labels_name = db.labels_name;
    db_training.labels_name = db.labels_name;
end
