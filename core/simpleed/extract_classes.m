function db = extract_classes(db, classes)
% Extact some classes from the database

% Find the files/bags that correspond to the classes
bags_labels = db.labels(cellfun(@(x) x(1), db.ind_file)); % Labels of bags
ind_bags = find(any(repmat(bags_labels, 1, numel(classes)) == ...
    repmat(classes(:)', numel(bags_labels), 1), 2));

% Extraction
ind_file = cell(numel(ind_bags), 1); % New indexes of files
ind = 1; % Cursor
ind_tot = []; % Indexes of classes to extract
for i_ind_bags = 1:numel(ind_bags)
    ibag = ind_bags(i_ind_bags);
    ind_bag = db.ind_file{ibag}; % Indexes of instances in the bag
    ind_tot = [ind_tot; ind_bag'];
    ind_file{i_ind_bags} = ind + (0:numel(ind_bag)-1);
    ind = ind + numel(ind_bag);
end
db.data = db.data(ind_tot, :);
db.labels = db.labels(ind_tot);
db.ind_file = ind_file;
% db.labels_name = db.labels_name(classes);

% if isfield(db, 'low_miles_features')
%     db.low_miles_features = db.low_miles_features(ind_bags, ind_tot);
%     db.bags_labels = db.bags_labels(ind_bags);
% end

end