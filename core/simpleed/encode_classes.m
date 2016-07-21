function [db, labels] = encode_classes(db, varargin)
    % Default values of the arguments
    [class_detection, class_vs] = ...
        check_argin(varargin, db.labels_name{1}, '');

    % Transform classe names into class numbers
    if (~isempty(class_detection) && isstr(class_detection))
        class_detection = find(cellfun(@(x) strcmpi(x, class_detection), ...
            db.labels_name));
        if (numel(class_detection) ~= 1)
            error('Wrong class to detect?');
        end
    end
    if (~isempty(class_vs) && isstr(class_vs))
        class_vs = find(cellfun(@(x) strcmpi(x, class_vs), ...
            db.labels_name));
        if (numel(class_vs) ~= 1)
            error('Wrong class to detect?');
        end
    end
    
    % One versus one problem (class_vs neq '' or -1)
    if (~isempty(class_vs) && class_vs > 0 && ...
            ~isempty(class_detection) && class_detection > 0)
        % Extract the two classes
        db = extract_classes(db, [class_detection, class_vs]);
        db.labels(db.labels ~= class_detection) = -1;
        % Change the labels of the bags if they exist
        if isfield(db, 'bags_labels')
            db.bags_labels(db.bags_labels ~= class_detection) = -1;
        end
    % One versus rest problem
    elseif (~isempty(class_detection) && class_detection > 0)
        % Change the labels of instances
        db.labels(db.labels ~= class_detection) = -1;
        % Change the labels of the bags if they exist
        if isfield(db, 'bags_labels')
            db.bags_labels(db.bags_labels ~= class_detection) = -1;
        end
    end
    
    labels = [class_detection, -1];
end