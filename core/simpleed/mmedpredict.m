function varargout = mmedpredict(y, x, model)
%NSVMPREDICT Non-negative Support Vector Machine prediction
%   [predicted_label, accuracy, decision_values] = NSVMPREDICT(labels, data, model)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - model: LibLINEAR-like model (from nsvmtrain for instance)
%
%   OUTPUT:
%   - predicted_label: prediction output vector
%   - accuracy: a vector with accuracy, mean squared error, squared correlation coefficient
%   - prob_estimates: If selected, probability estimate vector
%
% See also nsvmtrain

% Maxime Sangnier (CEA)
% Revision: 0.1     29-May-2014
%           0.2     14-Nov-2014

if (nargin < 3)
    error('Usage: ...');
end

if (model.nr_class > 2)
%     % Prediction
%     decision_values = x * model.w(:, 1:model.nr_feature)';
%     if (model.bias > 0) % Bias enabled
%         decision_values = decision_values + ...
%             repmat(model.w(:, end)', size(decision_values, 1), 1);
%     end
%     [~, ind_pred] = max(decision_values, [], 2);
%     predicted_label = model.Label(ind_pred);
%     % Output
%     varargout{1} = predicted_label;
%     varargout{2} = [sum(y == predicted_label) / numel(predicted_label) * 100; ... % Accuracy
%         0; ... % MSE
%         nan];
%     varargout{3} = decision_values;
    
    error('Not done yet.');
%     model.Parameter = 5;
%     model.w = full(model.w);
%     [varargout{1}, varargout{2}, varargout{3}] = predict(y, sparse(x), model, '-q');
end


% Format the MMED labels into classical classification labels:
% if y(1, i) == y(2, i) (ie start = end), y <- -1 (nothing to detect)
% else y <- 1
% lbl = ((-(y(1, :) == y(2, :)) + 0.5)*2)';
% lbl(lbl==1) = model.Label(1);
% lbl(lbl==-1) = model.Label(2);

% Prediction
lbl = nan * ones(2*size(y, 2), 1); % 2*… because there may be an event and a non-event per signal
ilbl = 1; % Index of labels
decision_values = nan * ones(size(lbl));
decision_values_wrt_time = cell(size(y, 2), 1);
% for ifile = 1:size(y, 2)
%     % Prediction for file ifile
%     detectOut = m_mexEval_ker(x{ifile}, model.w, model.b, model.kOpt, ...
%         model.sOpt);
%     decision_wrt_time = detectOut(3, :);
%     decision_values_wrt_time{ifile} = decision_wrt_time;
%     if (y(1, ifile) > 1)
%         decision_values(ilbl) = decision_wrt_time( y(2, ifile)-1 ); % End of the non-event
%         lbl(ilbl) = model.Label(2);
%         ilbl = ilbl + 1;
%     end
%     if (y(2, ifile) > 0)
%         decision_values(ilbl) = decision_wrt_time( y(2, ifile) ); % End of the event
%         lbl(ilbl) = model.Label(1);
%         ilbl = ilbl + 1;
%     end
%     if (y(1, ifile) == y(2, ifile) && y(1, ifile) == 0)
%         decision_values(ilbl) = decision_wrt_time( end ); % End of the non-event
%         lbl(ilbl) = model.Label(2);
%         ilbl = ilbl + 1;
%     end
% end
% error('vérifier pourquoi tout ça est différent !');

for ifile = 1:size(y, 2)
    % Prediction for file ifile
    if (y(1, ifile) > 1)
        detectOut = m_mexEval_ker(x{ifile}(:, 1:y(1, ifile)-1), ...
            model.w, model.b, model.kOpt, model.sOpt);
        decision_wrt_time = detectOut(3, :);
        decision_values_wrt_time{ifile} = decision_wrt_time;
        decision_values(ilbl) = decision_wrt_time( end ); % End of the non-event
        lbl(ilbl) = model.Label(2);
        ilbl = ilbl + 1;
    end
    if (y(2, ifile) > 0)
        detectOut = m_mexEval_ker(x{ifile}(:, y(1, ifile):y(2, ifile)), ...
            model.w, model.b, model.kOpt, model.sOpt);
        decision_wrt_time = detectOut(3, :);
        decision_values_wrt_time{ifile} = decision_wrt_time;
        decision_values(ilbl) = ...
            decision_wrt_time( y(2, ifile)-y(1, ifile)+1 ); % End of the event
        lbl(ilbl) = model.Label(1);
        ilbl = ilbl + 1;
    end
    if (y(1, ifile) == y(2, ifile) && y(1, ifile) == 0)
        detectOut = m_mexEval_ker(x{ifile}, model.w, model.b, model.kOpt, ...
            model.sOpt);
        decision_wrt_time = detectOut(3, :);
        decision_values_wrt_time{ifile} = decision_wrt_time;
        decision_values(ilbl) = decision_wrt_time( end ); % End of the non-event
        lbl(ilbl) = model.Label(2);
        ilbl = ilbl + 1;
    end
end
lbl(isnan(lbl)) = [];
decision_values(isnan(decision_values)) = [];

% Predicted label (    1 -> Label(1)    /    -1 -> Label(2)    )
predicted_label = ones(numel(decision_values), 1) * model.Label(1);
predicted_label(decision_values < 0) = model.Label(2);

% Find the prediction(    Label(1) -> 1    /    Label(2) -> -1    )
ytarget = ones(numel(decision_values), 1);
ytarget(lbl == model.Label(2)) = -1;

% Output
varargout{1} = predicted_label;
varargout{2} = [sum(lbl == predicted_label) / numel(predicted_label) * 100; ... % Accuracy
    mean((ytarget - decision_values).^2); ... % MSE
    nan];
varargout{3} = decision_values;
% varargout{3} = decision_wrt_time;
varargout{4} = decision_values_wrt_time;
varargout{5} = lbl;
end