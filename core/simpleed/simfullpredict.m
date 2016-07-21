function varargout = simfullpredict(y, x, model, varargin)
%SIMFULLPREDICT Proxy for NSVMPREDICT
%   [predicted_label, accuracy, decision_values, pred_cell, forecast_cell] = SIMFULLPREDICT(labels, data, model)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: cell #points x 1. For each point: 2D matrix #features x #time step
%   - model: LibLINEAR-like model (from nsvmtrain for instance)
%
%   OUTPUT:
%   - predicted_label: prediction output vector
%   - accuracy: a vector with accuracy, mean squared error, squared correlation coefficient
%   - decision_values: scores from the classifier
%   - pred_cell: step-by-step prediction
%   - forecast_cell: step-by-step forecast (of the prediction with complete
%   sequences)
%
% See also nsvmpredict

% Maxime Sangnier (Télécom ParisTech)
% Revision: 0.1     26-May-2015
% Revision: 0.2     29-Dec-2015

forecast = check_argin(varargin, false);
if (forecast)
    lambdapredict = @(y, data, model) nsvmforecast(y, data, model);
else
    lambdapredict = @(y, data, model) nsvmpredict(y, data, model);
end

seq_lengths = cellfun(@(x) size(x, 2), x);
n_time = max(seq_lengths);
scores_wrt_time = zeros(length(x), n_time);
for itime = 1:n_time
    temp = cellfun(@(x) x(:, min(itime, size(x, 2)))', x, ...
        'UniformOutput', false);
    data = cat(1, temp{:});

    % Prediction
    [~, ~, scores_wrt_time(:, itime)] = lambdapredict(y, ...
        data, model);
end
clear temp;

% Final prediction
decision_values = max(scores_wrt_time, [], 2);

% Convert scores_wrt_time to a cell and restore the
% true lengths of the sequences
pred_cell = mat2cell(scores_wrt_time, ones(1, numel(y)), n_time);
for ibag = 1:length(pred_cell)
    pred_cell{ibag}(:, seq_lengths(ibag)+1:end) = [];
end

% Predicted label (    1 -> Label(1)    /    -1 -> Label(2)    )
predicted_label = ones(numel(decision_values), 1) * model.Label(1);
predicted_label(decision_values < 0) = model.Label(2);

% Find the prediction(    Label(1) -> 1    /    Label(2) -> -1    )
ytarget = ones(numel(decision_values), 1);
ytarget(y == model.Label(2)) = -1;

% Output
varargout{1} = predicted_label;
varargout{2} = [sum(y == predicted_label) / numel(predicted_label) * 100; ... % Accuracy
    mean((ytarget - decision_values).^2); ... % MSE
    nan];
varargout{3} = decision_values;
varargout{4} = pred_cell;
varargout{5} = y;
end
