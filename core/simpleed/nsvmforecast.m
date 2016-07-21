function varargout = nsvmforecast(y, x, model)
%NSVMFORECAST Forecast the prediction for the complete sequences
%   [predicted_label, accuracy, decision_values] = NSVMFORECAST(labels, data, model)
%
%   INPUT:
%   - labels: column vector with two different values (preferably 1 and -1)
%   - data: 2D matrix (each row is a point)
%   - model: LibLINEAR-like model (from nsvmtrain for instance)
%
%   OUTPUT:
%   - predicted_label: prediction output vector
%   - accuracy: a vector with accuracy, mean squared error, squared correlation coefficient
%   - decision_values: scores from the classifier
%
% See also nsvmpredict, nsvmtrain

% Maxime Sangnier (Télécom ParisTech)
% Revision: 0.1
% Date: 29-Dec-2015

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

% Worst case
x(:, model.w(1:model.nr_feature) <= 0) = 1;

% Prediction
decision_values = x * model.w(1:model.nr_feature)';
if (model.bias > 0) % Bias enabled
    decision_values = decision_values + model.w(end);
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
end