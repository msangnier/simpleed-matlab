function auc = aucscore(ypred, y, posclass, doplot)
%AUCSCORE Calculates the aucscore.
%   auc = AUCSCORE(scores, labels, posclass, plot)
%
%   INPUT:
%   - scores: scores from the classifier
%   - labels: true labels
%   - posclass: positive class label (default: 1)
%   - plot: plot the Receiver Operating Characteristic (ROC) curve or not
%   (default: false)
%
%   OUTPUT:
%   - auc: area under the ROC curve
%
% See perfcurve

% Leustagos
% http://www.kaggle.com/c/GiveMeSomeCredit/forums/t/997/auc-code-for-matlab
% Revision: 0.1     14-Nov-2014

if (nargin < 3 || isempty(posclass))
    posclass = 1;
end

if (posclass < 0)
    posclass = 1;
end

if (nargin < 4 || isempty(doplot))
    doplot = false;
end

if (size(y, 2) ~= 1)
    y = y';
    ypred = ypred';
end

% if all(floor(ypred) == ypred)
%     warning('ypred should be real values, not integers. Did you swap ypred for y ?');
% end

[~,ind] = sort(ypred,'descend'); 
roc_y = y(ind);
stack_x = cumsum(roc_y ~= posclass)/sum(roc_y ~= posclass);
stack_y = cumsum(roc_y == posclass)/sum(roc_y == posclass);

if isnan(stack_x)
    stack_x = zeros(numel(ypred), 1);
end
if isnan(stack_y)
    stack_y = zeros(numel(ypred), 1);
end

auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)) .* ...
    stack_y(2:length(roc_y),1));

if (doplot)
    plot(stack_x, stack_y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC curve of (AUC = ' num2str(auc) ' )']);
end
