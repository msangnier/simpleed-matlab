function [xamoc, yamoc, auc] = timetodetection(scores, labels)
% 19-Nov-2014
% 16-Jan-2015   Change the way the time is normalized (globally -> for each sequence)
% 29-Dec-2015   Add 0 and 1 at the start and end of x/yamoc

% Transform the data to a cell if necessary
if isnumeric(scores)
    [N, n] = size(scores);
    scores = mat2cell(scores, ones(1, N), n);
end

% Get the final decision value
y = cellfun(@max, scores);
[~, ind] = sort(y, 'descend');

% Output
xamoc = [];
yamoc = [];

% Info
pos_bags = find(labels ~= -1);
neg_bags = find(labels == -1);
n_bags = numel(pos_bags);

for ib = 1:numel(y)
    b = -y(ind(ib));
    newscores = cellfun(@(x) x+b, scores, 'UniformOutput', false);
    fp = sum(cellfun(@max, newscores(neg_bags)) >= 0);
    
    % Time to detection
    ttd = ones(1, n_bags); % From 16-Jan-2015
    for ibag = 1:n_bags
        t = min(find(newscores{pos_bags(ibag)} >= 0));
        if ~isempty(t)
%             ttd(ibag) = t; % Before 16-Jan-2015
            % From 16-Jan-2015
            bag_len = numel(newscores{pos_bags(ibag)}) - 1;
            if (bag_len > 0)
                ttd(ibag) = (t-1) / bag_len;
            else
                ttd(ibag) = 0;
            end
        end
    end
    
    xamoc = [xamoc, fp];
    yamoc = [yamoc, mean(ttd)];
end
xamoc = xamoc / numel(neg_bags);
xamoc = [0, xamoc, 1]; % 29-Dec-2015
yamoc = [1, yamoc, 0]; % 29-Dec-2015

auc = sum((xamoc(2:end) - xamoc(1:end-1)) .* yamoc(1:end-1));

% If the curve is totally flat, the AUC is set to 1 because there should be
% something wrong
if all(yamoc == 0)
    auc = 1;
end