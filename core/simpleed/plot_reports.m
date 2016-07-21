function plot_reports(reports)
    % Convert a single report in a cell
    if isstruct(reports)
        reports = {reports};
    end
    
    cmap = hsv(length(reports)); % Colormap
%     cmap = [cmap; [0, 0, 0; 0.5, 0.3, 0]]; % Black and brown
    lstyle = repmat({'-'}, length(reports), 1);
    markers = repmat([{'+'}, {'o'}, {'*'}, {'x'}, {'s'}, {'d'}, {'^'}, ...
        {'v'}, {'>'}, {'<'}, {'p'}, {'h'}], 1, ceil(length(reports)/12));
%     lstyle = {lstyle{:}, '--', '--'};

    h = figure;
    for idx = 1:length(reports)
        hold on, plot(reports{idx}.xamoc(2:end-1), ...
            reports{idx}.yamoc(2:end-1), ...
            'color', cmap(idx, :), 'LineWidth', 2, ...
            'LineStyle', lstyle{idx}, 'Marker', markers{idx});
    end
    axis([0 1 0 1]);
    legend(cellfun(@(x) x.leg, reports, 'UniformOutput', false));
    xlabel('False Positive Rate'); 
    ylabel('Normalized Time to Detect');
    title(['AMOC curve']);
end