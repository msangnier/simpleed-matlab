function nsvm_weights = build_norm_weights(nsvm_low_weight, ...
            nsvm_high_weight, ind_file, mode, dec)
% Build the weights of the weighting L1-norm

    % Polynomial mode: poly[degree]
    if strcmpi( mode(1:min(4, numel(mode))), 'poly')
        d = str2num(mode(5:end)); % Degree
        mode = 'poly';
    end

    % Step mode: step[ratio] with nsvm_low_weight before ratio and nsvm_high_weight after
    if strcmpi( mode(1:min(4, numel(mode))), 'step')
        r = str2num(mode(5:end)); % Degree
        mode = 'step';
    end

    % Tanh mode: tanh[ratio] with nsvm_low_weight before ratio and nsvm_high_weight after
    if strcmpi( mode(1:min(4, numel(mode))), 'tanh')
        r = str2num(mode(5:end)); % Degree
        mode = 'tanh';
    end

    % Decimation
    ind_file = cellfun(@(x) x(1:dec:end), ind_file, ...
        'UniformOutput', false);

    nsvm_weights = zeros(sum(cellfun(@numel, ind_file)), 1);
    indw = 1;
    for ii = 1:length(ind_file)
        n = length(ind_file{ii});
        switch mode
            case 'linear'
                % Linear increase
                range = linspace(nsvm_low_weight, nsvm_high_weight, n);
            case 'killer'
                % Low weight on the first frame and high weight on the rest
                range = [nsvm_low_weight, nsvm_high_weight*ones(1, n-1)];
            case 'log'
                b = nsvm_low_weight;
                a = (nsvm_high_weight - b) / log(10*n);
                range = a*log([1:10:10*n]) + b;
            case 'exp'
                t = linspace(0, 1, n);
                range = (exp(t*4)-1)/(exp(4) - 1) * ...
                    (nsvm_high_weight-nsvm_low_weight)+nsvm_low_weight;
            case 'poly'
                t = linspace(0, 1, n);
                range = t.^d * ...
                    (nsvm_high_weight-nsvm_low_weight)+nsvm_low_weight;
            case 'step'
                t = linspace(0, 1, n);
                range = t;
                range(t < r) = nsvm_low_weight;
                range(t >= r) = nsvm_high_weight;
            case 'tanh'
                t = linspace(-r, 1-r, n) / min(r, 1-r) * 8;
                range = 0.5 * (1+tanh(t)) * ...
                    (nsvm_high_weight-nsvm_low_weight)+nsvm_low_weight;
            otherwise
                error('Unknown mode');
        end
        nsvm_weights(indw:indw+n-1) = range;
        indw = indw + n;
    end

end

