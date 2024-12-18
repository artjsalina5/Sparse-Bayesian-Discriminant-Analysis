function [varargout] = SBDA(y, X, group)
    % SBDA: Sparse Bayesian Discriminant Analysis
    %
    % Inputs:
    % y      - Target variable (response)
    % X      - Predictor variables (design matrix)
    % group  - Grouping variable (e.g., feature groups)
    %
    % Outputs:
    % model  - Struct containing weights (b) and bias (b0)
    % b, b0  - Alternatively, weights and bias are returned separately

    % Initialize the design matrix
    PHI = X;
    [N, P] = size(PHI);

    % Weight initialization
    group   = [0; group(:)]; % Convert group to column vector
    groupid = unique(group);
    NG      = length(groupid); % Number of unique groups

    % Parameter initialization
    alphas = 2 * ones(P, 1); % Initial alphas
    beta = 10; % Initial beta
    w = ones(P, 1); % Initial weights
    d_w = Inf; % Weight difference for convergence
    evidence = -Inf; % Evidence lower bound
    d_evidence = Inf; % Evidence difference for convergence
    maxit = 50; % Maximum number of iterations
    stopeps = 1e-6; % Convergence tolerance
    maxvalue = 1e9; % Maximum allowable alpha value

    d = myeig(PHI); % Eigenvalues (or equivalent measure for PHI)

    i = 1; % Iteration counter
    while (d_evidence > stopeps) && (d_w > stopeps) && (i < maxit)
        wold = w;
        evidenceold = evidence;

        %% Remove large alpha values
        index0 = find(alphas > maxvalue);
        index1 = setdiff(1:P, index0); % Indices of valid alphas

        if isempty(index1)
            disp('Optimization terminated: All alphas are too large.');
            break;
        end
        alphas1 = alphas(index1);
        PHI1 = PHI(:, index1);

        %% Compute Sigma
        [N1, P1] = size(PHI1);
        if P1 > N1
            % Use Woodbury matrix identity for large dimensions
            Sigma1 = woodburyinv(diag(alphas1), PHI1', PHI1, (1/beta) * eye(N));
        else
            Sigma1 = (diag(alphas1) + beta * PHI1' * PHI1)^(-1);
        end

        %% Update weights (w)
        diagSigma1 = diag(Sigma1);
        w1 = beta * Sigma1 * PHI1' * y;
        w(index1) = w1;
        if ~isempty(index0)
            w(index0) = 0;
        end

        %% Compute gamma
        gamma1 = 1 - alphas1 .* diagSigma1;
        gamma = zeros(size(alphas));
        gamma(index1) = gamma1;

        %% Update alphas
        for g = 1:NG
            index_ig = find(group == groupid(g));
            w_ig = w(index_ig);
            if norm(w_ig) == 0
                continue;
            end
            gamma_ig = gamma(index_ig);
            alpha_ig = sum(gamma_ig) / (w_ig' * w_ig);
            alphas(index_ig) = alpha_ig;
        end

        %% Update beta
        rmse = sum((y - PHI * w).^2);
        beta = (N - sum(gamma)) / rmse;

        %% Compute evidence
        evidence = (1/2) * sum(log(alphas)) + (N/2) * log(beta) ...
                 - (beta/2) * rmse - (1/2) * w' * diag(alphas) * w ...
                 - (1/2) * sum(log((beta * d + alphas))) - (N/2) * log(2 * pi);

        %% Check convergence criteria
        d_w = norm(w - wold);
        d_evidence = abs(evidence - evidenceold);

        disp(['INFO: Iteration ' num2str(i) ': evidence = ' num2str(evidence) ...
              ', wchange = ' num2str(d_w) ', rmse = ' num2str(rmse) ', beta = ' num2str(beta)]);

        i = i + 1;
    end

    % Display selected channels
    for j = 1:size(w, 1)
        if w(j) ~= 0
            fprintf('Selected channel: %d\n', j);
        end
    end

    if i < maxit
        fprintf('Optimization of alpha and beta successful.\n');
    else
        fprintf('Optimization terminated due to reaching maximum iterations.\n');
    end

    % Output weights and bias
    b = w(2:P);
    b0 = w(1);

    if nargout == 1
        model.b = b;
        model.b0 = b0;
        varargout{1} = model;
    elseif nargout == 2
        varargout{1} = b;
        varargout{2} = b0;
    end
end
