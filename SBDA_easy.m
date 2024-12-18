function [varargout] = SBDA_easy(y, X)
    % SBDA_easy: A simplified version of Sparse Bayesian Discriminant Analysis
    %
    % Inputs:
    % y  - Target variable (response)
    % X  - Predictor variables (design matrix)
    %
    % Outputs:
    % model  - Struct containing weights (b) and bias (b0)
    % b, b0  - Alternatively, weights and bias are returned separately

    % Add bias term to design matrix
    PHI = cat(2, ones(size(X, 1), 1), X);
    [N, P] = size(PHI);

    % Parameter initialization
    alphas = 2 * ones(P, 1); % Initial alpha values
    beta = 10; % Initial beta value
    w = ones(P, 1); % Initial weights
    d_w = Inf; % Weight change for convergence
    evidence = -Inf; % Evidence lower bound
    d_evidence = Inf; % Evidence change for convergence
    maxit = 50; % Maximum number of iterations
    stopeps = 1e-6; % Convergence tolerance
    maxvalue = 1e9; % Maximum allowable alpha value

    d = myeig(PHI); % Compute eigenvalues of PHI for regularization

    i = 1; % Iteration counter
    while (d_evidence > stopeps) && (d_w > stopeps) && (i < maxit)
        % Save old values for convergence checks
        wold = w;
        evidenceold = evidence;

        %% Remove large alpha values
        index0 = find(alphas > maxvalue); % Identify large alphas
        index1 = setdiff(1:P, index0); % Remaining indices

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
            Sigma1 = woodburyinv(diag(alphas1), PHI1', PHI1, (1 / beta) * eye(N));
        else
            % Direct computation for small dimensions
            Sigma1 = (diag(alphas1) + beta * PHI1' * PHI1)^(-1);
        end

        %% Update weights (w)
        diagSigma1 = diag(Sigma1);
        w1 = beta * Sigma1 * PHI1' * y;
        w(index1) = w1; % Update valid weights
        if ~isempty(index0)
            w(index0) = 0; % Set weights corresponding to large alphas to 0
        end

        %% Update alphas
        for g = 1:P1
            alphas1(g) = 1 / (diagSigma1(g) + w1(g)^2);
        end
        alphas(index1) = alphas1;

        %% Update beta
        rmse = sum((y - PHI1 * w).^2); % Residual sum of squares
        beta = N1 / (trace(PHI1' * PHI1 * Sigma1) + rmse);

        %% Compute evidence
        evidence = (1 / 2) * sum(log(alphas)) + (N / 2) * log(beta) ...
                 - (beta / 2) * rmse - (1 / 2) * w' * diag(alphas) * w ...
                 - (1 / 2) * sum(log((beta * d + alphas))) - (N / 2) * log(2 * pi);

        %% Check convergence criteria
        d_w = norm(w - wold);
        d_evidence = abs(evidence - evidenceold);

        % Uncomment to display iteration details
        % disp(['INFO: Iteration ' num2str(i) ': evidence = ' num2str(evidence) ...
        %       ', w change = ' num2str(d_w) ', rmse = ' num2str(rmse) ', beta = ' num2str(beta)]);

        i = i + 1;
    end

    %% Optimization termination message
    if i < maxit
        fprintf('Optimization of alpha and beta successful.\n');
    else
        fprintf('Optimization terminated due to reaching maximum iterations.\n');
    end

    % Extract weights and bias
    b = w(2:P); % Weights (excluding bias term)
    b0 = w(1);  % Bias term

    % Return outputs
    if nargout == 1
        model.b = b;
        model.b0 = b0;
        varargout{1} = model;
    elseif nargout == 2
        varargout{1} = b;
        varargout{2} = b0;
    end
end
