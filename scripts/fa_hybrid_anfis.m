% FA Optimization (Fast Search)
function best_config = fa_hybrid_anfis(X, Y)
    warning('off', 'all');
    rng(42);
    
    fprintf('Firefly Algorithm Optimization\n');
    
    % Firefly parameters
    alpha = 0.5;      % Randomness parameter
    beta0 = 1.0;      % Attractiveness at d=0
    gamma = 1.0;      % Light absorption coefficient
    alpha_damp = 0.9; % Damping ratio of alpha
    
    pop_size = 10;
    max_gen = 10;
    
    n_features = size(X, 2);
    
    % Dimensions: 10 Features + 4 SC params
    % [f1..10, radius, squashFactor, acceptRatio, rejectRatio]
    lb = [zeros(1, n_features), 0.1, 0.5, 0.5, 0.1];
    ub = [ones(1, n_features),  1.0, 2.0, 0.9, 0.5];
    dim = length(lb);
    
    % Initialize population
    fireflies = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(ub - lb, pop_size, 1);
    fitness = -inf(pop_size, 1);
    
    % Result Caching
    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    
    % Weights for fitness
    w1 = 1.0;  % F1
    w2 = 1.0;  % ROC_AUC
    w3 = 0.1;  % Rules penalty
    w4 = 0.1;  % Features penalty
    max_rules = 50; % For normalization
    
    % Cross-validation generator for fitness evaluation
    cv = cvpartition(Y, 'KFold', 3, 'Stratify', true);
    
    % Early stopping
    best_fitness_history = -inf;
    generations_without_improvement = 0;
    
    % Track all unique configs evaluated for Phase 2 output
    all_evaluated = {};
    all_evaluated_fitness = [];
    
    % Evaluation helper
    function [fit_val, nRules, bin_feat, params] = eval_firefly(fi)
        % Decode
        bin_feat = fi(1:n_features) >= 0.5;
        if sum(bin_feat) == 0
            % Force at least one feature if all 0
            bin_feat(randi(n_features)) = 1;
        end
        radius = round(fi(n_features+1), 2);
        squash = round(fi(n_features+2), 2);
        accept = round(fi(n_features+3), 2);
        reject = round(fi(n_features+4), 2);
        
        params = [radius, squash, accept, reject];
        
        % Ensure accept > reject logic
        if reject >= accept
            reject = accept - 0.1;
        end
        if reject <= 0, reject = 0.05; end
        
        % Make cache key
        feat_str = sprintf('%d', bin_feat);
        key = sprintf('%s_%.2f_%.2f_%.2f_%.2f', feat_str, radius, squash, accept, reject);
        
        if isKey(cache, key)
            cached_val = cache(key);
            fit_val = cached_val.fitness;
            nRules = cached_val.rules;
            return;
        end
        
        % Subtractive clustering options
        opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', radius, ...
                            'SquashFactor', squash, 'AcceptRatio', accept, 'RejectRatio', reject);
        
        X_sub = X(:, bin_feat);
        
        % 3-fold inner cv
        f1_scores = NaN(cv.NumTestSets, 1);
        auc_scores = NaN(cv.NumTestSets, 1);
        fold_rules = NaN(cv.NumTestSets, 1);
        
        for k = 1:cv.NumTestSets
            trIdx = cv.training(k);
            teIdx = cv.test(k);
            
            trX = X_sub(trIdx, :);
            trY = Y(trIdx);
            teX = X_sub(teIdx, :);
            teY = Y(teIdx);
            
            % Normalize
            [trX_norm, mu, sigma] = zscore(trX);
            sigma(sigma == 0) = 1;
            teX_norm = (teX - mu) ./ sigma;
            
            try
                inFIS = genfis(trX_norm, trY, opt);
                rules_k = length(inFIS.rule);
                fold_rules(k) = rules_k;
                
                if rules_k <= 1 || rules_k > max_rules * 2
                    f1_scores(k) = 0;
                    auc_scores(k) = 0.5;
                    continue;
                end
                
                anfisOpt = anfisOptions('EpochNumber', 50, ...
                                        'DisplayANFISInformation', 0, ...
                                        'DisplayErrorValues', 0, ...
                                        'DisplayStepSize', 0, ...
                                        'DisplayFinalResults', 0);
                anfisOpt.InitialFIS = inFIS;
                [outFIS, ~, ~] = anfis([trX_norm, trY], anfisOpt);
                
                % Eval
                preds = evalfis(outFIS, teX_norm);
                
                [Xroc, Yroc, Troc, AUC] = perfcurve(teY, preds, 1);
                score_thr = 0.6 * Yroc + 0.4 * (1 - Xroc);
                [~, optIdx] = max(score_thr);
                if isempty(optIdx), optIdx=1; end
                thr = Troc(optIdx);
                
                class_preds = double(preds >= thr);
                
                TP = sum((class_preds == 1) & (teY == 1));
                FP = sum((class_preds == 1) & (teY == 0));
                FN = sum((class_preds == 0) & (teY == 1));
                
                prec = TP / max(TP + FP, 1);
                rec = TP / max(TP + FN, 1);
                f1_scores(k) = (2 * prec * rec) / max(prec + rec, 1e-10);
                auc_scores(k) = AUC;
            catch ME
                f1_scores(k) = 0;
                auc_scores(k) = 0.5;
                fold_rules(k) = max_rules;
            end
        end
        
        m_f1 = mean(f1_scores, 'omitnan');
        m_auc = mean(auc_scores, 'omitnan');
        nRules = mean(fold_rules, 'omitnan');
        nFeatures = sum(bin_feat);
        
        if isnan(nRules), nRules = max_rules; end
        
        fit_val = w1 * m_f1 + w2 * m_auc - w3 * (nRules / max_rules) - w4 * (nFeatures / n_features);
        
        cache(key) = struct('fitness', fit_val, 'rules', nRules, 'f1', m_f1, 'auc', m_auc);
        
        all_evaluated{end+1} = {bin_feat, params, fit_val};
        all_evaluated_fitness(end+1) = fit_val;
    end
    
    % Main FA Loop
    for gen = 1:max_gen
        % Evaluate population
        rules_log = zeros(pop_size, 1);
        for i = 1:pop_size
            [fitness(i), rules_log(i), ~, ~] = eval_firefly(fireflies(i, :));
        end
        
        % Move fireflies
        fireflies_new = fireflies;
        for i = 1:pop_size
            for j = 1:pop_size
                if fitness(j) > fitness(i)
                    % Calculate distance
                    r = norm(fireflies(i, :) - fireflies(j, :)) / sqrt(dim);
                    beta = beta0 * exp(-gamma * r^2);
                    
                    % Move
                    fireflies_new(i, :) = fireflies(i, :) + ...
                                          beta * (fireflies(j, :) - fireflies(i, :)) + ...
                                          alpha * (rand(1, dim) - 0.5) .* (ub - lb);
                    
                    % Bound
                    fireflies_new(i, :) = max(fireflies_new(i, :), lb);
                    fireflies_new(i, :) = min(fireflies_new(i, :), ub);
                end
            end
        end
        
        fireflies = fireflies_new;
        alpha = alpha * alpha_damp;
        
        % Find best this generation
        [gen_best_fit, best_idx] = max(fitness);
        best_features = fireflies(best_idx, 1:n_features) >= 0.5;
        best_radius = fireflies(best_idx, n_features+1);
        
        fprintf('Gen %2d | Best Fit: %7.4f | Rules: %4.1f | Feats: %2d | Radius: %.2f\n', ...
            gen, gen_best_fit, rules_log(best_idx), sum(best_features), best_radius);
            
        if gen_best_fit > best_fitness_history + 1e-4
            best_fitness_history = gen_best_fit;
            generations_without_improvement = 0;
        else
            generations_without_improvement = generations_without_improvement + 1;
        end
        
        if generations_without_improvement >= 5
            fprintf('Early stopping triggered after 5 generations without improvement.\n');
            break;
        end
    end
    
    % Return only the best configuration
    fprintf('\nExtracting Best Configuration...\n');
    [~, best_idx] = max(all_evaluated_fitness);
    
    best_config = all_evaluated{best_idx};
    bin_feat = best_config{1};
    params = best_config{2};
    fit = best_config{3};
    
    fprintf('Best Fit = %.4f | Feats = %d | Params = [%.2f, %.2f, %.2f, %.2f]\n', ...
        fit, sum(bin_feat), params(1), params(2), params(3), params(4));
end
