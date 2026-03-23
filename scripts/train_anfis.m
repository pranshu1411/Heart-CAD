rng(42);

dataPath = '../processed-data/cleveland_processed.csv';
data = readtable(dataPath);

selectedFeatures = {'cp', 'thalach', 'ca', 'oldpeak'};
continuousFeatures = {'thalach', 'oldpeak'};
targetFeature = 'target';

X = data{:, selectedFeatures};
Y = data{:, targetFeature};

contIdx = find(ismember(selectedFeatures, continuousFeatures));

cv = cvpartition(Y, 'KFold', 10);

metrics = struct('Accuracy', zeros(10,1), 'Sensitivity', zeros(10,1), ...
                 'Specificity', zeros(10,1), 'Precision', zeros(10,1), ...
                 'F1_Score', zeros(10,1), 'AUC', zeros(10,1), ...
                 'RMSE', zeros(10,1));
best_radii = zeros(10, 1);
rule_counts = zeros(10, 1);
fis_all = cell(10, 1);
all_thresholds = zeros(10, 1);
radius_log = {};

radii_grid = [0.3, 0.4, 0.5, 0.6];

fprintf('Starting 10-Fold Stratified Cross Validation for ANFIS...\n');

for fold = 1:cv.NumTestSets
    fprintf('\n--- Fold %d ---\n', fold);

    trainIdx = cv.training(fold);
    testIdx = cv.test(fold);

    trainX_raw = X(trainIdx, :);
    trainY     = Y(trainIdx);
    testX_raw  = X(testIdx, :);
    testY      = Y(testIdx);

    % --- Grid search with true nested 5-fold inner CV (stratified) ---
    best_mean_auc = -inf;
    best_fis = [];
    best_r = NaN;

    for r = radii_grid
        inner_cv = cvpartition(trainY, 'KFold', 5, 'Stratify', true);
        inner_aucs = NaN(inner_cv.NumTestSets, 1);
        inner_rules = NaN(inner_cv.NumTestSets, 1);
        all_inner_failed = true;

        for k = 1:inner_cv.NumTestSets
            innerTrIdx  = inner_cv.training(k);
            innerValIdx = inner_cv.test(k);

            innerTrainX_raw = trainX_raw(innerTrIdx, :);
            innerTrainY     = trainY(innerTrIdx);
            innerValX_raw   = trainX_raw(innerValIdx, :);
            innerValY       = trainY(innerValIdx);

            % Feature scaling — fit on inner train only
            innerTrainX = innerTrainX_raw;
            innerValX   = innerValX_raw;

            if ~isempty(contIdx)
                [innerTrainX(:, contIdx), mu, sigma] = zscore(innerTrainX_raw(:, contIdx));
                zeroVar = (sigma == 0);
                if any(zeroVar)
                    fprintf('  [WARN] Zero-variance feature(s) in fold %d, inner %d — skipping scaling.\n', fold, k);
                    sigma(zeroVar) = 1;
                end
                innerValX(:, contIdx) = (innerValX_raw(:, contIdx) - mu) ./ sigma;
            end

            innerTrainData = [innerTrainX, innerTrainY];
            innerValData   = [innerValX, innerValY];

            try
                opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', r);
                inFIS = genfis(innerTrainX, innerTrainY, opt);
                nRules = length(inFIS.rule);
                inner_rules(k) = nRules;

                if nRules <= 1
                    fprintf('  [WARN] Radius %.2f, inner fold %d: only %d rule — skipped.\n', r, k, nRules);
                    continue;
                end

                anfisOpt = anfisOptions('EpochNumber', 50, ...
                                        'ValidationData', innerValData, ...
                                        'DisplayANFISInformation', 0, ...
                                        'DisplayErrorValues', 0, ...
                                        'DisplayStepSize', 0, ...
                                        'DisplayFinalResults', 0);
                anfisOpt.InitialFIS = inFIS;
                [outFIS, ~, ~, chkFIS, ~] = anfis(innerTrainData, anfisOpt);

                if ~isempty(chkFIS)
                    current_fis = chkFIS;
                else
                    current_fis = outFIS;
                end

                val_preds = evalfis(current_fis, innerValX);
                [~, ~, ~, val_auc] = perfcurve(innerValY, val_preds, 1);
                inner_aucs(k) = val_auc;
                all_inner_failed = false;

            catch ME
                fprintf('  [WARN] Radius %.2f failed, inner fold %d: %s\n', r, k, ME.message);
                continue;
            end
        end

        mean_auc = mean(inner_aucs, 'omitnan');
        mean_rules = mean(inner_rules, 'omitnan');
        radius_log{end+1, 1} = fold;
        radius_log{end, 2} = r;
        radius_log{end, 3} = mean_auc;
        radius_log{end, 4} = mean_rules;
        fprintf('  Radius %.2f -> mean inner AUC: %.4f, mean rules: %.1f\n', r, mean_auc, mean_rules);

        if ~all_inner_failed && mean_auc > best_mean_auc
            best_mean_auc = mean_auc;
            best_r = r;
        end
    end

    % --- Retrain on full outer train with best radius ---
    % Scale full outer train
    fullTrainX = trainX_raw;
    testX = testX_raw;

    if ~isempty(contIdx)
        [fullTrainX(:, contIdx), mu, sigma] = zscore(trainX_raw(:, contIdx));
        zeroVar = (sigma == 0);
        if any(zeroVar), sigma(zeroVar) = 1; end
        testX(:, contIdx) = (testX_raw(:, contIdx) - mu) ./ sigma;
    end

    if isnan(best_r)
        fprintf('  [ERROR] All radii failed for fold %d. Attempting fallback radius 0.5...\n', fold);
        best_r = 0.5;
    end

    try
        opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', best_r);
        inFIS = genfis(fullTrainX, trainY, opt);

        % Use 80/20 stratified holdout on outer train for early stopping
        cv_es = cvpartition(trainY, 'HoldOut', 0.2, 'Stratify', true);
        esTrainX = fullTrainX(cv_es.training, :);
        esTrainY = trainY(cv_es.training);
        esValX   = fullTrainX(cv_es.test, :);
        esValY   = trainY(cv_es.test);

        anfisOpt = anfisOptions('EpochNumber', 50, ...
                                'ValidationData', [esValX, esValY], ...
                                'DisplayANFISInformation', 0, ...
                                'DisplayErrorValues', 0, ...
                                'DisplayStepSize', 0, ...
                                'DisplayFinalResults', 0);
        anfisOpt.InitialFIS = inFIS;
        [outFIS, ~, ~, chkFIS, ~] = anfis([esTrainX, esTrainY], anfisOpt);

        if ~isempty(chkFIS)
            best_fis = chkFIS;
        else
            best_fis = outFIS;
        end
    catch ME
        fprintf('  [FATAL] Retrain failed for fold %d: %s\n', fold, ME.message);
        continue;
    end

    fis_all{fold} = best_fis;
    best_radii(fold) = best_r;
    rule_counts(fold) = length(best_fis.rule);

    % Threshold tuning — use holdout validation predictions for Youden's J
    val_preds = evalfis(best_fis, esValX);
    [Xroc, Yroc, Troc, ~] = perfcurve(esValY, val_preds, 1);

    [~, optIdx] = max(Yroc - Xroc);
    optimal_threshold = Troc(optIdx);

    if isnan(optimal_threshold) || isempty(optimal_threshold)
        optimal_threshold = 0.5;
    end
    all_thresholds(fold) = optimal_threshold;

    % Evaluate on outer test set
    test_preds = evalfis(best_fis, testX);
    test_class_preds = double(test_preds >= optimal_threshold);

    TP = sum((test_class_preds == 1) & (testY == 1));
    TN = sum((test_class_preds == 0) & (testY == 0));
    FP = sum((test_class_preds == 1) & (testY == 0));
    FN = sum((test_class_preds == 0) & (testY == 1));

    metrics.Accuracy(fold)    = (TP + TN) / length(testY);
    metrics.Sensitivity(fold) = TP / max((TP + FN), 1);
    metrics.Specificity(fold) = TN / max((TN + FP), 1);
    metrics.Precision(fold)   = TP / max((TP + FP), 1);
    metrics.F1_Score(fold)    = (2 * TP) / max((2 * TP + FP + FN), 1);
    metrics.RMSE(fold)        = sqrt(mean((test_preds - testY).^2));

    try
        [~, ~, ~, AUC_test] = perfcurve(testY, test_preds, 1);
        metrics.AUC(fold) = AUC_test;
    catch ME
        fprintf('  [WARN] AUC computation failed for fold %d: %s\n', fold, ME.message);
        metrics.AUC(fold) = NaN;
    end

    fprintf('  Rules: %d | Radius: %.1f | Threshold: %.2f\n', rule_counts(fold), best_r, optimal_threshold);
    fprintf('  Acc: %.3f | Sens: %.3f | Spec: %.3f | Prec: %.3f | F1: %.3f | AUC: %.3f\n', ...
        metrics.Accuracy(fold), metrics.Sensitivity(fold), metrics.Specificity(fold), ...
        metrics.Precision(fold), metrics.F1_Score(fold), metrics.AUC(fold));
end

%% Final Results
fprintf('  FINAL 10-FOLD CV RESULTS (ANFIS)\n');
fprintf('Accuracy:    %.4f +/- %.4f\n', mean(metrics.Accuracy), std(metrics.Accuracy));
fprintf('Sensitivity: %.4f +/- %.4f\n', mean(metrics.Sensitivity), std(metrics.Sensitivity));
fprintf('Specificity: %.4f +/- %.4f\n', mean(metrics.Specificity), std(metrics.Specificity));
fprintf('Precision:   %.4f +/- %.4f\n', mean(metrics.Precision), std(metrics.Precision));
fprintf('F1 Score:    %.4f +/- %.4f\n', mean(metrics.F1_Score), std(metrics.F1_Score));
fprintf('ROC AUC:     %.4f +/- %.4f\n', mean(metrics.AUC, 'omitnan'), std(metrics.AUC, 'omitnan'));
fprintf('RMSE:        %.4f +/- %.4f\n', mean(metrics.RMSE), std(metrics.RMSE));
fprintf('Avg Rules:   %.1f +/- %.1f\n', mean(rule_counts), std(rule_counts));

% Limitation note
fprintf('\nNote: ANFIS does not natively support class-weighted training.\n');
fprintf('Mild class imbalance in Cleveland data is handled via stratified CV only.\n');

%% Radius vs AUC Log
fprintf('\n--- Radius Selection Log (rules vs AUC per outer fold) ---\n');
fprintf('%-6s %-8s %-12s %-12s\n', 'Fold', 'Radius', 'Mean AUC', 'Mean Rules');
for i = 1:size(radius_log, 1)
    fprintf('%-6d %-8.2f %-12.4f %-12.1f\n', radius_log{i,1}, radius_log{i,2}, radius_log{i,3}, radius_log{i,4});
end

%% Save results
resultsDir = '../results';
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

save(fullfile(resultsDir, 'anfis_results.mat'), 'fis_all', 'metrics', ...
     'rule_counts', 'best_radii', 'all_thresholds', 'radius_log');
fprintf('\nWorkspace saved to %s/anfis_results.mat\n', resultsDir);

% CSV export
foldNums = (1:10)';
T = table(foldNums, metrics.Accuracy, metrics.Sensitivity, metrics.Specificity, ...
          metrics.Precision, metrics.F1_Score, metrics.AUC, metrics.RMSE, ...
          rule_counts, best_radii, all_thresholds, ...
          'VariableNames', {'Fold', 'Accuracy', 'Sensitivity', 'Specificity', ...
                            'Precision', 'F1_Score', 'AUC', 'RMSE', ...
                            'Rules', 'Radius', 'Threshold'});

meanRow = table(0, mean(metrics.Accuracy), mean(metrics.Sensitivity), ...
                mean(metrics.Specificity), mean(metrics.Precision), ...
                mean(metrics.F1_Score), mean(metrics.AUC, 'omitnan'), ...
                mean(metrics.RMSE), mean(rule_counts), NaN, NaN, ...
                'VariableNames', T.Properties.VariableNames);
T = [T; meanRow];

csvPath = fullfile(resultsDir, 'anfis_results.csv');
writetable(T, csvPath);
fprintf('Results CSV saved to %s\n', csvPath);

fprintf('Done.\n');
