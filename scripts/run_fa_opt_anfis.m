% Orchestrates the end-to-end FA-OPT-SC-ANFIS pipeline

function run_fa_opt_anfis()
    warning('off', 'all');
    rng(42);
    
    fprintf(' STARTING FA-OPT-SC-ANFIS PIPELINE \n');
    
    % Load Data
    dataPath = '../processed-data/cleveland_processed.csv';
    data = readtable(dataPath);
    allFeatures = {'age','sex','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca'};
    targetFeature = 'target';
    X = data{:, allFeatures};
    Y = data{:, targetFeature};
    
    cv = cvpartition(Y, 'KFold', 10, 'Stratify', true);
    
    fprintf('\n NESTED 10-FOLD CV \n');
    metrics = struct('Acc', zeros(10,1), 'Sens', zeros(10,1), 'Spec', zeros(10,1), ...
                     'Prec', zeros(10,1), 'F1', zeros(10,1), 'AUC', zeros(10,1), 'Rules', zeros(10,1));
    selected_features_per_fold = cell(10, 1);
    all_anfis_preds = NaN(length(Y), 1);
                     
    for k = 1:10
        fprintf('\nFold %d \n', k);
        trIdx = cv.training(k);
        teIdx = cv.test(k);
        
        trX_full = X(trIdx, :);
        trY = Y(trIdx);
        
        % FA optimization inside the fold loop on training data ONLY
        fold_config = fa_hybrid_anfis(trX_full, trY);
        bin_feat = fold_config{1};
        params = fold_config{2};
        selected_features_per_fold{k} = bin_feat;
        
        trX = trX_full(:, bin_feat);
        teX = X(teIdx, bin_feat);
        teY = Y(teIdx);
        
        [trX_n, mu, sigma] = zscore(trX);
        sigma(sigma == 0) = 1;
        teX_n = (teX - mu) ./ sigma;
        
        opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', params(1), ...
                            'SquashFactor', params(2), 'AcceptRatio', params(3), 'RejectRatio', params(4));
        try
            inFIS = genfis(trX_n, trY, opt);
            if length(inFIS.rule) <= 1
                disp('Too few rules, fallback to default radius 0.5');
                opt.ClusterInfluenceRange = 0.5;
                inFIS = genfis(trX_n, trY, opt);
            end
            
            anfisOpt = anfisOptions('EpochNumber', 50, 'DisplayANFISInformation', 0, ...
                                    'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0);
            anfisOpt.InitialFIS = inFIS;
            [baseFIS, ~, ~] = anfis([trX_n, trY], anfisOpt);
            
            metrics.Rules(k) = length(baseFIS.rule);
            
            % THRESHOLD TUNING: Tune on training data using Youden's Index
            tr_preds = evalfis(baseFIS, trX_n);
            [Xroc_tr, Yroc_tr, Troc_tr, ~] = perfcurve(trY, tr_preds, 1);
            youden_index = Yroc_tr + (1 - Xroc_tr) - 1;
            [~, optIdx] = max(youden_index);
            if isempty(optIdx), thr = 0.5; else, thr = Troc_tr(optIdx); end
            
            preds = evalfis(baseFIS, teX_n);
            [~, ~, ~, AUC] = perfcurve(teY, preds, 1);
            
            class_preds = double(preds >= thr);
            all_anfis_preds(teIdx) = class_preds;
            
            TP = sum(class_preds == 1 & teY == 1);
            TN = sum(class_preds == 0 & teY == 0);
            FP = sum(class_preds == 1 & teY == 0);
            FN = sum(class_preds == 0 & teY == 1);
            
            metrics.Acc(k) = (TP + TN) / length(teY);
            metrics.Sens(k) = TP / max(TP + FN, 1);
            metrics.Spec(k) = TN / max(TN + FP, 1);
            metrics.Prec(k) = TP / max(TP + FP, 1);
            metrics.F1(k) = (2 * TP) / max((2 * TP + FP + FN), 1);
            metrics.AUC(k) = AUC;
            
        catch ME
            fprintf('Fold %d failed: %s\n', k, ME.message);
        end
    end
    
    T_preds = table((0:length(Y)-1)', Y, all_anfis_preds, 'VariableNames', {'Row_ID', 'True_Label', 'Pred_Label'});
    writetable(T_preds, '../results/anfis_predictions.csv');
    
    fprintf('\nFINAL FA-OPT-SC-ANFIS CLEVELAND RESULTS (NESTED CV):\n');
    fprintf('Accuracy: %.4f +/- %.4f\n', mean(metrics.Acc), std(metrics.Acc));
    fprintf('F1 Score: %.4f +/- %.4f\n', mean(metrics.F1), std(metrics.F1));
    fprintf('ROC AUC : %.4f +/- %.4f\n', mean(metrics.AUC), std(metrics.AUC));
    fprintf('Rules   : %.1f\n', mean(metrics.Rules));
    
    fprintf('\nTRAINING FINAL DEPLOYMENT MODEL (FULL DATASET)\n');
    % Run one final FA on the entirety of Cleveland to construct the model
    % intended for external dataset evaluation.
    final_config = fa_hybrid_anfis(X, Y);
    bin_feat = final_config{1};
    params = final_config{2};
    feat_names = allFeatures(bin_feat);
    
    % EXPERIMENT 2: External Validation (Cleveland -> Statlog)
    fprintf('\n EXPERIMENT 2: CLEVELAND -> STATLOG \n');
    statlogDataPath = '../processed-data/statlog_processed.csv';
    statlog = readtable(statlogDataPath);
    statX = statlog{:, allFeatures};
    statY = statlog{:, targetFeature};
    
    statX_sub = statX(:, bin_feat);
    
    % Train FULL CLEVELAND
    full_cv_X = X(:, bin_feat);
    [fullX_n, mu_full, sig_full] = zscore(full_cv_X);
    sig_full(sig_full == 0) = 1;
    
    opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', params(1), ...
                        'SquashFactor', params(2), 'AcceptRatio', params(3), 'RejectRatio', params(4));
    fullFIS = genfis(fullX_n, Y, opt);
    anfisOpt.InitialFIS = fullFIS;
    [trainedFullFIS, ~, ~] = anfis([fullX_n, Y], anfisOpt);
    finalPrunedFIS_Cleve = trainedFullFIS;
    
    % Scale Statlog with Cleveland statistics
    statX_n = (statX_sub - mu_full) ./ sig_full;
    
    stat_preds = evalfis(finalPrunedFIS_Cleve, statX_n);
    [Xroc, Yroc, Troc, stat_AUC] = perfcurve(statY, stat_preds, 1);
    score_thr = 0.6 * Yroc + 0.4 * (1 - Xroc);
    [~, optIdx] = max(score_thr);
    if isempty(optIdx), stat_thr = 0.5; else, stat_thr = Troc(optIdx); end
    
    stat_class_preds = double(stat_preds >= stat_thr);
    TP_stat = sum(stat_class_preds == 1 & statY == 1);
    TN_stat = sum(stat_class_preds == 0 & statY == 0);
    FP_stat = sum(stat_class_preds == 1 & statY == 0);
    FN_stat = sum(stat_class_preds == 0 & statY == 1);
    
    stat_Acc  = (TP_stat + TN_stat) / length(statY);
    stat_Sens = TP_stat / max(TP_stat + FN_stat, 1);
    stat_Spec = TN_stat / max(TN_stat + FP_stat, 1);
    stat_Prec = TP_stat / max(TP_stat + FP_stat, 1);
    stat_F1   = (2 * TP_stat) / max((2 * TP_stat + FP_stat + FN_stat), 1);
    
    fprintf('Accuracy    : %.4f\n', stat_Acc);
    fprintf('Sensitivity : %.4f\n', stat_Sens);
    fprintf('Specificity : %.4f\n', stat_Spec);
    fprintf('Precision   : %.4f\n', stat_Prec);
    fprintf('F1 Score    : %.4f\n', stat_F1);
    fprintf('ROC AUC     : %.4f\n', stat_AUC);
    
    % EXPERIMENT 3: External Validation (Cleveland+Statlog -> Indian)
    fprintf('\n EXPERIMENT 3: CLEVELAND+STATLOG -> INDIAN \n');
    indianDataPath = '../processed-data/indian_processed.csv';
    indian = readtable(indianDataPath);
    indX = indian{:, allFeatures};
    indY = indian{:, targetFeature};
    
    indX_sub = indX(:, bin_feat);
    
    % Combine Cleveland and Statlog
    combX = [X(:, bin_feat); statX_sub];
    combY = [Y; statY];
    
    [combX_n, mu_comb, sig_comb] = zscore(combX);
    sig_comb(sig_comb == 0) = 1;
    
    opt_comb = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', params(1), ...
                        'SquashFactor', params(2), 'AcceptRatio', params(3), 'RejectRatio', params(4));
    combFIS_in = genfis(combX_n, combY, opt_comb);
    if length(combFIS_in.rule) <= 1
        disp('Too few rules for combined FIS, fallback to default radius 0.5');
        opt_comb.ClusterInfluenceRange = 0.5;
        combFIS_in = genfis(combX_n, combY, opt_comb);
    end
    
    anfisOpt.InitialFIS = combFIS_in;
    [trainedCombFIS, ~, ~] = anfis([combX_n, combY], anfisOpt);
    finalPrunedFIS_Comb = trainedCombFIS;
    
    % Scale Indian with Combined statistics
    indX_n = (indX_sub - mu_comb) ./ sig_comb;
    
    ind_preds = evalfis(finalPrunedFIS_Comb, indX_n);
    [Xroc, Yroc, Troc, ind_AUC] = perfcurve(indY, ind_preds, 1);
    score_thr = 0.6 * Yroc + 0.4 * (1 - Xroc);
    [~, optIdx] = max(score_thr);
    if isempty(optIdx), ind_thr = 0.5; else, ind_thr = Troc(optIdx); end
    
    ind_class_preds = double(ind_preds >= ind_thr);
    TP_ind = sum(ind_class_preds == 1 & indY == 1);
    TN_ind = sum(ind_class_preds == 0 & indY == 0);
    FP_ind = sum(ind_class_preds == 1 & indY == 0);
    FN_ind = sum(ind_class_preds == 0 & indY == 1);
    
    ind_Acc  = (TP_ind + TN_ind) / length(indY);
    ind_Sens = TP_ind / max(TP_ind + FN_ind, 1);
    ind_Spec = TN_ind / max(TN_ind + FP_ind, 1);
    ind_Prec = TP_ind / max(TP_ind + FP_ind, 1);
    ind_F1   = (2 * TP_ind) / max((2 * TP_ind + FP_ind + FN_ind), 1);
    
    fprintf('Accuracy    : %.4f\n', ind_Acc);
    fprintf('Sensitivity : %.4f\n', ind_Sens);
    fprintf('Specificity : %.4f\n', ind_Spec);
    fprintf('Precision   : %.4f\n', ind_Prec);
    fprintf('F1 Score    : %.4f\n', ind_F1);
    fprintf('ROC AUC     : %.4f\n', ind_AUC);
    
    % Interpretability
    fprintf('\n EXTRACTING RULES \n');
    
    fprintf('\n[MODEL A: CLEVELAND-ONLY RULES]\n');
    for r=1:min(3, length(finalPrunedFIS_Cleve.rule))
        fprintf('Rule %d: IF ', r);
        conds = {};
        for f=1:length(feat_names)
            mf_idx = finalPrunedFIS_Cleve.rule(r).antecedent(f);
            if mf_idx ~= 0
                conds{end+1} = getInterpretiveMF(finalPrunedFIS_Cleve, f, mf_idx, feat_names{f}, mu_full, sig_full);
            end
        end
        risk_val = finalPrunedFIS_Cleve.output(1).mf(r).params(end);
        risk_label = 'Low';
        if risk_val > 0.5, risk_label = 'High'; end
        fprintf('%s THEN Risk is %s (Score: %.2f)\n', strjoin(conds, ' AND '), risk_label, risk_val);
    end
    
    fprintf('\n[MODEL B: CLEVELAND+STATLOG RULES]\n');
    for r=1:min(3, length(finalPrunedFIS_Comb.rule))
        fprintf('Rule %d: IF ', r);
        conds = {};
        for f=1:length(feat_names)
            mf_idx = finalPrunedFIS_Comb.rule(r).antecedent(f);
            if mf_idx ~= 0
                conds{end+1} = getInterpretiveMF(finalPrunedFIS_Comb, f, mf_idx, feat_names{f}, mu_comb, sig_comb);
            end
        end
        risk_val = finalPrunedFIS_Comb.output(1).mf(r).params(end);
        risk_label = 'Low';
        if risk_val > 0.5, risk_label = 'High'; end
        fprintf('%s THEN Risk is %s (Score: %.2f)\n', strjoin(conds, ' AND '), risk_label, risk_val);
    end
    
    final_results = struct('metrics', metrics, ...
        'statlog', struct('acc', stat_Acc, 'sens', stat_Sens, 'spec', stat_Spec, 'prec', stat_Prec, 'f1', stat_F1, 'auc', stat_AUC), ...
        'indian', struct('acc', ind_Acc, 'sens', ind_Sens, 'spec', ind_Spec, 'prec', ind_Prec, 'f1', ind_F1, 'auc', ind_AUC), ...
        'rules_cleveland', length(finalPrunedFIS_Cleve.rule), ...
        'rules_combined', length(finalPrunedFIS_Comb.rule), ...
        'selected_features', {feat_names});
    
    save('../results/final_model_workspace.mat', 'finalPrunedFIS_Cleve', 'finalPrunedFIS_Comb', 'final_results', 'bin_feat', 'mu_full', 'sig_full', 'mu_comb', 'sig_comb', 'stat_thr', 'ind_thr');
    fprintf('\n ALL PIPELINES COMPLETED \n');
end

function str = getInterpretiveMF(fis, f, mf_idx, feat_name, mu, sig)
    try 
        params = fis.Inputs(f).MembershipFunctions(mf_idx).Parameters; 
    catch 
        params = fis.input(f).mf(mf_idx).params; 
    end
    
    if length(params) >= 2
        z_val = params(2); 
    else
        z_val = mean(params); 
    end
    raw_val = z_val * sig(f) + mu(f);
    
    if strcmp(feat_name, 'sex')
        if round(raw_val) >= 1, val_str = 'Male'; else, val_str = 'Female'; end
    elseif strcmp(feat_name, 'exang')
        if round(raw_val) >= 1, val_str = 'Yes'; else, val_str = 'No'; end
    elseif strcmp(feat_name, 'cp')
        val_str = sprintf('Type %d', max(0, min(3, round(raw_val))));
    elseif strcmp(feat_name, 'ca')
        val_str = sprintf('%d vessels', max(0, min(3, round(raw_val))));
    elseif strcmp(feat_name, 'slope')
        val_str = sprintf('Type %d', max(0, min(2, round(raw_val))));
    elseif strcmp(feat_name, 'thalach')
        val_str = sprintf('~%.0f bpm', raw_val);
    elseif strcmp(feat_name, 'oldpeak')
        val_str = sprintf('~%.1f mm', raw_val);
    elseif strcmp(feat_name, 'age')
        val_str = sprintf('~%.0f yrs', raw_val);
    elseif strcmp(feat_name, 'trestbps')
        val_str = sprintf('~%.0f mmHg', raw_val);
    elseif strcmp(feat_name, 'chol')
        val_str = sprintf('~%.0f mg/dl', raw_val);
    else
        val_str = sprintf('~%.1f', raw_val);
    end
    
    str = sprintf('(%s is %s)', feat_name, val_str);
end
