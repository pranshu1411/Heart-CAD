clear; clc;
warning('off', 'all');

resultsFile = '../results/anfis_results.mat';
if ~exist(resultsFile, 'file')
    error('anfis_results.mat not found. Please run train_anfis.m first.');
end
load(resultsFile);

[~, best_fold] = max(metrics.AUC);
best_fis = fis_all{best_fold};

dataPath = '../processed-data/cleveland_processed.csv';
data = readtable(dataPath);

selectedFeatures = {'cp', 'thalach', 'ca', 'oldpeak', 'exang', 'slope'};
X = data{:, selectedFeatures};
Y = data{:, 'target'};

rng(42);
cv = cvpartition(Y, 'KFold', 10);
trainIdx = cv.training(best_fold);
[~, mu, sigma] = zscore(X(trainIdx, :));
zeroVar = (sigma == 0);
if any(zeroVar), sigma(zeroVar) = 1; end

fprintf('\n=== ANFIS CLINICAL CLUSTER PROFILER ===\n');

try numRules = length(best_fis.Rules); catch, numRules = length(best_fis.rule); end

clusters = struct('Severity_Score', {}, 'Z_Centers', {}, 'Raw_Centers', {});

for i = 1:numRules
    clusters(i).Z_Centers = zeros(1, length(selectedFeatures));
    clusters(i).Raw_Centers = zeros(1, length(selectedFeatures));
    
    for j = 1:length(selectedFeatures)
        try mf_idx = best_fis.Rules(i).Antecedent(j); catch, mf_idx = best_fis.rule(i).antecedent(j); end
        
        if mf_idx > 0
            try params = best_fis.Inputs(j).MembershipFunctions(mf_idx).Parameters; catch, params = best_fis.input(j).mf(mf_idx).params; end
            if length(params) >= 2, z_val = params(2); else, z_val = mean(params); end
            
            clusters(i).Z_Centers(j) = z_val;
            clusters(i).Raw_Centers(j) = z_val * sigma(j) + mu(j);
        end
    end
    
    zs = clusters(i).Z_Centers;
    clusters(i).Severity_Score = zs(1) - zs(2) + zs(3) + zs(4) + zs(5) + zs(6);
end

[~, sortIdx] = sort([clusters.Severity_Score], 'descend');
clusters = clusters(sortIdx);

highRiskIdx = [];
lowRiskIdx = [];
mixedRiskIdx = [];

for i = 1:numRules
    c = clusters(i);
    score = c.Severity_Score;
    
    if score > 1.5
        riskLevel = 'High-Severity Indicators';
        highRiskIdx(end+1) = i;
        riskTypeStr = 'high-severity';
        interpSuffix = 'This pattern aligns with recognized markers of elevated cardiovascular risk.';
    elseif score < -1.5
        riskLevel = 'Low-Severity Indicators';
        lowRiskIdx(end+1) = i;
        riskTypeStr = 'low-severity';
        interpSuffix = 'This pattern aligns with typical markers of preserved cardiovascular function.';
    else
        riskLevel = 'Mixed / Ambiguous Indicators';
        mixedRiskIdx(end+1) = i;
        riskTypeStr = 'mixed/ambiguous';
        interpSuffix = 'This pattern presents overlapping clinical markers requiring further diagnostic context.';
    end
    
    fprintf('\n--- PATIENT CLUSTER %d (%s) ---\n', i, riskLevel);

    for j = 1:length(selectedFeatures)
        [val_str, interpretation] = getDeepClinicalTerm(selectedFeatures{j}, c.Raw_Centers(j));
        fprintf('  • %-10s : %-12s (Insight: %s)\n', selectedFeatures{j}, val_str, interpretation);
    end
    
    fprintf('\n  → Interpretation:\n');
    fprintf('    This cluster identifies a %s physiological pattern characterized by:\n', riskTypeStr);
    
    [~, bestF] = sort(abs(c.Z_Centers), 'descend');
    feature_real_names = {'Chest Pain', 'Max Heart Rate', 'Major Vessels', 'ST Depression', 'Exercise Angina', 'ST Slope'};
    
    clean_parts = {};
    for k = 1:3
        featIdx = bestF(k);
        [~, term] = getDeepClinicalTerm(selectedFeatures{featIdx}, c.Raw_Centers(featIdx));
        clean_parts{end+1} = sprintf('%s (%s)', feature_real_names{featIdx}, term);
    end
    
    fprintf('    - %s\n    - %s\n    - %s\n', clean_parts{1}, clean_parts{2}, clean_parts{3});
    fprintf('    %s\n', interpSuffix);
end

fprintf('\n=== CLUSTER SUMMARY ===\n');
if isempty(highRiskIdx), high_str = 'None'; else, high_str = strjoin(arrayfun(@num2str, highRiskIdx, 'UniformOutput', false), ', '); end
if isempty(lowRiskIdx),  low_str = 'None';  else, low_str = strjoin(arrayfun(@num2str, lowRiskIdx, 'UniformOutput', false), ', '); end
if isempty(mixedRiskIdx), mix_str = 'None'; else, mix_str = strjoin(arrayfun(@num2str, mixedRiskIdx, 'UniformOutput', false), ', '); end

fprintf('- High-Severity clusters : %s\n', high_str);
fprintf('- Low-Severity clusters  : %s\n', low_str);
fprintf('- Mixed (Ambiguous)      : %s\n', mix_str);

fprintf('\n* DISCLAIMER *\n');
fprintf('These profiles represent unsupervised mathematical cluster centers discovered by ANFIS.\n');
fprintf('They demonstrate model interpretability and do not constitute clinical diagnostics.\n\n');

function [val_str, term] = getDeepClinicalTerm(featureName, val)
    if strcmp(featureName, 'thalach')
        val_str = sprintf('%.0f bpm', val);
        if val < 135, term = 'Poor peak exertion capability';
        elseif val <= 160, term = 'Moderate/Average cardiac capability';
        else, term = 'High/Preserved cardiac function';
        end
    elseif strcmp(featureName, 'exang')
        cat_val = round(val);
        val_str = sprintf('Class %d', cat_val);
        if cat_val >= 1, term = 'Present (Ischemia under stress)';
        else, term = 'Absent (Good exercise tolerance)';
        end
    elseif strcmp(featureName, 'cp')
        cat_val = max(0, min(3, round(val)));
        val_str = sprintf('Type %d', cat_val);
        if cat_val == 0, term = 'Typical Angina (Classic)';
        elseif cat_val == 1, term = 'Atypical Angina';
        elseif cat_val == 2, term = 'Non-anginal Pain';
        else, term = 'Asymptomatic (Silent/High Risk Ischemia)';
        end
    elseif strcmp(featureName, 'ca')
        cat_val = max(0, min(3, round(val)));
        if cat_val == 1, val_str = '1 vessel'; else, val_str = sprintf('%d vessels', cat_val); end
        if cat_val == 0, term = 'Clear (0 Major blockages)';
        elseif cat_val == 1, term = '1-Vessel Disease (Moderate)';
        else, term = 'Multi-Vessel Disease (High Risk)';
        end
    elseif strcmp(featureName, 'oldpeak')
        val_str = sprintf('%.1f mm', val);
        if val > 1.5, term = 'Significant Ischemic Stress (Dangerous)';
        elseif val > 0.5, term = 'Borderline ST Depression';
        else, term = 'Minimal/No ST Depression (Normal)';
        end
    elseif strcmp(featureName, 'slope')
        cat_val = max(0, min(2, round(val)));
        val_str = sprintf('Type %d', cat_val);
        if cat_val == 0, term = 'Upsloping (Normal/Benign)';
        elseif cat_val == 1, term = 'Flat (Abnormal Ischemia)';
        else, term = 'Downsloping (Severe Ischemia)';
        end
    else
        val_str = sprintf('%.2f', val);
        term = 'Unknown';
    end
end
