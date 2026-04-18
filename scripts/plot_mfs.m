function plot_mfs()
    load('../results/final_model_workspace.mat', 'finalPrunedFIS_Cleve', 'bin_feat');
    allFeatures = {'age','sex','cp','trestbps','chol','thalach','exang','oldpeak','slope','ca'};
    feat_names = allFeatures(bin_feat);
    
    try
        num_inputs = length(finalPrunedFIS_Cleve.Inputs);
    catch
        num_inputs = length(finalPrunedFIS_Cleve.input);
    end
    
    if ~exist('../analysis_images/anfis_mfs', 'dir')
        mkdir('../analysis_images/anfis_mfs');
    end
    
    for i = 1:num_inputs
        fig = figure('Visible', 'off', 'Color', 'white');
        plotmf(finalPrunedFIS_Cleve, 'input', i);
        title(sprintf('Membership Functions for %s', feat_names{i}));

        % Force white background and black axes/text for print publication
        ax = gca;
        ax.Color       = 'white';
        ax.XColor      = 'black';
        ax.YColor      = 'black';
        ax.GridColor   = [0.15 0.15 0.15];
        ax.TitleFontWeight = 'bold';
        set(ax.Title,  'Color', 'black');
        set(ax.XLabel, 'Color', 'black');
        set(ax.YLabel, 'Color', 'black');

        saveas(fig, sprintf('../analysis_images/anfis_mfs/mf_%s.png', feat_names{i}));
        close(fig);
    end
    
    disp('Membership function plots saved in analysis_images/anfis_mfs/');
end
