function [preprocessedTrain] = preprocess_NB(trainData, flag, targetCol)
 % Convert target column to numeric (TRUE -> 1, FALSE -> 0)
    trainData.(targetCol) = strcmp(trainData.(targetCol), 'TRUE');
    
    numericalCols = {'Administrative', 'Informational', 'ProductRelated', ...
        'ProductRelated_Duration', 'Informational_Duration', 'Administrative_Duration', ...
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'};
    columns_to_remove = {'Browser', 'OperatingSystems', 'Region',...
        'TrafficType','VisitorType', 'Month', 'Weekend'};

    % Remove categorical data and less informative or redundant columns
    trainData = removevars(trainData, columns_to_remove);
    
    % Remove outliers (top 1.5%) for each numerical feature
    if flag
        disp('Removing outliers (top 1.5%) from numerical features...');
        for i = 1:length(numericalCols)
            col = numericalCols{i};

            % Calculate the 98th percentile for the current column
            threshold = prctile(trainData.(col), 98.5);
            % Identify rows where the feature exceeds the threshold
            outlierIdx = trainData.(col) > threshold;
            % Display number of outliers removed for the feature
            disp(['Feature: ', col, ', Outliers Removed: ', num2str(sum(outlierIdx))]);
            % Remove outliers
            trainData(outlierIdx, :) = [];
        end
        % Display the size of the dataset after outlier removal
        disp(['Dataset size after outlier removal: ', num2str(height(trainData))]);
    end
    


    % Normalize numerical features
    trainData = normalize_table(trainData, numericalCols);

    % Return feature names (excluding target column)
    preprocessedTrain = trainData;
end

