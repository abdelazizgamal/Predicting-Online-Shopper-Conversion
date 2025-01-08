function [preprocessedTrain] = preprocess_LR(trainData, targetCol)
    % Convert target column to numeric (TRUE -> 1, FALSE -> 0)
    trainData.(targetCol) = strcmp(trainData.(targetCol), 'TRUE');

    % Define columns to process
    categoricalCols = {'Month', 'VisitorType', 'Weekend'};
    % One-hot encode categorical features
    trainData = dummyvar_table(trainData, categoricalCols);
    
    columns_to_remove = {'Browser', 'OperatingSystems', 'Region', 'TrafficType', ...
        'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', ...
        'Month_Mar', 'Month_May', 'Month_Oct', 'Month_Sep', 'Weekend_FALSE'};

    % Remove less informative or redundant columns
    trainData = removevars(trainData, columns_to_remove);

    numericalCols = {'Administrative', 'Informational', 'ProductRelated', ...
        'ProductRelated_Duration', 'Informational_Duration', 'Administrative_Duration', ...
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'};
    % Normalize numerical features
    trainData = normalize_table(trainData, numericalCols);

    % Return feature names (excluding target column)
    preprocessedTrain = trainData;
end
