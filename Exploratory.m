% Load Dataset
data = readtable('online_shoppers_intention.csv'); % Replace with actual dataset file path

% Preprocessing
disp('Preprocessing the dataset...');
targetCol = 'Revenue'; % Target variable
numericalCols = {'Administrative', 'Informational','ProductRelated', ...
    'ProductRelated_Duration','Informational_Duration','Administrative_Duration' ...
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'};


% Convert the TargetCol categorical 'true' and 'false' to numeric 1 and 0
data.Revenue = strcmp(data.Revenue, 'TRUE'); % 'true' -> 1, 'false' -> 0

% Convert categorical features to one-hot encoding
categoricalCols = {'Month', 'VisitorType', 'Weekend'};
data = dummyvar_table(data, categoricalCols);

% Visualize class imbalance
figure;
histogram(data.Revenue, 'Normalization', 'count', 'FaceColor', 'blue');
title('Class Distribution (Revenue)');
xlabel('Class (0: No, 1: Yes)');
ylabel('Count');

% Initialize a table to store correlations
correlationTable = table();
% Compute correlation for each predictor column
Cols = setdiff(data.Properties.VariableNames, targetCol); % Exclude target column from features
allCols = [Cols, targetCol];  % Combine features and target column
correlationMatrix = corr(table2array(data(:, allCols)));
% Create a heatmap of the correlation matrix
figure;
heatmap(allCols, allCols, correlationMatrix, ...
    'Title', 'Correlation Heatmap', ...
    'XLabel', 'Features', ...
    'YLabel', 'Features', ...
    'Colormap', parula, ...
    'ColorbarVisible', 'on');

% Define the list of features to plot
numFeatures = length(numericalCols); % Total number of features
featuresPerRow = 5; % Number of features per row
numRows = ceil(numFeatures / featuresPerRow);

% Create figure for box plots
figure;
for i = 1:numFeatures
    % Subplot for each numerical feature
    subplot(numRows, featuresPerRow, i);
    boxplot(data.(numericalCols{i}), 'Notch', 'on');
    title(numericalCols{i});
    ylabel('Value');
    set(gca, 'XTickLabel', {});
end
% Add a title to the entire figure
sgtitle('Box Plots of Numerical Features');

% Separate the data by class (Revenue = 0 and Revenue = 1)
data_class_0 = data(data.Revenue == 0, :); % Class 0 (No Purchase)
data_class_1 = data(data.Revenue == 1, :); % Class 1 (Purchase)
% Create figure for histograms
figure;
for i = 1:numFeatures
    subplot(numRows, featuresPerRow, i);
    % Plot histograms for each class
    histogram(data_class_0.(numericalCols{i}), 'Normalization', 'probability', ...
        'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Class 0 (No Purchase)');
    hold on;
    histogram(data_class_1.(numericalCols{i}), 'Normalization', 'probability', ...
        'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Class 1 (Purchase)');
    hold off;
    title(numericalCols{i});
    xlabel('Value');
    ylabel('Probability');
    legend('show', 'Location', 'northeast');
end
sgtitle('Histograms of Features');

% Remove outliers (top 1.5%) for each numerical feature
disp('Removing outliers (top 1.5%) from numerical features...');
for i = 1:length(numericalCols)
    col = numericalCols{i};
    threshold = prctile(data.(col), 98.5);
    outlierIdx = data.(col) > threshold;
    disp(['Feature: ', col, ', Outliers Removed: ', num2str(sum(outlierIdx))]);
    
    % Remove outliers
    data(outlierIdx, :) = [];
end

% Display the size of the dataset after outlier removal
disp(['Dataset size after outlier removal: ', num2str(height(data))]);

% Normalize numerical features
numericalCols = {'Administrative', 'Informational','ProductRelated', ...
    'ProductRelated_Duration','Informational_Duration','Administrative_Duration' ...
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'};
data = normalize_table(data, numericalCols);


% Separate the data by class (Revenue = 0 and Revenue = 1)
data_class_0 = data(data.Revenue == 0, :); % Class 0 (No Purchase)
data_class_1 = data(data.Revenue == 1, :); % Class 1 (Purchase)
% Create figure for histograms
figure;
for i = 1:numFeatures
    % Subplot for each numerical feature
    subplot(numRows, featuresPerRow, i);
    
    % Plot histograms for each class
    histogram(data_class_0.(numericalCols{i}), 'Normalization', 'probability', ...
        'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Class 0 (No Purchase)');
    hold on;
    histogram(data_class_1.(numericalCols{i}), 'Normalization', 'probability', ...
        'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Class 1 (Purchase)');
    hold off;
    
    % Add labels and title
    title(numericalCols{i});
    xlabel('Value');
    ylabel('Probability');
    legend('show', 'Location', 'northeast');
end
% Add a title to the entire figure
sgtitle('Class-Specific Histograms for Numerical Features');
