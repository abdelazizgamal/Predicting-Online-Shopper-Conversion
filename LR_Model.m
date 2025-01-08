% % Load Dataset
% data = readtable('online_shoppers_intention.csv'); % Replace with actual dataset file path
% % 
% % Split data into train and test (7:3 split) and save to CSV files
% disp('Splitting data into training and testing sets...');
% rng(42); % Set seed for reproducibility
% [trainIdx, testIdx] = crossvalind('HoldOut', height(data), 0.3);
% trainData = data(trainIdx, :);
% testData = data(testIdx, :);
% % 
% % % Save train and test datasets to CSV
% writetable(trainData, 'train_data.csv');
% writetable(testData, 'test_data.csv');
% disp('Train and test datasets saved as train_data.csv and test_data.csv.');

trainData = readtable('train_data.csv');
testData = readtable('test_data.csv');
targetCol = 'Revenue';% Target variable
% Preprocess the train dataset
disp('Preprocessing the train dataset...');
preprocessedTrain = preprocess_LR(trainData, targetCol);
feature_names = preprocessedTrain.Properties.VariableNames(~ismember(preprocessedTrain.Properties.VariableNames, targetCol));
% Preprocess the test dataset
disp('Preprocessing the test dataset...');
preprocessedTest = preprocess_LR(testData, targetCol);

% Extract features (X) and target (y) for training and testing
X_train = table2array(preprocessedTrain(:, ~ismember(preprocessedTrain.Properties.VariableNames, targetCol)));
y_train = table2array(preprocessedTrain(:, targetCol));
X_test = table2array(preprocessedTest(:, ~ismember(preprocessedTest.Properties.VariableNames, targetCol)));
y_test = table2array(preprocessedTest(:, targetCol));

disp(['Class 0 (majority): ', num2str(sum(y_train == 0))]);
disp(['Class 1 (minority): ', num2str(sum(y_train == 1))]);
% Identify minority and majority classes
minorityClass = find(y_train == 1);
majorityClass = find(y_train == 0);
% % Set a fixed random seed for reproducibility
rng(42);
% % % Undersample the minority class
undersampleIdx = randsample(majorityClass, length(minorityClass), false);
X_undersampled = [X_train(undersampleIdx, :); X_train(minorityClass, :)];  % Combine the undersampled majority and the minority class
y_undersampled = [y_train(undersampleIdx); y_train(minorityClass)];  % Corresponding target values
% Verify the class distribution after Undersampling
disp('Class distribution after undersampled:');
disp(['Class 0 (majority): ', num2str(sum(y_undersampled == 0))]);
disp(['Class 1 (minority): ', num2str(sum(y_undersampled == 1))]);



% Logistic Regression and GridSearchCV
disp('Training Logistic Regression model...');
py.importlib.import_module('sklearn.linear_model');
py.importlib.import_module('sklearn.model_selection');
py.importlib.import_module('sklearn.metrics');

logreg = py.sklearn.linear_model.LogisticRegression(pyargs('max_iter', 2000));
param_grid = py.dict(pyargs('penalty', {'l2','l1'}, 'C', py.list([0.001, 0.01, 0.1, 1]), 'solver', {'liblinear', 'saga'}));

grid_search = py.sklearn.model_selection.GridSearchCV(logreg, param_grid, ...
    pyargs('cv', int32(10), 'scoring', 'accuracy', 'return_train_score', py.True));

% Start timer for GridSearchCV
tic;
grid_search.fit(py.numpy.array(X_undersampled), py.numpy.array(y_undersampled));
elapsed_time_cv = toc;
disp(['Time taken for GridSearchCV: ', num2str(elapsed_time_cv), ' seconds']);

% Get best parameters and performance metrics
best_params = py.getattr(grid_search, 'best_params_');
best_score = py.getattr(grid_search, 'best_score_');
disp(['Best Parameters: ', char(best_params)]);
disp(['Best Cross-Validation Accuracy: ', num2str(best_score)]);

% Final Model Evaluation
best_model = py.getattr(grid_search, 'best_estimator_');
y_pred_prob = double(best_model.predict_proba(X_test));
y_pred_prob = y_pred_prob(:,2);
y_pred = double(y_pred_prob > 0.5);
y_test_v = double(y_test);

% Confusion Matrix and Metrics
confMatrix = confusionmat(y_test_v, y_pred);
accuracy = sum(diag(confMatrix)) / sum(confMatrix(:));
precision = confMatrix(2, 2) / sum(confMatrix(:, 2));
recall = confMatrix(2, 2) / sum(confMatrix(2, :));
f1_score = 2 * (precision * recall) / (precision + recall);
[XV, YV, ~, AUC] = perfcurve(y_test_v, y_pred_prob, 1);

% Display Metrics
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F1-Score: ', num2str(f1_score)]);
disp(['AUC: ', num2str(AUC)]);

% Plot Metrics
figure;
heatmap(confMatrix, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', ...
    'YLabel', 'Actual', 'Colormap', parula, 'ColorbarVisible', 'on');

figure;
plot(XV, YV, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');

% Coefficients and Visualization
coefficients = double(py.array.array('d', py.numpy.nditer(best_model.coef_)));
coeff_table = table(feature_names', coefficients', 'VariableNames', {'Feature', 'Coefficient'});
% Sort the table for better visualization (optional)
coeff_table = sortrows(coeff_table, 'Coefficient', 'descend');
disp('Feature Coefficients:');
disp(coeff_table);

figure;
bar(coeff_table.Coefficient);
set(gca, 'XTick', 1:numel(coeff_table.Feature), 'XTickLabel', coeff_table.Feature, 'XTickLabelRotation', 45);
ylabel('Coefficient Value');
title('Feature Coefficients');
