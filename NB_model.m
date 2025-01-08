trainData = readtable('train_data.csv');
testData = readtable('test_data.csv');
% Preprocess the train dataset
targetCol = 'Revenue';
disp('Preprocessing the train dataset...');
preprocessedTrain = preprocess_NB(trainData, 1, targetCol);
feature_names = preprocessedTrain.Properties.VariableNames(~ismember(preprocessedTrain.Properties.VariableNames, targetCol));

% Preprocess the test dataset
disp('Preprocessing the test dataset...');
preprocessedTest = preprocess_NB(testData, 0, targetCol);

% Extract features (X) and target (y) for training and testing
X_train = table2array(preprocessedTrain(:, ~ismember(preprocessedTrain.Properties.VariableNames, targetCol)));
y_train = table2array(preprocessedTrain(:, targetCol));
X_test = table2array(preprocessedTest(:, ~ismember(preprocessedTest.Properties.VariableNames, targetCol)));
y_test = table2array(preprocessedTest(:, targetCol));

%Apply UnderSampling
% Identify minority and majority classes
minorityClass = find(y_train == 1);
majorityClass = find(y_train == 0);
% % Set a fixed random seed for reproducibility
rng(42);
undersampleIdx = randsample(majorityClass, length(minorityClass)*2, false);%undersampling with 2 X length of minorityClass
X_undersampled = [X_train(undersampleIdx, :); X_train(minorityClass, :)];  % Combine the undersampled majority and the minority class
y_undersampled = [y_train(undersampleIdx); y_train(minorityClass)];  % Corresponding target values
%Verify the class distribution after Underssampling
disp('Class distribution after undersampled:');
disp(['Class 0 (majority): ', num2str(sum(y_undersampled == 0))]);
disp(['Class 1 (minority): ', num2str(sum(y_undersampled == 1))]);


% Import necessary Python modules for Naive Bayes and GridSearchCV
disp('Training Gaussian Naive Bayes model...');
py.importlib.import_module('sklearn.naive_bayes');
py.importlib.import_module('sklearn.model_selection');
py.importlib.import_module('sklearn.metrics');

% Initialize Gaussian Naive Bayes
gnb = py.sklearn.naive_bayes.GaussianNB();
% Define hyperparameter grid for var_smoothing
param_grid_NB = py.dict(pyargs('var_smoothing', py.list([1e-9, 1e-8, 1e-7, 1e-6, 1e-5,0.1,0.01,0.001,0.0001])));

% Set up GridSearchCV with 10-fold cross-validation
disp('Performing GridSearch with 10-fold Cross-Validation...');
grid_search_NB = py.sklearn.model_selection.GridSearchCV(gnb, param_grid_NB, ...
    pyargs('cv', int32(10), 'scoring', 'accuracy', 'return_train_score', py.True));

% Start timer for GridSearchCV
tic;
grid_search_NB.fit(py.numpy.array(X_undersampled), py.numpy.array(y_undersampled));
elapsed_time_cv = toc;
disp(['Time taken for GridSearchCV: ', num2str(elapsed_time_cv), ' seconds']);

% Get best parameters and performance metrics
best_params_NB = py.getattr(grid_search_NB, 'best_params_');
best_score_NB = py.getattr(grid_search_NB, 'best_score_');
disp(['Best Parameters: ', char(best_params_NB)]);
disp(['Best Cross-Validation accuracy_NB: ', num2str(best_score_NB)]);

% Final Model Evaluation
best_model_NB = py.getattr(grid_search_NB, 'best_estimator_');
y_pred_prob_NB = double(best_model_NB.predict_proba(X_test));
y_pred_prob_NB =y_pred_prob_NB(:,2);
y_pred_NB = double(y_pred_prob_NB > 0.5);
y_test_v_NB = double(y_test);

% Confusion Matrix and Metrics
confMatrix_NB = confusionmat(y_test_v_NB, y_pred_NB);
accuracy_NB = sum(diag(confMatrix_NB)) / sum(confMatrix_NB(:));
precision_NB = confMatrix_NB(2, 2) / sum(confMatrix_NB(:, 2));
recall_NB = confMatrix_NB(2, 2) / sum(confMatrix_NB(2, :));
f1_score_NB = 2 * (precision_NB * recall_NB) / (precision_NB + recall_NB);
[XV_NB, YV_NB, ~, AUC_NB] = perfcurve(y_test_v_NB, y_pred_prob_NB, 1);

% Calculate conditional probabilities for key features
conditional_probs = [];
for i = 1:length(feature_names)
    col = feature_names{i};
    prob_class1 = mean(preprocessedTest.(col)(preprocessedTest.Revenue == 1)); % P(Feature | Revenue=1)
    prob_class0 = mean(preprocessedTest.(col)(preprocessedTest.Revenue == 0)); % P(Feature | Revenue=0)
    conditional_probs = [conditional_probs; prob_class1, prob_class0];
    disp(['Feature: ', col, ', P(Feature | Class=1): ', num2str(prob_class1), ...
          ', P(Feature | Class=0): ', num2str(prob_class0)]);
end

% Display Metrics
disp(['accuracy_NB: ', num2str(accuracy_NB)]);
disp(['precision_NB: ', num2str(precision_NB)]);
disp(['recall_NB: ', num2str(recall_NB)]);
disp(['F1-Score: ', num2str(f1_score_NB)]);
disp(['AUC_NB: ', num2str(AUC_NB)]);

% Plot Metrics
figure;
heatmap(confMatrix_NB, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', ...
    'YLabel', 'Actual', 'Colormap', parula, 'ColorbarVisible', 'on');

figure;
plot(XV_NB, YV_NB, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');

% Create a bar chart for  probabilities for key features
figure;
bar(conditional_probs, 'grouped');
xticks(1:length(feature_names));
xticklabels(feature_names);
xtickangle(45);
ylabel('Conditional Probability');
title('Conditional Probabilities of Features Given Target Class');
legend({'P(Feature | Revenue=1)', 'P(Feature | Revenue=0)'}, 'Location', 'northwest');
grid on;

