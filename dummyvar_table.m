function tbl = dummyvar_table(tbl, categoricalVars)
% Converts categorical variables into dummy variables
    for i = 1:length(categoricalVars)
        catVar = categoricalVars{i};
        
        % Ensure the column is a categorical array
        if ~iscategorical(tbl.(catVar))
            tbl.(catVar) = categorical(tbl.(catVar));
        end
        
        % Get unique categories and create dummy variables
        categoriesList = categories(tbl.(catVar)); % Now it works as 'tbl.(catVar)' is categorical
        dummies = dummyvar(tbl.(catVar)); % Generate dummy variables
        dummyNames = strcat(catVar, '_', string(categoriesList)); % Create dummy variable names
        
        % Add dummy variables to the table
        tbl(:, dummyNames) = array2table(dummies, 'VariableNames', cellstr(dummyNames));
        
        % Remove the original categorical column
        tbl.(catVar) = [];
    end
end

