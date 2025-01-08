function tbl = normalize_table(tbl, numericalVars)
    % Normalizes numerical variables to have mean 0 and std 1
    for i = 1:length(numericalVars)
        numVar = numericalVars{i};
        tbl.(numVar) = (tbl.(numVar) - mean(tbl.(numVar))) / std(tbl.(numVar));
    end
end
