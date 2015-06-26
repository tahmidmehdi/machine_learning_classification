% Author: Tahmid Mehdi
% Trains data using SVM and performs k-fold cross-validation 

% Performs SVM algorithm to classify data in X and calculates error rate and time elapsed.
% PRE: X is the matrix of inputs
%      Y is the vector of class attributes
%      folds is the number of tests to perform (int >=1)
function [error, time] = TrainTestSVM(X, Y, folds)
    m = size(X, 1);
    error = zeros(folds);
    % binary classification
    if size(unique(Y),1) == 2
        for test = 1:folds
            CVSVMModel = fitcsvm(X,Y, 'Holdout', 0.3);
            error(test) = kfoldLoss(CVSVMModel);
        end
        % finds time elapsed
        f = @() fitcsvm(X,Y);
        time = timeit(f);
    % multiclass classification
    else
        for test = 1:folds
            % picks 30% of data for testing
            holdout = round(m*0.3);
            testIndices = randperm(m, holdout);
            dataTrain = [];
            dataTest = [];
            classTrain = [];
            classTest = [];
            for i = 1:m
                if any(i==testIndices)
                    dataTest = vertcat(dataTest, X(i, :));
                    classTest = vertcat(classTest, Y(i));
                else
                    dataTrain = vertcat(dataTrain, X(i, :));
                    classTrain = vertcat(classTrain, Y(i));
                end
            end

            result = multisvm(dataTrain, classTrain, dataTest);
            % calculates error rate
            misclassifications = 0;
            for j = 1:length(result)
                misclassifications = misclassifications + (result(j) ~= classTest(j));
            end
            error(test) = misclassifications/length(result);
        end
        f = @() multisvm(dataTrain, classTrain, dataTest);
        time = timeit(f);
    end
end