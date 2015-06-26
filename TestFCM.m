% Author: Tahmid Mehdi
% Performs and tests fuzzy c-means clustering 

% Performs fcm algorithm to classify data in X and calculates accuracy rate and time elapsed.
% PRE: X is the matrix of inputs
%      Y is the vector of class attributes
%      c is the number of clusters (int >0)
%      folds is the number of tests to perform (int >=1)
function [accuracy, time] = TestFCM(X, Y, c, folds)
    opts = [nan;nan;nan;0];
    accuracy = zeros(folds);
    m = size(X, 1);
    
    for test = 1:folds
        [~, U, ~] = fcm(X, c, opts);
        % result is a vector with the ith entry equalling the class which
        % it has the highest degree of membership in 
        [~, result] = max(U);
        % finds accuracy by pairwise agreement among data points
        if c==2
            correctPairs = 0;
            for i = 1:m-1
                for j = (i+1):m
                    correctPairs = correctPairs + ((Y(i)==Y(j))==(result(i)==result(j)));
                end
            end
            accuracy = correctPairs/(0.5*m*(m-1));
        % finds rand index which is an accuracy measure
        else
            accMetrics = PartAgreeCoef(result', Y);
            accuracy = accMetrics.ri;
        end
    end
    % finds time
    f = @() fcm(X,c);
    time = timeit(f);
end