% Author: Tahmid Mehdi
% Performs and tests k-means clustering 

% Performs k-means algorithm to classify data in X and calculates accuracy rate and time elapsed.
% PRE: X is the matrix of inputs
%      Y is the vector of class attributes
%      k is the number of clusters (int >0)
%      folds is the number of tests to perform (int >=1)
function [accuracy, time] = TestKMeans(X, Y, k, folds)
    rng(1); % ensures replication
    m = size(X, 1);
    accuracy = zeros(folds);
    
    for test = 1:folds
        result = kmeans(X, k);
        % finds accuracy by pairwise agreement among data points
        if k==2
            correctPairs = 0;
            for i = 1:m-1
                for j = (i+1):m
                    correctPairs = correctPairs + ((Y(i)==Y(j))==(result(i)==result(j)));
                end
            end
            accuracy = correctPairs/(0.5*m*(m-1));
        % finds rand index which is an accuracy measure
        else
            accMetrics = PartAgreeCoef(result, Y);
            accuracy = accMetrics.ri;
        end
    end
    % finds time
    f = @() kmeans(X,k);
    time = timeit(f);
end