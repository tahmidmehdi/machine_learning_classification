# machine_learning_classification
SVM, K-Means and FCM

TrainTestSVM (Support Vector Machine)
Tahmid Mehdi
MATLAB
Trains data and performs k-fold cross-validation with 30% holdout to calculate error. It supports binary and multiclass classification.
https://github.com/tahmidmehdi/machine_learning_classification
June 2015

TestKMeans (K-Means)
Tahmid Mehdi
MATLAB
Performs k-means clustering on data and calculates accuracy using rand index or pairwise agreement.
https://github.com/tahmidmehdi/machine_learning_classification
June 2015

TestFCM (Fuzzy C-Means)
Tahmid Mehdi
MATLAB
Performs fuzzy c-means clustering on data and calculates accuracy using rand index or pairwise agreement.
https://github.com/tahmidmehdi/machine_learning_classification
June 2015

Special thanks to Cody Neuburger and Comparing Partitions.
The following functions are required but not provided in this repo:

PartAgreeCoef function was provided by UMMI@IMM and is available at http://darwin.phyloviz.net/ComparingPartitions/PartAgreeCoef.m
Copyright (C) 2009  UMMI@IMM
This file is part of Comparing Partitions website <http://www.comparingpartitions.info/>.    
Comparing Partitions website is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

multisvm function was written by 
Cody Neuburger cneuburg@fau.edu
Florida Atlantic University, Florida USA
This code was adapted and cleaned from Anand Mishra's multisvm function
found at http://www.mathworks.com/matlabcentral/fileexchange/33170-multi-class-support-vector-machine/
