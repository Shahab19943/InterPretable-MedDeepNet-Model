%% Classification Using ASO-Selected MedDeepNet Features
% Author: Shahab Ul Hassan
% Purpose: Train and evaluate machine learning classifiers using
% ASO-selected features extracted from MedDeepNet.

clc;
clear;
close all;

%% =======================
% Random Seed
% =======================
rng(42);

%% =======================
% Load Selected Features
% =======================
load('sFeat_ASO.mat');          % Selected features after ASO
load('X.mat');                  % Ground-truth labels

features = sFeat(:,1:600);
labels = string(X);

%% =======================
% Train/Test Split
% =======================
cv = cvpartition(labels,'HoldOut',0.30);

trainIdx = training(cv);
testIdx  = test(cv);

XTrain = features(trainIdx,:);
XTest  = features(testIdx,:);

YTrain = labels(trainIdx);
YTest  = labels(testIdx);

%% =======================
% Classifier Library
% =======================
classifiers = {

templateSVM('KernelFunction','linear'),...
'Linear SVM';

templateSVM('KernelFunction','polynomial','PolynomialOrder',2),...
'Quadratic SVM';

templateSVM('KernelFunction','polynomial','PolynomialOrder',3),...
'Cubic SVM';

templateSVM('KernelFunction','gaussian','KernelScale','auto'),...
'Fine Gaussian SVM';

templateSVM('KernelFunction','gaussian','KernelScale',10),...
'Medium Gaussian SVM';

templateSVM('KernelFunction','gaussian','KernelScale',100),...
'Coarse Gaussian SVM';

templateKNN('NumNeighbors',1),...
'Fine KNN';

templateKNN('NumNeighbors',10),...
'Medium KNN';

templateKNN('NumNeighbors',50),...
'Coarse KNN';

templateKNN('NumNeighbors',10,...
'DistanceWeight','squaredinverse'),...
'Weighted KNN';

templateKNN('NumNeighbors',10,...
'Distance','cosine'),...
'Cosine KNN';

};

%% =======================
% Result Table
% =======================
results = table;

%% =======================
% Classification
% =======================
for k = 1:size(classifiers,1)

    fprintf('Running %s...\n',classifiers{k,2});

    tic
    model = fitcecoc(...
        XTrain,...
        YTrain,...
        'Learners',classifiers{k,1});
    trainTime = toc;

    tic
    [YPred,~] = predict(model,XTest);
    predictionTime = toc/numel(YTest);

    cm = confusionmat(YTest,YPred);

    TP = diag(cm);
    FP = sum(cm,1)' - TP;
    FN = sum(cm,2) - TP;
    TN = sum(cm(:)) - TP - FP - FN;

    accuracy = sum(TP)/sum(cm(:));
    precision = mean(TP./max(TP+FP,1));
    recall = mean(TP./max(TP+FN,1));
    specificity = mean(TN./max(TN+FP,1));
    f1 = mean(2*(precision*recall)/max(precision+recall,1));

    results = [results;

        table(...
        string(classifiers{k,2}),...
        accuracy,...
        precision,...
        recall,...
        specificity,...
        f1,...
        trainTime,...
        predictionTime,...
        'VariableNames',...
        {'Classifier',...
        'Accuracy',...
        'Precision',...
        'Recall',...
        'Specificity',...
        'F1Score',...
        'TrainingTime',...
        'PredictionTime'})];

end

%% =======================
% Display Results
% =======================
disp(results);

%% =======================
% Save Results
% =======================
writetable(results,...
    'Classification_Results.csv');
