%% MedDeepNet Training Script
% Author: Shahab Ul Hassan
% Purpose: Reproducible training of MedDeepNet for lung CT classification

clc;
clear;
close all;

%% =======================
% Reproducibility
% =======================
rng(42);   % Fixed seed for reproducibility

%% =======================
% Load Dataset
% =======================
imds = imageDatastore('dataset/path', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% =======================
% Train/Validation Split
% =======================
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Save split for reproducibility (IMPORTANT for reviewer)
save('dataset_split.mat','imdsTrain','imdsValidation');

%% =======================
% Data Augmentation (Training only)
% =======================
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-3, 3], ...
    'RandYTranslation', [-3, 3], ...
    'RandScale', [0.9, 1.1]);

imdsTrainAug = augmentedImageDatastore([227 227], imdsTrain, ...
    'DataAugmentation', augmenter);

%% =======================
% Visualize Samples
% =======================
numTrainImages = numel(imdsTrain.Files);
idx = randperm(numTrainImages,16);

figure;
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end

%% =======================
% MedDeepNet Architecture
% =======================
layers = [
    imageInputLayer([227 227 3],"Name","imageinput")

    convolution2dLayer([13 13],64,"Padding","same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer

    convolution2dLayer([11 11],64,"Padding","same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer

    convolution2dLayer([11 11],96,"Padding","same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer

    convolution2dLayer([11 11],96,"Padding","same")
    batchNormalizationLayer
    reluLayer
    globalAveragePooling2dLayer

    convolution2dLayer([9 9],128,"Padding","same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([9 9],128,"Padding","same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([7 7],128,"Padding","same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([7 7],256,"Padding","same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([7 7],512,"Padding","same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([5 5],1024,"Padding","same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer([3 3],2048,"Padding","same")
    batchNormalizationLayer
    reluLayer

    dropoutLayer(0.5)

    globalAveragePooling2dLayer
    fullyConnectedLayer(2048)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

%% =======================
% Training Options
% =======================
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',300, ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize',128, ...
    'Momentum',0.9, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',50, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',50, ...
    'Verbose',false, ...
    'L2Regularization',1e-4, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto');

%% =======================
% Train Network
% =======================
netT = trainNetwork(imdsTrainAug, layers, options);

%% =======================
% Save Model
% =======================
save('MedDeepNet.mat','netT');

%% =======================
% Evaluation
% =======================
YPred = classify(netT, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation);
disp(['Validation Accuracy: ', num2str(accuracy)]);

figure;
plotconfusion(YValidation, YPred);