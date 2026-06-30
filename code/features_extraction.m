%% Feature Extraction from MedDeepNet

clc; clear; close all;

rng(42);

load MedDeepNet.mat
net = netT;

datasetPath = 'dataset/path';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

inputSize = net.Layers(1).InputSize;

augImds = augmentedImageDatastore(inputSize(1:2), imds);

% Extract features from FC layer
layerName = 'fc_1';

features = activations(net, augImds, layerName, 'OutputAs','rows');
labels = imds.Labels;

save('orgfeaturesTrain.mat','features','labels');
