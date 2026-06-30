%% Feature Extraction + ASO Feature Selection
% Author: Shahab Ul Hassan
% Purpose: Extract deep features from MedDeepNet and apply ASO selection
% for reproducible lung CT classification pipeline

clc;
clear;
close all;

%% =======================
% Reproducibility
% =======================
rng(42,'twister');   % Fixed seed for reproducibility

%% =======================
% Load Dataset
% =======================
datasetPath = 'dataset/Modified_Dataset';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% =======================
% Load Trained MedDeepNet
% =======================
load('MedDeepNet.mat');   % contains netT
net = netT;

%% =======================
% Input Size
% =======================
inputSize = net.Layers(1).InputSize;

%% =======================
% Preprocess Images
% =======================
augImds = augmentedImageDatastore(inputSize(1:2), imds);

%% =======================
% Feature Extraction (FC layer)
% =======================
featureLayer = 'fc_1';

orgfeaturesTrain = activations(net, augImds, featureLayer, ...
    'OutputAs','rows');

Y = imds.Labels;

save('orgfeaturesTrain.mat','orgfeaturesTrain','Y');

%% =======================
% Convert Labels
%% =======================
label = string(Y);

feat = orgfeaturesTrain;

%% =======================
% Hold-out Split (ASO)
% =======================
ho = 0.2;

HO = cvpartition(label, 'HoldOut', ho, 'Stratify', false);

save('ASO_split.mat','HO');

%% =======================
% ASO Parameters
% =======================
N        = 10;
max_Iter = 25;
alpha    = 50;
beta     = 0.2;


%% =======================
% Atom Search Optimization
% =======================
[sFeat, Sf, Nf, curve] = jASO(feat, label, N, max_Iter, alpha, beta, HO);


%% =======================
% Save Results
% =======================
save('sFeat_ASO.mat','sFeat','Sf','Nf','curve');

disp('Feature extraction + ASO completed successfully');