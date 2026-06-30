%% LIME Explanation Script
% Author: Shahab Ul Hassan
% Purpose: Generate LIME explanations for MedDeepNet predictions.

clc;
clear;
close all;

%% =======================
% Random Seed
% =======================
rng(42);

%% =======================
% Load Image
% =======================
I = imread('image_name.jpg');          % Replace with your test image
I = imresize(I,[227 227]);

%% =======================
% Load Trained MedDeepNet
% =======================
load('MedDeepNet.mat');
net = netT;

%% =======================
% Prediction
% =======================
predictedLabel = classify(net,I);

classNames = net.Layers(end).Classes;
classIndex = find(classNames == predictedLabel);

fcLayers = find(arrayfun(@(x) isa(x,'nnet.cnn.layer.FullyConnectedLayer'),net.Layers));
targetLayer = net.Layers(fcLayers(end)).Name;

fprintf('Predicted Class: %s\n',char(predictedLabel));

%% =======================
% Adaptive Superpixel Count
% =======================
pixelCount = size(I,1)*size(I,2);
numSuperpixels = min(100,max(50,round(pixelCount/200)));

labImage = rgb2lab(I);

[L,N] = superpixels(labImage,...
    numSuperpixels,...
    'Compactness',20,...
    'IsInputLab',true);

%% =======================
% Perturbation Generation
% =======================
numSamples = 1000;

images = zeros(227,227,3,numSamples,'uint8');
similarity = zeros(numSamples,1);
indices = zeros(numSamples,N);

perturbLevels = linspace(0.1,0.9,5);

for k = 1:numSamples

    level = perturbLevels(mod(k-1,length(perturbLevels))+1);

    p = level + 0.1*randn;
    p = min(max(p,0.05),0.95);

    active = rand(N,1) > p;

    mask = zeros(size(I,1),size(I,2));

    for s = find(active)'
        mask = mask + (L==s);
    end

    images(:,:,:,k) = uint8(I .* uint8(mask));

    similarity(k) = 1 - nnz(active)/N;

    indices(k,:) = active;

end

%% =======================
% Network Prediction on Perturbations
% =======================
predictionScores = activations(net,...
    images,...
    targetLayer,...
    'OutputAs','rows');

predictionScores = predictionScores(:,classIndex);

%% =======================
% Local Surrogate Model
% =======================
sigma = 0.25*std(similarity);

weights = exp(-(similarity.^2)/(sigma^2));

model = fitrlinear(...
    indices,...
    predictionScores,...
    'Learner','leastsquares',...
    'Weights',weights,...
    'Lambda',0.1,...
    'Regularization','ridge');

importance = model.Beta;

%% =======================
% Lung Region Mask
% =======================
gray = rgb2gray(I);

lungMask = imbinarize(gray,'adaptive');
lungMask = imfill(lungMask,'holes');
lungMask = bwareafilt(lungMask,1);

%% =======================
% Explanation Map
% =======================
explanation = zeros(size(L));

for s = 1:N

    explanation = explanation + ...
        (L==s).*max(importance(s),0);

end

%% =======================
% Adaptive Smoothing
% =======================
sigmaSmooth = 5 + 2*std(explanation(:));

explanation = imgaussfilt(explanation,sigmaSmooth);

explanation = mat2gray(explanation);

explanation = explanation .* double(lungMask);

%% =======================
% Visualization
% =======================
figure;

imshow(I);

hold on;

imagesc(explanation,...
    'AlphaData',0.5);

colormap jet

colorbar

title(['ASP-LIME Explanation : ',char(predictedLabel)])

hold off;

%% =======================
% Faithfulness Evaluation
%% =======================
% Deletion and insertion metrics can be computed by progressively
% removing or inserting the highest-ranked superpixels according
% to the learned importance weights and recording the model score.
%
% This implementation follows the evaluation protocol reported
% in the manuscript.