clc; clear; close all;

%% Input dataset path
inputPath = 'dataset/path';
outputPath = 'dataset/resized_augmented';

imds = imageDatastore(inputPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Resize size
targetSize = [227 227];
 
%%  train/valid split

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

%% Augmentation settings
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandScale', [0.9 1.1]);

augimds = augmentedImageDatastore(targetSize, imds, ...
    'DataAugmentation', augmenter);

%% Create output folder structure
classes = categories(imds.Labels);

for i = 1:numel(classes)
    mkdir(fullfile(outputPath, char(classes{i})));
end

%% Save augmented images
disp('Saving resized + augmented dataset...');

numAugPerImage = 1; % you can increase if needed
count = 1;

reset(augimds);

while hasdata(augimds)
    [img, info] = read(augimds);

    label = char(info.Label);

    saveName = fullfile(outputPath, label, ...
        sprintf('img_%05d.png', count));

    imwrite(img, saveName);
    count = count + 1;
end

disp('Stage 1 completed: resized + augmented dataset saved.');


 %% ---- SEGMENTATION ----
% Input / Output paths
inputPath = 'dataset/resized_augmented';
outputPath = 'dataset/segmented';

imds = imageDatastore(inputPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

classes = categories(imds.Labels);

for i = 1:numel(classes)
    mkdir(fullfile(outputPath, char(classes{i})));
end

%% Parameters
k = 8;
max_iter = 100;
tol = 1e-4;
weights = [0.5 0.25 0.25];

disp('Starting segmentation...');

reset(imds);

count = 1;

while hasdata(imds)

    I = read(imds);
    I = im2double(I);

    % RGB → Lab
    cform = makecform('srgb2lab');
    I_lab = applycform(I, cform);

    [m,n,c] = size(I_lab);
    pixels = reshape(I_lab, [], c);
    pixels = pixels ./ max(pixels(:));

    % Initialize centers
    rand_idx = randperm(size(pixels,1), k);
    centers = pixels(rand_idx,:);

    %% K-medians loop
    for iter = 1:max_iter

        wp = pixels .* weights;
        wc = centers .* weights;

        D = pdist2(wp, wc, 'cityblock');
        [~, labels] = min(D, [], 2);

        new_centers = zeros(k,c);

        for j = 1:k
            cluster = pixels(labels==j,:);
            if ~isempty(cluster)
                new_centers(j,:) = median(cluster,1);
            else
                new_centers(j,:) = centers(j,:);
            end
        end

        if max(abs(new_centers-centers),[],'all') < tol
            break;
        end

        centers = new_centers;
    end

    %% Create segmented image
    rng(42);
    colors = rand(k,3);

    seg = reshape(colors(labels,:), m, n, 3);

    %% Save output
    info = imds.Files{count};
    [~, name, ~] = fileparts(info);
    label = char(imds.Labels(count));

    saveName = fullfile(outputPath, label, [name '_seg.png']);
    imwrite(seg, saveName);

    count = count + 1;
end

disp('Stage 2 completed: segmented dataset saved.');
