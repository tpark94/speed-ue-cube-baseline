% TRAINODN  Train ODN
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Train ODN with YOLOv4 backbone. Note that the implementation here
%         is slightly different from the description in the paper, as it
%         uses MATLAB's built-in YOLOv4 functionalities instead.
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% REFERENCE:
% - https://www.mathworks.com/help/vision/ug/object-detection-using-yolov4-deep-learning.html
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------
clear all; close all; clc;
addpath(genpath('utils'));

% Inputs:
dataroot = 'D:\UE5Datasets\mathworks_training_v3_20230529';
chkpName = "odn_train_20230611_cubesatv3";

camerafn = fullfile(dataroot, '..\camera.json');
inputSize = [416, 416];

%% Load things & prepare Datastores
%  Load camera intrinsics & keypoints
camera = jsondecode(fileread(camerafn));
load("cubesatPoints.mat", "sat3dPoints");

% %  Uncomment to create CSV files if not already created
% createCSV(dataroot, camera, 'train');
% createCSV(dataroot, camera, 'validation');

%  Read CSVs
varNames = {'filename', 'xmin', 'ymin', 'xmax', 'ymax', ...
            'qw', 'qx', 'qy', 'qz', 'rx', 'ry', 'rz', ...
            'kx1', 'ky1', 'kx2', 'ky2', 'kx3', 'ky3', 'kx4', 'ky4', ...
            'kx5', 'ky5', 'kx6', 'ky6', 'kx7', 'ky7', 'kx8', 'ky8', ...
            'kx9', 'ky9', 'kx10', 'ky10', 'kx11', 'ky11'};
varTypes = {'char'}; [varTypes{2:34}] = deal('double');
opts = delimitedTextImportOptions('VariableNames', varNames, 'VariableTypes', varTypes);

csvTrain = readtable(fullfile(dataroot, "labels_krn", "train.csv"),      opts);
csvVal   = readtable(fullfile(dataroot, "labels_krn", "validation.csv"), opts);

%  Images
imdsTrain = imageDatastore(fullfile(dataroot, 'images', csvTrain.filename));
imdsVal   = imageDatastore(fullfile(dataroot, 'images', csvVal.filename));

%  Bbox is given in [xmin, ymin, xmax, ymax] (pix)
%  Convert to [xmin, ymin, w, h] (pix) format
bboxTrain        = table2array(csvTrain(:,2:5));        % [xmin, ymin, xmax, ymax] (pix)
bboxTrain(:,3:4) = bboxTrain(:,3:4) - bboxTrain(:,1:2); % [xmin, ymin, w, h] (pix)
bboxVal          = table2array(csvVal(:,2:5));          % [xmin, ymin, xmax, ymax] (pix)
bboxVal(:,3:4)   = bboxVal(:,3:4) - bboxVal(:,1:2);     % [xmin, ymin, w, h] (pix)

%  Convert to table
%  NOTE: If put a dummy sample with two objects just to make sure
%  `cell2table` function converts cell array into table of cells.
bboxTrainCell        = num2cell(bboxTrain, 2);
bboxTrainCell{end+1} = rand(2,4);
bboxTrainTable       = cell2table(bboxTrainCell, "VariableNames", "Cubesat"); 
bboxTrainTable       = bboxTrainTable(1:length(bboxTrain),:);

bboxValCell        = num2cell(bboxVal, 2);
bboxValCell{end+1} = rand(2,4);
bboxValTable       = cell2table(bboxValCell, "VariableNames", "Cubesat"); 
bboxValTable       = bboxValTable(1:length(bboxVal),:);

%  To boxLabelDatastore
bboxTrain = boxLabelDatastore(bboxTrainTable);
bboxVal   = boxLabelDatastore(bboxValTable);

%   Combine
trainingDataRaw    = combine(imdsTrain, bboxTrain);
valindationDataRaw = combine(imdsVal, bboxVal);

%% Data augmentation pipeline 
%  Pre-process
trainingDataProcess   = transform(trainingDataRaw,    @(data) preprocessODN(data, inputSize));
validationDataProcess = transform(valindationDataRaw, @(data) preprocessODN(data, inputSize));

%  Random data augmentation (50% prob. for each)
augTrainingData = transform(trainingDataProcess, @(data) randomAugTrainBbox(data, 0.5));

% %% (Optional) Validate processing
% dataOrg = read(trainingDataProcess);
% dataAug = read(augTrainingData);
% figure;
% while true
%     montage({insertShape(dataOrg{1}, "rectangle", dataOrg{2}), ...
%              insertShape(dataAug{1}, "rectangle", dataAug{2})});
% 
%     waitforbuttonpress;
%     dataOrg = read(trainingDataProcess);
%     dataAug = read(augTrainingData);
%     clf;
% end
% close; 
% reset(augTrainingData);
% reset(trainingDataProcess);

%% Anchor boxes
%  Estimate 6 anchor boxes from the training data
[anchors, meanIoU] = estimateAnchorBoxes(bboxTrain, 6);

%  Sort anchors based on IoU
area = anchors(:, 1) .* anchors(:,2);
[~,idx] = sort(area, "descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:) ; anchors(4:6,:)};

%% Create & train ODN
checkpointPath = fullfile("checkpoints", chkpName);
if ~isfolder(checkpointPath); mkdir(checkpointPath); end

%  Build
odn = yolov4ObjectDetector("tiny-yolov4-coco", "Cubesat", anchorBoxes, InputSize=[inputSize, 3]);

%  Configuration
options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    MiniBatchSize=48, ...
    Shuffle="every-epoch", ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ... % epoch
    LearnRateDropFactor=0.98, ...
    L2Regularization=5e-5, ...
    ResetInputNormalization=false, ...
    BatchNormalizationStatistics="moving", ...
    GradientThreshold=1.0, ...
    GradientThresholdMethod="l2norm", ...
    CheckpointPath=checkpointPath, ...
    CheckpointFrequency=1, ...
    CheckpointFrequencyUnit="epoch", ...
    ValidationData=[],...
    ValidationFrequency=1, ... % iteration, but epochs for custom train
    OutputNetwork="last-iteration", ...
    Plots="training-progress", ...
    VerboseFrequency=100, ...
    ExecutionEnvironment='auto', ...
    DispatchInBackground=true);

%% Train 
[odn, info] = trainYOLOv4ObjectDetector(augTrainingData, odn, options);

% Test & Analyses
detectionResults = detect(odn, validationDataProcess);

[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, validationDataProcess);
figure
plot(recall, precision, 'k', 'lineWidth', 2.0);
xlabel("Recall", 'FontSize', 16)
ylabel("Precision", 'FontSize', 16)
grid on
title(sprintf("Average Precision = %.2f",ap), 'FontSize', 16)
ylim([0.9, 1]);




