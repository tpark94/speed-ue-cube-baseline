% TRAINKRN  Train KRN
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Train KRN with YOLOv3 backbone. Closely follows the description
%         in the paper.
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% REFERENCE: https://arxiv.org/abs/1909.00392
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------
clear all; close all; clc
addpath(genpath('utils'));

% Inputs:
dataroot = "D:\UE5Datasets\mathworks_training_v3_20230529";
chkpName = "krn_train_20230608_cubesatv3";

camerafn = fullfile(dataroot, '../camera.json');
inputSize = [224, 224];

%% Load things & prepare Datastores
%  Load camera intrinsics & keypoints
camera = jsondecode(fileread(camerafn));
camera.distCoeffs = zeros(5, 1);

load("cubesatPoints.mat", "sat3dPoints");

% %  Uncomment to create CSV files if not already created
% createCSV(dataroot, camera, 'train');
% createCSV(dataroot, camera, 'test');

%% Create datastore's
%  Read CSVs
varNames = {'filename', 'xmin', 'ymin', 'xmax', 'ymax', ...
            'qw', 'qx', 'qy', 'qz', 'rx', 'ry', 'rz', ...
            'kx1', 'ky1', 'kx2', 'ky2', 'kx3', 'ky3', 'kx4', 'ky4', ...
            'kx5', 'ky5', 'kx6', 'ky6', 'kx7', 'ky7', 'kx8', 'ky8', ...
            'kx9', 'ky9', 'kx10', 'ky10', 'kx11', 'ky11'};
varTypes = {'char'}; [varTypes{2:34}] = deal('double');
opts = delimitedTextImportOptions('VariableNames', varNames, 'VariableTypes', varTypes);

csvTrain = readtable(fullfile(dataroot, "labels_krn", "train.csv"), opts);
csvVal   = readtable(fullfile(dataroot, "labels_krn", "validation.csv"), opts);

%  Images
imdsTrain = imageDatastore(fullfile(dataroot, 'images', csvTrain.filename));
imdsVal   = imageDatastore(fullfile(dataroot, 'images', csvVal.filename));

%  Labels
%  - train: bbox & keypoints (bbox for processing only)
%  - validation: poses
bboxTrain = arrayDatastore(csvTrain(:, 2:5)); % [xmin, ymin, xmax, ymax] (pix)
kptsTrain = arrayDatastore(csvTrain(:, 13:34));
bboxVal   = arrayDatastore(csvVal(:, 2:5));
posesVal  = arrayDatastore(csvVal(:, 6:12));

%  Combine
trainingDataRaw    = combine(imdsTrain, bboxTrain, kptsTrain);
valindationDataRaw = combine(imdsVal, bboxVal, posesVal);

%% Data augmentation pipeline 

trainingDataProcess   = transform(trainingDataRaw,    @(data) preprocessKRN(data, inputSize, true));
validationDataProcess = transform(valindationDataRaw, @(data) preprocessKRN(data, inputSize, false));

% Random data augmentation to perform during training:
augTrainingData = transform(trainingDataProcess, @(data) randomAugTrainKpt(data, 0.5));

%% Validate processing
% figure
% while true
%     data = read(augTrainingData);
%     kpts = reshape(data{2}, 2, 11) * 224;
%     
%     imshow(data{1}); hold on
%     scatter(kpts(1,:), kpts(2,:), 'gx')
% 
%     waitforbuttonpress
%     clf
% end
% reset(augTrainingData);

%% Create & train KRN
checkpointPath = fullfile("checkpoints", chkpName);
if ~isfolder(checkpointPath); mkdir(checkpointPath); end

%  Build
inputSizeRGB = [inputSize, 3];
krnGraph = buildKRN(11, inputSizeRGB);

%  Configuration
options = trainingOptions("adam", ...
    MaxEpochs=150, ...
    MiniBatchSize=48, ...
    Shuffle="every-epoch", ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=1, ... % epoch
    LearnRateDropFactor=0.96, ...
    L2Regularization=5e-5, ...
    ResetInputNormalization=false,...
    BatchNormalizationStatistics="moving", ...
    GradientThreshold=1.0, ...
    GradientThresholdMethod="global-l2norm", ...
    CheckpointPath=checkpointPath, ...
    CheckpointFrequency=1, ...
    CheckpointFrequencyUnit="epoch", ...
    ValidationData=[],...
    ValidationFrequency=2500, ... % iteration, but epochs for custom train
    OutputNetwork="last-iteration", ...
    Plots="training-progress", ...
    VerboseFrequency=100, ...
    ExecutionEnvironment='auto', ...
    DispatchInBackground=true);

%% Train 
net = trainNetwork(augTrainingData, krnGraph, options);

%% Test
figure
while true
    data = read(validationDataProcess);
    out  = predict(net, data{1});
    kpts = reshape(out, 2, 11) * 224;
    
    imshow(data{1}); hold on
    scatter(kpts(1,:), kpts(2,:), 'gx')

    waitforbuttonpress
    clf
end
reset(validationDataProcess);




