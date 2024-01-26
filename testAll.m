% TESTALL  Test ODN + KRN in sequence
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Test both ODN and KRN in sequence, i.e., do ODN prediction first,
%         and use predicted bboxes from ODN to process images for KRN. PnP
%         is used to compute pose from predicted keypoints.
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------
clear all; close all; clc
addpath(genpath('utils'));

% Inputs:
dataroot = "D:\UE5Datasets\mathworks_traj_v2_20230628";
camerafn = fullfile(dataroot, '../camera.json');

% chkpt_odn = "odn_train_20230526_speed_b48_e100/net_checkpoint__25000__2023_05_27__01_52_39.mat";
% chkpt_krn = "krn_train_20230601_speed_b48_e300_mean_l2g_adam/net_checkpoint__75000__2023_06_05__23_12_09.mat";
chkpt_odn = "odn_train_20230611_cubesatv3\net_checkpoint__50000__2023_06_12__15_52_56.mat";
chkpt_krn = "krn_train_20230608_cubesatv3\net_checkpoint__75000__2023_06_09__19_15_27.mat";


odnInputSize = [416, 416];
krnInputSize = [224, 224];

%% Load things & prepare Datastores
%  Load camera intrinsics & keypoints
camera = jsondecode(fileread(camerafn));

load("cubesatPoints.mat", "sat3dPoints");
load("cubesatEdges.mat",  "satEdges");    % For debugging only

%  Read CSVs
varNames = {'filename', 'xmin', 'ymin', 'xmax', 'ymax', ...
            'qw', 'qx', 'qy', 'qz', 'rx', 'ry', 'rz', ...
            'kx1', 'ky1', 'kx2', 'ky2', 'kx3', 'ky3', 'kx4', 'ky4', ...
            'kx5', 'ky5', 'kx6', 'ky6', 'kx7', 'ky7', 'kx8', 'ky8', ...
            'kx9', 'ky9', 'kx10', 'ky10', 'kx11', 'ky11'};
varTypes = {'char'}; [varTypes{2:34}] = deal('double');
opts     = delimitedTextImportOptions('VariableNames', varNames, 'VariableTypes', varTypes);

% Update to test.csv if using trajectory dataset or validation.csv for
% training dataset
csvVal   = readtable(fullfile(dataroot, "labels_krn", "test.csv"), opts);

%  Images
imdsVal  = imageDatastore(fullfile(dataroot, 'images', csvVal.filename));

%  Labels
bboxVal  = arrayDatastore(csvVal(:, 2:5));
posesVal = arrayDatastore(csvVal(:, 6:12));
kptsVal  = arrayDatastore(csvVal(:, 13:34));

%  Combine
validationDataRaw = combine(imdsVal, bboxVal, posesVal, kptsVal);

%% Load checkpoints into both models
odn = load(fullfile("checkpoints", chkpt_odn), 'net');
krn = load(fullfile("checkpoints", chkpt_krn), 'net');

%% Run predictions
% Import OpenCV for PnP
import clib.opencv.*;
import vision.opencv.util.*;

N = size(csvVal, 1);

% Allocate arrays for predictions
p_bbox  = cell(1, N);
p_kpts  = cell(1, N);
p_trans = cell(1, N);
p_quat  = cell(1, N);

% Allocate arrays for metrics
m_iou   = zeros(1, N); % ODN - IoU
m_kpts  = zeros(1, N); % KRN - Kpts avg Euclidean distance [pix]
m_trans = zeros(1, N); % KRN - translation error [m] 
m_rot   = zeros(1, N); % KRN - rotation error [deg]
m_pose  = zeros(1, N); % KRN - pose error (SPEED)

idx_odn_fail = []; % In case ODN fails

reset(validationDataRaw);

% ============================== Main Loop ============================== %
for ii = 1:N

    %  Read data
    data    = read(validationDataRaw);
    I       = data{1};
    bbox_gt = table2array(data{2}); % [xmin, ymin, xmax, ymax] (pix. org.)
    pose_gt = table2array(data{3});
    kpts_gt = table2array(data{4});

    if length(size(I)) < 3
        I = repmat(I, [1, 1, 3]);
    end
    
    imgSize = size(I, [1, 2]); % [H, W]
    imgSize = imgSize([2, 1]); % [W, H]
    
    % ============================== 1. ODN ============================== %
    % Process image
    img = imresize(I, odnInputSize);
    img = im2single(img);

    % Detect bbox with IoU threshold 0.5
    [bbox_pr, scores] = detect(odn.net, img, 'Threshold', 0.5); % [xmin, ymin, w, h] 
    bbox_pr = double(bbox_pr);                        % (pix., resized)

    % In case there is no bbox with threshold above 0.5
    if isempty(bbox_pr)
        idx_odn_fail = [idx_odn_fail, ii];
        continue;
    end
    
    % Ground-truth to [xmin, ymin, w, h] (pix. org.)
    bbox_gt(3) = bbox_gt(3) - bbox_gt(1);
    bbox_gt(4) = bbox_gt(4) - bbox_gt(2);
    
    % Prediction to [xmin, ymin, w, h] (pix. org.)
    bbox_pr = bbox_pr ./ [odnInputSize, odnInputSize] .* [imgSize, imgSize];
    
    % Choose highest scoring bbox if multiple predictions
    if length(scores) > 1
        [best_score, best_idx] = max(scores);

        bbox_pr = bbox_pr(best_idx, :);
    end
    iou = bboxOverlapRatio(bbox_pr, bbox_gt);
    m_iou(ii) = iou;
    
    % ============================== 2. KRN ============================== %
    % Process image with predicted bbox
    % Recall: bbox [xmin, ymin, w, h] (pix. org.)
    x = bbox_pr(1) + bbox_pr(3) / 2;
    y = bbox_pr(2) + bbox_pr(4) / 2;
    w = bbox_pr(3);
    h = bbox_pr(4);
    
    [xmin, ymin, xmax, ymax] = getSquareRoI(x, y, w, h, imgSize([2, 1]), false);
    roi = [xmin, ymin, xmax - xmin, ymax - ymin]; % [xmin, ymin, w, h] (pix. org.)
    img = imcrop(I, roi);
    img = imresize(img, krnInputSize);
    
    % Predict keypoints
    img  = im2single(img);
    kpts = predict(krn.net, img); % [1 x 22] normalized keypoints (x1, y1, x2, y2, ...)

    % Predicted keypoints to [2 x 11] normalized keypoints
    kpts = reshape(double(kpts), [2, 11]);
    
    % Keypoints to pixels in original image
    kpts_pr(1,:) = kpts(1,:) * (xmax - xmin) + xmin;
    kpts_pr(2,:) = kpts(2,:) * (ymax - ymin) + ymin;

    % Ground-truth keypoints to [2 x 11] (pix)
    kpts_gt = reshape(kpts_gt, [2, 11]);

    % ============================== 3. PnP ============================== %
    % Ground truth position and quaternion vectors
    q_vbs2target_gt = pose_gt(1:4);
    r_Vo2To_vbs_gt  = pose_gt(5:7);
   
    % EPnP using MATLAB-OpenCV Interface
    % - Rp: world -> VBS
    % - Tp: VBS -> world in VBS
    kpts2d = kpts_pr';
    kpts3d = sat3dPoints';

    % Create Input and Output clib arrays and set options for cv.solvePnP:
    [kpts2dMat,kpts2d_clibArr]        = createMat(kpts2d);
    [kpts3dMat, kpts3d_clibArr]       = createMat(kpts3d);
    [cameraMat,cameraMat_clibArr]     = createMat(camera.cameraMatrix);
    [distCoeffMat, distCoeff_clibArr] = createMat("Input"); % empty array (zero distortion)
    [rvecMat, rvec]                   = createMat; % Output rotation vector
    [tvecMat, tvec]                   = createMat; % Output translation vector
    useExtrinsicGuess                 = false;
    flags                             = 1; % corresponds to EPnP as solver

    % use OpenCV solvePnP function
    RetVal = cv.solvePnP(kpts3d_clibArr,...
                         kpts2d_clibArr,...
                         cameraMat_clibArr,...
                         distCoeff_clibArr,...
                         rvec,...
                         tvec,...
                         useExtrinsicGuess,...
                         flags);

    % Convert rvec and tvec back to MATLAB arrays and convert rvec to
    % quaternion 
    [R_prMat, R_pr_clibArr] = createMat;
    cv.Rodrigues(rvec, R_pr_clibArr); %Convert rvec to rotation matrix
    R_pr = getImage(R_pr_clibArr); % Convert rotation matrix to MATLAB array
    q_vbs2target_pr = dcm2quat(R_pr'); % Rotation matrix to quaternion
    r_Vo2To_vbs_pr  = getImage(tvec)';

    % =================== Record Predictions & Metrics =================== %
    
    %  [DEBUG] - Plot images with ground truth or predicted
    %  bbox/keypoints/wireframe
    % imshow(I); hold on;
    % scatter(kpts_pr(1,:), kpts_pr(2,:), 16, 'gx');
    % xmin = bbox_pr(1); ymin = bbox_pr(2); xmax = xmin + bbox_pr(3); ymax = ymin + bbox_pr(4);
    % xmin = bbox_gt(1); ymin = bbox_gt(2); xmax = xmin + bbox_gt(3); ymax = ymin + bbox_gt(4);
    % plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], 'Color', 'y', 'LineWidth', 1.5);
    % plotWireframe(camera, satEdges, q_vbs2target_pr, r_Vo2To_vbs_pr, 1, 'Color', 'r', 'LineWidth', 1.5);
    % plotWireframe(camera, satEdges, q_vbs2target_gt, r_Vo2To_vbs_gt, 1, 'Color', 'r', 'LineWidth', 1.5);
    % waitforbuttonpress

    % Record predictions
    p_bbox{ii}  = bbox_pr;
    p_kpts{ii}  = kpts_pr;
    p_trans{ii} = r_Vo2To_vbs_pr;
    p_quat{ii}  = q_vbs2target_pr;
    
    % Metrics
    m_kpts(ii)  = mean(vecnorm(kpts_pr - kpts_gt, 2, 1)); % [pix]
    m_trans(ii) = positionError(r_Vo2To_vbs_pr, r_Vo2To_vbs_gt);
    m_rot(ii)   = orientationError(q_vbs2target_pr, q_vbs2target_gt); % [deg]
    m_pose(ii)  = m_trans(ii) / norm(r_Vo2To_vbs_gt) + deg2rad(m_rot(ii));

    displayProgress(ii, N);
end

idx_odn_succ = setdiff(1:N, idx_odn_fail);

fprintf("Num. of failed ODN detections: %d\n", length(idx_odn_fail));
fprintf("Mean   IoU [-]:   %.3f\n", mean(m_iou(idx_odn_succ)));
fprintf("Mean   E_t [m]:   %.3f +/- %.3f\n", mean(m_trans(idx_odn_succ)), std(m_trans(idx_odn_succ)));
fprintf("Median E_t [m]:   %.3f\n", median(m_trans(idx_odn_succ)));
fprintf("Mean   E_R [deg]: %.3f +/- %.3f\n", mean(m_rot(idx_odn_succ)), std(m_rot(idx_odn_succ)));
fprintf("Median E_R [deg]: %.3f\n", median(m_rot(idx_odn_succ)));
fprintf("Mean   E_p [-]:   %.3f +/- %.3f\n", mean(m_pose(idx_odn_succ)), std(m_pose(idx_odn_succ)));

%% Save predictions
[~,dataname,~] = fileparts(dataroot);
outputdir = fullfile('outputs', dataname);
if ~isfolder(outputdir); mkdir(outputdir); end

save(fullfile(outputdir, "predictions.mat"), ...
        "p_bbox", "p_kpts", "p_trans", "p_quat", ...
        "m_kpts", "m_trans", "m_rot", "m_pose", ...
        "idx_odn_fail");
%% Analyses
% Visualize the images with the worst errors
[sortVal, sortIdx] = sort(m_rot, 'descend');

f = figure("color", "w");
for idx = 1:N
    fn = fullfile(dataroot, "images", csvVal.filename{idx});
    I  = imread(fn);

    % Image
    fig = imshow(I); title(sprintf("Image: %s", csvVal.filename{idx}));
    
    bb = p_bbox{idx};
    bb(3) = bb(1) + bb(3); bb(4) = bb(2) + bb(4); % Convert from [xmin,ymin,w,h] to [xmin,ymin,xmax,ymax]
    hold on; plot(gca,bb([1, 1, 3, 3, 1]), bb([2, 4, 4, 2, 2]));
    hold on; scatter(gca,p_kpts{idx}(1,:), p_kpts{idx}(2,:),'green');
    legend('predicted bounding box','predicted kpts')

    waitforbuttonpress
    clf
end


