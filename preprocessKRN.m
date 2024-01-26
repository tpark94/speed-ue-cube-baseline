function data = preprocessKRN(data, targetSize, isTrain)
% PREPROCESSKRN  Preprocess dataset to train/test KRN
% -------------------------------------------------------------------------
% SYNTAX: newdata = preprocessKRN(data, targetSize, isTrain)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Takes the datastore consisting of {image, bbox, keypoints} for
%         training and {image, bbox, pose} for test. Image is cropped and
%         resized to `targetSize`, but cropped RoI is perturbed about the
%         ground-truth bounding box randomly for training. Keypoints are
%         also adjusted according to the final RoI if training.
% -------------------------------------------------------------------------
% INPUTS:
%         data       [-]   - Datastore
%         targetSize [pix] - Final image size (H x W)
%         isTrain    [-]   - Boolean 1 if training and 0 if test
% -------------------------------------------------------------------------
% OUTPUTS:
%         newdata          - Datastore
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------
for ii = 1:size(data, 1)
    I    = data{ii, 1};
    bbox = data{ii, 2}; % [xmin, ymin, xmax, ymax] (pix)

    if isTrain
        kpts = data{ii, 3}; % [kx1, ky1, ..., kxN, kyN] (pix)
    else
        pose = data{ii, 3}; % [q1, q2, q3, q4, t1, t2, t3]
    end

    % Bbox to [x, y, w, h] format
    w = bbox.xmax - bbox.xmin;
    h = bbox.ymax - bbox.ymin;
    x = bbox.xmin + w / 2;
    y = bbox.ymin + h / 2;

    % Square RoI based on ground-truth bbox
    imgSize = size(I); % [H, W] (pix)
    [xmin, ymin, xmax, ymax] = getSquareRoI(x, y, w, h, imgSize, isTrain);

    % Crop and resize image
    I = imcrop(I, [xmin, ymin, xmax - xmin, ymax - ymin]);
    I = imresize(I, targetSize);

    % To RGB if grayscale
    if length(size(I)) < 3
        I = repmat(I, [1, 1, 3]);
    end

    % ========== Clean up ========== %
    I = im2single(I); % Image to floats [0, 1]

    bbox.xmin = xmin; % Store final RoI to bbox
    bbox.xmax = xmax;
    bbox.ymin = ymin;
    bbox.ymax = ymax;

    if isTrain
        % Adjust & normalize keypoints
        kptsX = (kpts{1, 1:2:end} - xmin) / (xmax - xmin);
        kptsY = (kpts{1, 2:2:end} - ymin) / (ymax - ymin);
        kpts  = zeros(size(kpts));
        kpts(1:2:end) = kptsX;
        kpts(2:2:end) = kptsY;
    
        data(ii, 1:2) = {I, kpts}; % Bbox unnecessary for training KRN
    else
        data(ii, 1:3) = {I, table2array(bbox), table2array(pose)};
    end
end