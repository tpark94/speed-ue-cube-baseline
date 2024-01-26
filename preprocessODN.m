function data = preprocessODN(data, targetSize)
% PREPROCESSODN  Preprocess dataset to train/test ODN
% -------------------------------------------------------------------------
% SYNTAX: newdata = preprocessODN(data, targetSize)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Takes the datastore consisting of {image, bbox}. Simply resize
%         images and adjust bbox parameters accordingly.
% -------------------------------------------------------------------------
% INPUTS:
%         data       [-]   - Datastore
%         targetSize [pix] - Final image size (H x W)
% -------------------------------------------------------------------------
% OUTPUTS:
%         newdata          - Datastore
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------
for ii = 1:size(data,1)
    I    = data{ii, 1};
    bbox = data{ii, 2}; % [x, y, w, h] (pix)

    imgOrgSize = size(I, [1,2]);

    % Resize image
    I = imresize(I, targetSize);

    % To RGB if grayscale
    if length(size(I)) < 3
        I = repmat(I, [1, 1, 3]);
    end

    % ========== Clean up ========== %
    I = im2single(I); % Image to floats [0, 1]

    scale = targetSize ./ imgOrgSize; % Adjust bbox
    bbox  = bboxresize(bbox, scale);

    data(ii, 1:2) = {I, bbox};
end