function [xmin, ymin, xmax, ymax] = getSquareRoI(x, y, w, h, imgSize, isTrain)
% GETSQUAREROI  Get square RoI based on ground-truth bbox
% -------------------------------------------------------------------------
% SYNTAX: [xmin, ymin, xmax, ymax] = getSquareRoI(x, y, w, h, imgSize, isTrain)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Given ground-truth bbox [x, y, w, h], get a new RoI [xmin, ymin,
%         xmax, ymax] according to the description in [PSD19].
%         - If training, bbox is enlarged and shifted by random factors
%         - If testing, bbox is enlarged by a constant factor
% -------------------------------------------------------------------------
% INPUTS:
%         x       [pix] - Bbox center x-coordinate
%         y       [pix] - Bbox center y-coordinate
%         w       [pix] - Bbox width
%         h       [pix] - Bbox height
%         imgSize [pix] - Original image size (H x W)
%         isTrain [-]   - Boolean 1 if training and 0 if test
% -------------------------------------------------------------------------
% OUTPUTS:
%         xmin    [pix] - New RoI upper-left x-coordinate
%         ymin    [pix] - New RoI upper-left y-coordinate
%         xmax    [pix] - New RoI lower-right x-coordinate
%         ymax    [pix] - New RoI lower-right x-coordinate
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------
roiSize = max(w, h); % base square roi size (pix)

if isTrain
    % Enlarge tight roi by random factor within [1, 1.5]
    roiSize = (1 + 0.5 * rand) * roiSize; 

    % Shift expanded roi by random factor
    fx = 0.2 * (2 * rand - 1) * roiSize;
    fy = 0.2 * (2 * rand - 1) * roiSize;
else
    % Constant enlargement factor
    roiSize = 1.2 * roiSize;
    fx = 0; fy = 0;
end

xmin = max(0, x - roiSize / 2 + fx);
xmax = min(imgSize(2), x + roiSize / 2+ fx);
ymin = max(0, y - roiSize / 2 + fy);
ymax = min(imgSize(1), y + roiSize / 2+ fy);



