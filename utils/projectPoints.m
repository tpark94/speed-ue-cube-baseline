function imgp = projectPoints(objp, q, t, cam)
% PROJECTPOINTS  Project 3D points to 2D image given pose and camera
% -------------------------------------------------------------------------
% SYNTAX: imgp = projectPoints(objp, q, t, cam)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Perform projective transformation of input `objp` [3 x N] given
%         camera (`cam`), orientation as quaternion (`q`), and position
%         (`t`).
% -------------------------------------------------------------------------
% INPUTS:
%         objp [-] - [3 x N] Object 3D points
%         q    [-] - Quaternion representing orientation of target w.r.t. 
%                    camera -- quaternion is scalar-first
%         t    [-] - Translation vector from camera to target expressed in
%                    camera's reference frame
%         cam  [-] - Structure containing camera intrinsic parameters
% -------------------------------------------------------------------------
% OUTPUTS:
%         imgp [-] - [2 x N] Projected 2D image points
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------

% [3 x N] object points
if size(objp, 1) ~= 3
    objp = objp';
end

% [1 x 4] quaternions
if size(q, 2) ~= 4
    q = q';
end

% [3 x 1] translations
if size(t, 1) ~= 3
    t = t';
end

% P_vbs = t + R*P_tango
% where 
%   t : translation from vbs_origin to tango_origin
%   R : DCM from tango_frame to vbs_frame 
poseMat = [ quat2dcm(q)' t ];

% 2D pixels, homogenous coord [x y 1]
v2dh = poseMat * [objp; ones(1, size(objp, 2))];

% Back to regular, original coord [x y]. Size: (2 x N)
x = v2dh(1,:) ./ v2dh(3,:);
y = v2dh(2,:) ./ v2dh(3,:);

% Apply distortion if applicable
if isfield(cam, "distCoeffs")
    dist  = cam.distCoeffs;
    x0    = x; y0 = y;
    r2    = x0.^2 + y0.^2;
    cdist = 1 + dist(1)*r2 + dist(2)*r2.^2 + dist(5)*r2.^3;
    x     = x0.*cdist + dist(3)*2*x0.*y0 + dist(4)*(r2 + 2*x0.^2);
    y     = y0.*cdist + dist(3)*(r2 + 2*y0.^2) + dist(4)*2*x0.*y0;
end

% Projective transformation
imgp = cam.cameraMatrix * [x ; y ; ones(1, size(objp, 2))];
imgp = imgp(1:2,:);
