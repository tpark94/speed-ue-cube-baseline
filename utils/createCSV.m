function createCSV(dataroot, camera, split, varargin)
% CREATECSV  Create CSV file of processed labels
% -------------------------------------------------------------------------
% SYNTAX: createCSV(dataroot, camerafn, source, split)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Convert the .json label file into a CSV file containing following
%         information in the columns
%         - absolute path to image 
%         - bbox [xmin, ymin, xmax, ymax] (pix)
%         - pose (quaternion) [qw, qx, qy, qz]
%         - pose (translation) [tx, ty, tz]
%         - keypoints [kx1, ky1, ..., kx11, ky11] (pix)
%
%         The CSV file will be saved to `dataroot`/labels_krn/`split`.csv.
% -------------------------------------------------------------------------
% INPUTS:
%         dataroot   [-] - Absolute path to where the dataset is located
%         camera     [-] - Structure containing camera intrinsics
%         split      [-] - "train", "validation", "test"
% -------------------------------------------------------------------------
% OUTPUTS:
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------

% Cubesat Model Files
stlFile  = 'cubesat/cubesat.stl';
kptsFile = 'cubesat/cubesatPoints.mat';
edgeFile = 'cubesat/cubesatEdges.mat';

narginchk(3, 4);
debug = false;
if nargin == 4
    debug = varargin{1};
end

% ========== Housekeeping ========== %
split = lower(split);
assert(ismember(split, ["train", "validation", "test"]), ...
       "split must be either train, validation, or test.");

% Where to save CSV
splitdir = fullfile(dataroot, 'labels_krn');
if ~exist(splitdir, 'dir'); mkdir(splitdir); end
csvpath = fullfile(splitdir, sprintf('%s.csv', split));


% ========== Read necessary files ========== %
% Label file
json = jsondecode(fileread(fullfile(dataroot, strcat(split, '.json'))));

% Cubesat STL file [cm]
stl      = stlread(stlFile);
vertices = stl.Points' / 100; % [cm] -> [m] (Cubesat)

% Keypoints
load(kptsFile, 'sat3dPoints'); % [3 x 11] (m)

% ========== Create CSV ========== %
if debug
    f = figure('visible', 'on', 'color', 'w');
    fprintf("Debug mode - bbox and kpts will be visualized\n");
else
    csvID = fopen(csvpath, 'w');
    fprintf('Creating %s ...\n', csvpath);
end

for idx = 1:length(json)
%     % File name
%     fpath = json(idx).filename; 
    fpath = json(idx).filename + ".png"; 

    % Pose labels
    q = json(idx).q_vbs2tango_true;
    t = json(idx).r_Vo2To_vbs_true; % [m]

    % Bounding box from projecting STL
    stlVertices2D = projectPoints(vertices, q, t, camera);
    xmin = min(stlVertices2D(1,:));
    xmax = max(stlVertices2D(1,:));
    ymin = min(stlVertices2D(2,:));
    ymax = max(stlVertices2D(2,:));

    % Keypoints from projecting 3D keypoints
    keypoints = projectPoints(sat3dPoints, q, t, camera); % [2 x 11]
    
    if debug
        load(edgeFile, 'satEdges'); % Load wireframe model

        % image
        I = imread(fullfile(dataroot, 'images', fpath));
        imshow(I);

        hold on;
        plot([xmin, xmax, xmax, xmin, xmin], ...
             [ymin, ymin, ymax, ymax, ymin], ...
             'g', 'lineWidth', 1.5);

        % wireframe
        plotWireframe(camera, satEdges, q, t, 'Color', 'y', 'LineWidth', 1.2);

        waitforbuttonpress
        clf
    else
        fprintf(csvID, '%s, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f', ...
                        fpath, xmin, ymin, xmax, ymax, q(1), q(2), q(3), q(4), t(1), t(2), t(3));
        for ii = 1:size(keypoints,2)
            fprintf(csvID, ', %.6f, %.6f', keypoints(1,ii), keypoints(2,ii));
        end
        fprintf(csvID, '\n');
    end

    displayProgress(idx, length(json));
end

fclose(csvID);

end
