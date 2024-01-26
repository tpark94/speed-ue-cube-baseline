function satEdges = plotWireframe(cam, satEdges, q_vbs2target, r_Vo2To_vbs, varargin)
% PLOTWIREFRAME  Plot satellite wireframe based on given pose
% -------------------------------------------------------------------------
% SYNTAX: plotWireframe(cam, satEdges, q_vbs2tango, r_Vo2To_vbs, scale)
%         plotWireframe(__, Name, Value)
%
%         satEdges = plotWireframe(__)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Plot the satellite wireframe model on current figure based on the
%         given pose (`q_vbs2target`, `r_Vo2To_vbs`)
% -------------------------------------------------------------------------
% INPUTS:
%         cam          [-] - Structure containing camera intrinsics
%         satEdges     [-] - Structure containing wireframe points &
%                            connectivity
%         q_vbs2target [-] - Quaternion vector (scalar-first)
%         r_Vo2To_vbs  [-] - Position vector from VBS to target in VBS
%                            frame
%         varargin     [-] - (Optional) Additional arguments for plotting
% -------------------------------------------------------------------------
% OUTPUTS:
%         satEdges     [-] - Input structure with additional information on
%                            reprojected point 2D coordinates
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------

narginchk(4, Inf); % At least 4 inputs

% Default arguments
lineWidth = 2;
color     = 'g';

% Process additional arguments
if nargin > 4
    for i = 1:2:length(varargin)
        if strcmp(varargin{i}, 'LineWidth')
            lineWidth = varargin{i + 1};
        end
        if strcmp(varargin{i}, 'Color')
            color = varargin{i + 1};
        end
    end
end

% Check input sizes to ensure concatenation
if size(r_Vo2To_vbs, 2) ~= 1
    r_Vo2To_vbs = r_Vo2To_vbs'; % [3 x 1]
end
if size(q_vbs2target, 2) ~= 4
    q_vbs2target = q_vbs2target'; % [1 x 4]
end

% Reproject the model points
satEdges.reprImagePoint1 = projectPoints(satEdges.point1(1:3,:), ...
                                 q_vbs2target, r_Vo2To_vbs, cam);
satEdges.reprImagePoint2 = projectPoints(satEdges.point2(1:3,:), ...
                                 q_vbs2target, r_Vo2To_vbs, cam);

% Plot on image
hold on
for i = 1:length(satEdges.point1)
    U = [satEdges.reprImagePoint1(1,i), satEdges.reprImagePoint2(1,i)];
    V = [satEdges.reprImagePoint1(2,i), satEdges.reprImagePoint2(2,i)];
    plot(U, V, 'LineWidth', lineWidth, 'Color', color')
end
xlim([0 cam.Nu]); ylim([0 cam.Nv]);
