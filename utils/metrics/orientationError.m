function eR = orientationError(R1, R2)
% ORIENTATIONERROR  Return orientation error between two orientation inputs
% -------------------------------------------------------------------------
% SYNTAX: eR = orientationError(R1, R2)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Return angular distance between two attitudes which can be either
%         3 x 3 DCM or 4 x 1 quaternion vectors
% -------------------------------------------------------------------------
% INPUTS:
%         R1 [-] - First orientation input
%         R2 [-] - Second orientation input
% -------------------------------------------------------------------------
% OUTPUTS:
%         eR [deg] - orientation error
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------

% Convert to rotation matrices if necessary 
if ~all(size(R1) == [3, 3])
    assert(size(R1, 1) == 4 || size(R1, 2) == 4);
    
    % Convert to rotation matrices
    R1 = quat2dcm(R1);
    R2 = quat2dcm(R2);
end

eR = acosd((trace(R2' * R1) - 1)/2); % [deg]

end
