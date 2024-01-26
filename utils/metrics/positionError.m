function eT = positionError(t1, t2)
% POSITIONERROR  Return position error between two translation vectors
% -------------------------------------------------------------------------
% SYNTAX: eT = positionError(t1, t2)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Return Euclidean error between two translation vectors
% -------------------------------------------------------------------------
% INPUTS:
%         t1 [-] - First translation vector
%         t2 [-] - Second translation vector
% -------------------------------------------------------------------------
% OUTPUTS:
%         eT [-] - Translation error
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------

% Size check
if size(t1, 1) ~= 3
    t1 = t1';
end
if size(t2, 1) ~= 3
    t2 = t2';
end

eT = norm(t1 - t2);
