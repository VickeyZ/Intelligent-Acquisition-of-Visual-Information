% Projector-Camera Stereo calibration parameters:

% Intrinsic parameters of camera:
fc_left = [ 3696.381122 3695.758057 ]; % Focal Length
cc_left = [ 1305.658303 1032.738145 ]; % Principal point
alpha_c_left = [ 0.000000 ]; % Skew
kc_left = [ -0.519871 0.103430 -0.001503 0.000985 0.000000 ]; % Distortion

% Intrinsic parameters of projector:
fc_right = [ 1429.713257 1430.013463 ]; % Focal Length
cc_right = [ 508.115032 580.661735 ]; % Principal point
alpha_c_right = [ 0.000000 ]; % Skew
kc_right = [ 0.087113 -0.379831 -0.003141 -0.001263 0.000000 ]; % Distortion

% Extrinsic parameters (position of projector wrt camera):
om = [ -0.039184 0.033237 0.025858 ]; % Rotation vector
T = [ 66.671113 -91.618580 -188.138611 ]; % Translation vector
