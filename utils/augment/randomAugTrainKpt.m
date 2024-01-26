function data = randomAugTrainKpt(data, p)
% RANDOMAUGTRAINKPT  Random augmentation for KRN
% -------------------------------------------------------------------------
% SYNTAX: augData = randomAugTrainKpt(data, p)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Performs a list of random augmentations each with probability
%         `p`
% -------------------------------------------------------------------------
% INPUTS:
%         data [-] - Datastore
%         p    [-] - Probability of performing each augmentation
% -------------------------------------------------------------------------
% OUTPUTS:
%         augData [-] - Datastore
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------
for ii = 1:size(data, 1)
    
    img       = data{ii, 1}; % [0, 1] single
    keypoints = data{ii, 2};

    % Augment #1 - BRIGHTNESS/CONTRAST
    if rand() < p
%         img = jitterColorHSV(img, "Contrast" ,[0.5, 2], "Brightness", [-0.1, 0.1]);

        alpha = log([0.5, 2.0]);
        beta  = [-25, 25] / 255;
        
        % Contrast
        loga = rand() * (alpha(2) - alpha(1)) + alpha(1);
        a    = exp(loga);

        % Brightness
        b    = rand() * (beta(2) - beta(1)) + beta(1);

        % Apply
        img = single(a) * img + single(b);
        img = min(max(img, 0), 1);
    end

    % Augment #2 - GAUSSIAN NOISE
    if rand() < p
%         img = imnoise(img, "gaussian", 0, 0.01);

        std = 25 / 255;

        noise = single(randn(size(img)) * std);
        img   = min(max(img + noise, 0), 1);
    end

    % Augment #3 - FLIP (Horizontal & Vertical)
    if rand() < p
        if rand() < 0.5 
            % Horizontal flip
            img = flip(img, 2);
            keypoints(1, 1:2:end) = 1 - keypoints(1, 1:2:end);
        else 
            % Vertical flip
            img = flip(img, 1);
            keypoints(1, 2:2:end) = 1 - keypoints(1, 2:2:end);
        end
    end

    % Augment #4 - ROTATE in 90 degree increments
    if rand() < p
        randRot = randi(4)-1;
        img = rot90(img, randRot);

        x = keypoints(1, 1:2:end);
        y = keypoints(1, 2:2:end);

        switch randRot
            case 0
                continue;
            case 1
                % 90 deg
                keypoints(1, 1:2:end) = y;
                keypoints(1, 2:2:end) = 1 - x;
            case 2
                % 180 deg
                keypoints(1, 1:2:end) = 1 - x;
                keypoints(1, 2:2:end) = 1 - y;
            case 3
                % 270 deg
                keypoints(1, 1:2:end) = 1 - y;
                keypoints(1, 2:2:end) = x;
            otherwise
                error('Tsk tsk should not be reaching this');
        end
    end
    
    data(ii,1:2) = {img, keypoints};
end