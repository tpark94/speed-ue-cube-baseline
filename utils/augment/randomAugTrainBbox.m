function data = randomAugTrainBbox(data, p)
% RANDOMAUGTRAINBBOX  Random augmentation for ODN
% -------------------------------------------------------------------------
% SYNTAX: augData = randomAugTrainBbox(data, p)
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
for ii = 1:size(data,1)
    
    img  = data{ii, 1};
    bbox = data{ii, 2}; % [xmin, ymin, w, h] (pix of resized)

    [himg, wimg] = size(img, [1,2]);
    xmin = bbox(1); ymin = bbox(2); w = bbox(3); h = bbox(4);

    % Augment #1 - BRIGHTNESS/CONTRAST
    if rand() < p
        img = jitterColorHSV(img, "Contrast" ,[0.5, 2], "Brightness", [0, 0.25]);
    end

    % Augment #2 - GAUSSIAN NOISE
    if rand() < p
        img = imnoise(img, "gaussian");
    end

    % Augment #3 - FLIP (Horizontal & Vertical)
    if rand() < p
        if rand() < 0.5 
            % Horizontal flip
            img  = flip(img, 2);
            xmin = wimg - xmin - w;
        else 
            % Vertical flip
            img  = flip(img, 1);
            ymin = himg - ymin - h;
        end
    end

    bbox = [xmin, ymin, w, h];

    % Augment #4 - ROTATE in 90 degree increments
    if rand() < p
        randRot = randi(4) - 1;
        img = rot90(img, randRot); % COUNTER-CLOCKWISE rotation

        switch randRot
            case 0
                % No rotation
                continue
            case 1
                % 90 deg
                bbox = [ymin, wimg - xmin - w, h, w];
            case 2
                % 180 deg
                bbox = [wimg - xmin - w, himg - ymin - h, w, h];
            case 3
                % 270 deg
                bbox = [himg - ymin - h, xmin, h, w];
            otherwise
                error('Tsk tsk should not be reaching this');
        end
    end
    
    data(ii, 1:2) = {img, bbox};

end