function lgraph = buildKRN(numKpts, inputSize)
% BUILDKRN  Build KRN graph
% -------------------------------------------------------------------------
% SYNTAX: lgraph = buildKRN(numKpts)
% -------------------------------------------------------------------------
% DESCRIPTION:
%         Build KRN with MobileNetV2 backbone.
% -------------------------------------------------------------------------
% INPUTS:
%         numKpts [-] - Number of keypoints to detect
% -------------------------------------------------------------------------
% OUTPUTS:
%         lgraph      - layerGraph of KRN
% -------------------------------------------------------------------------
% AUTHORS: Tae Ha "Jeff" Park, Zahra Ahmed
% -------------------------------------------------------------------------
% COPYRIGHT: (c) 2023 Stanford’s Space Rendezvous Laboratory
% -------------------------------------------------------------------------

% MobileNetv2 backbone
net    = mobilenetv2('Weights', 'imagenet');
lgraph = layerGraph(net);

% "base" ends at layer 147
lgraph = removeLayers(lgraph, {'Conv_1', 'Conv_1_bn', 'out_relu', ...
                               'global_average_pooling2d_1', ...
                               'Logits', 'Logits_softmax', ...
                               'ClassificationLayer_Logits'});

% Add extra layers
extra = [
    depthwiseConvolution(1024, 1, 'extra_1')
    depthwiseConvolution(1024, 1, 'extra_2')
    concatenationLayer(3, 2, 'Name', 'extra_concat') % Along channel dim.
    depthwiseConvolution(1024, 1, 'extra_3')
];
lgraph = addLayers(lgraph, extra);
lgraph = connectLayers(lgraph, 'block_16_project_BN', 'extra_1_depthwise');

% - router from 'block_12_add'
lgraph = addLayers(lgraph, routerLayer(64, 'router'));
lgraph = connectLayers(lgraph, 'block_12_add', 'router_conv');
lgraph = connectLayers(lgraph, 'router_reshape', 'extra_concat/in2');

% Head layer
lgraph = addLayers(lgraph, convolution2dLayer(7, 2 * numKpts, 'Name', 'head_conv'));
lgraph = connectLayers(lgraph, 'extra_3_pointwise_relu', 'head_conv');

% Regression layer as output (implements MSE loss)
lgraph = addLayers(lgraph, regressionLayer('Name', 'output_regression'));
lgraph = connectLayers(lgraph, 'head_conv', 'output_regression');

% NOTE:
% The first layer of MobileNetv2 here is `imageInputLayer` with default
% normalization behavior of 'zscore'. Replace mean/std of the first
% ImageInputLayer to that of ImageNet.
%
% Note that this means the input image must be normalized to [0, 1].
mean = [0.485, 0.456, 0.406];
std  = [0.229, 0.224, 0.225];

newLayer = imageInputLayer(inputSize, ...
                           'Normalization', 'zscore', ...
                           'Mean', reshape(mean, [1, 1, 3]), ...
                           'StandardDeviation', reshape(std, [1, 1, 3]));
lgraph = replaceLayer(lgraph, 'input_1', newLayer);

end


function conv = depthwiseConvolution(cout, stride, layerName)
% Helper function for depthwise convolution operations
conv = [
    % depth-wise
    groupedConvolution2dLayer(3, 1, 'channel-wise', 'Stride', stride, ...
                              'Padding', 'same', ...
                              'Name', [layerName, '_depthwise'])
    batchNormalizationLayer('Name', [layerName, '_depthwise_BN'])
    reluLayer('Name', [layerName, '_depthwise_relu'])

    % point-wise
    convolution2dLayer(1, cout, 'Stride', 1, ...
                       'Name', [layerName, '_pointwise'])
    batchNormalizationLayer('Name', [layerName, '_pointwise_BN'])
    reluLayer('Name', [layerName, '_pointwise_relu'])
];
end


function router = routerLayer(cout, layerName)
% Helper function to route a signal from a specified layer
router = [
    convolution2dLayer(1, cout, 'Stride', 1, 'Name', [layerName, '_conv'])
    batchNormalizationLayer('Name', [layerName, '_BN'])
    leakyReluLayer(0.2, 'Name', [layerName, '_leakyrelu'])
    functionLayer(@(in) reshapeTensor(in), Name=[layerName, '_reshape'])
];
end


function out = reshapeTensor(in)
% Helper function to reshape tensor by stride 2
% - in:  [H, W, C, B]
% - out: [H/2, W/2, 4C, B]
s = 2; % stride

H = size(in, 1); W = size(in, 2); C = size(in, 3); B = size(in, 4);

out = reshape(in, H/s, s, W/s, s, C, B);
out = permute(out, [1, 3, 2, 4, 5, 6]);
out = reshape(out, H/s, W/s, s*s*C, B);
end
