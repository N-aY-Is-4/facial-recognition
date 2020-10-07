clc;
close all;
clear all;
 %% load network
 net=alexnet;
 inputSize = net.Layers(1).InputSize;
 % Dataset  
matlabpath='C:\Users\ayush\MATLAB_CAPSTONE\Dataset1';
data=fullfile(matlabpath,'trainingset');
train=imageDatastore(data,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
[imdsTrain,imdsValidation]=splitEachLabel(train,0.8,'randomized');

% number of classes
numClasses = numel(categories(train.Labels));  

% Transfer the layers to the new classification task by replacing the last three layers with a fully connected layer,
% a softmax layer, and a classification output layer.
%  To learn faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor
% and BiasLearnRateFactor values of the fully connected layer.
 
 layers=[imageInputLayer([227 227 3])
         net(2:end-3)
         fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
         softmaxLayer
         classificationLayer()
         ]
 

%% training
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%% 

% An epoch is a full training cycle on the entire training data set. 

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

TrainNet=trainNetwork(augimdsTrain,layers,options);


%% accuracy

pred=classify(TrainNet,imdsValidation);
accuracy=mean(pred==imdsValidation.Labels);

 %% Testing
 
a=imread('9.jpg');
out=classify(TrainNet,a);

figure,imshow(a);
title(string(out))

msgbox(string(out))


    
    
