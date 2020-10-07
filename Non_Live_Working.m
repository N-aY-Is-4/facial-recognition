clc;
close all;
clear all;
%% condition loop
for i=1:5
%     img=getsnapshot(video);
    %%Taking an Image
    [fname, path] = uigetfile('.jpg','Open a Face as input for training');
    fname=strcat(path,fname);
    im=imread(fname);
    img = imresize(im,[227 227]);
    imshow(img);
    title('Input Face');
    pause(1);
%   imshow(img);
    imwrite(img,'Image1.jpg');
%% resize image
    I1=imread('Image1.jpg');
    I2 = imresize(I1,[227 227]); 
%   In=rgb2gray(I2); % use if the image containing RGB value 3
%   figure;imshow(I2);
    imwrite(I2,'Image2.jpg') ;
%% Cascade Object Detector
    faceDetector=vision.CascadeObjectDetector();
%% Input Image
    img=imread('Image2.jpg');
%% Detect Human Face
    bbox=step(faceDetector,img);
%     figure;imshow(img);
   
    %% Display Detected Face
    if ~isempty(bbox)
        Face=insertObjectAnnotation(img,'rectangle',bbox,'Face');
        figure;imshow(Face);
        imwrite(Face,'Student.jpg') ;
%crop the rounded face and save it
        FacCrop=imcrop(Face,bbox);
        fname = sprintf('Image4.jpg',i);
        fpath = fullfile('C:\Users\ayush\MATLAB_CAPSTONE\Dataset1\sample_set', fname);
        imwrite(FacCrop, fpath);
        I3=imread('Student.jpg');
        I4 = imresize(I3,[227 227]); 
%         figure;imshow(I4);
        mytitle=strcat('Face Number:',num2str(i));
        figure;imshow(Face);title(mytitle);
        i=i+1;
        disp("Face Found");
        %% load network
        net=alexnet;
        inputSize = net.Layers(1).InputSize;
        % Dataset  
        matlabpath='C:\Users\ayush\MATLAB_CAPSTONE\Dataset1';
        data=fullfile(matlabpath,'trainingset');
        train=imageDatastore(data,'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
        [imdsTrain,imdsValidation]=splitEachLabel(train,0.85,'randomized');

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
               ];
 

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
                                  'MaxEpochs',8, ...
                                  'InitialLearnRate',0.001, ...
                                  'Shuffle','every-epoch', ...
                                  'ValidationData',augimdsValidation, ...
                                  'ValidationFrequency',2, ...
                                  'Verbose',false, ...
                                  'Plots','training-progress');

        TrainNet=trainNetwork(augimdsTrain,layers,options);


        %% accuracy

        pred=classify(TrainNet,imdsValidation);
        accuracy=mean(pred==imdsValidation.Labels);
        
        %% Testing
 
        a=imread('Student.jpg');
        out=classify(TrainNet,a);
        figure,imshow(a);
        title(string(out))
        msgbox(string(out))
        while(1)
            z=input('Student Identified ??\n','s');
                 if z=='N' || z=='n'
                    pause(2);
                    w=input('Check again ??\n','s');
                        if w=='N' || w=='n'
                         pause(2);
                         return;
                     elseif w=='y' || w=='Y'
                         break;
                      end
                 elseif z=='y' || z=='Y'
                    m=input('Next Student!!!, Y/N :','s');
                     if m=='N' || m=='n'
                         pause(2);
                         return;
                     elseif m=='y' || m=='Y'
                         break;
                     end
                 end
          end
    else
        disp("No Student Found");
        while(1)
            m=input('Done Clicking!!!, Y/N :','s');
                if m=='Y' || m=='y'
                    pause(2);
                    image_folder='C:\Users\ayush\MATLAB_CAPSTONE\Dataset1\sample_set';
                    filenames=dir(fullfile(image_folder,'*.jpg'));
                    total_images=numel(filenames);
                        for n=1:total_images
                            f=fullfile(image_folder, filenames(n).name);
                            our_images=imread(f);
                            figure(n);imshow(our_images);
                        end
                    return;
                elseif m=='N' || m=='n'
                       break;
                end
             
        end
        continue;
        
 
    end
    

end


