classdef TrainingNetwork < handle
    properties(SetAccess = public)
        valid;
        
        layers; %the array of layer objects for our network
        numLayers;
        userDimmensions; %the array specifying the network's dimension, from the user
        
        numSamples;
        numInputs;
        numOutputs;
        sampleMatrix; %the matrix containing all samples
        
        isContinuous;
        minVal;
        maxVal;
        numSectors;
        sectorSize;
        
        learningRate;
        tolerance;
        maxIterations;
        
        backError; 
        RMSE;
    end
    methods
        function trainingNetworkObj = TrainingNetwork(sampleMatrix, userDimensions, tolerance, maxIterations, minVal, maxVal, isContinuous)
            trainingNetworkObj.valid = true;
            if(~trainingNetworkObj.validateInputs(sampleMatrix, userDimensions, tolerance, maxIterations, minVal, maxVal))
                trainingNetworkObj.valid = false;
                return
            end
            
            %Normalize the values between 0 and pi
            trainingNetworkObj.sampleMatrix = (sampleMatrix - minVal)/(maxVal-minVal + 1)*2*pi;
            trainingNetworkObj.minVal = minVal;
            trainingNetworkObj.maxVal = maxVal;
            trainingNetworkObj.isContinuous = isContinuous;
            
            %Initialize num outputs, inputs, samples
            numOutputs = userDimensions(length(userDimensions));
            trainingNetworkObj.numOutputs =  numOutputs;
            trainingNetworkObj.numInputs = length(sampleMatrix(1,:)) - numOutputs;
            trainingNetworkObj.numSamples = length(sampleMatrix(:,1));
            
            %Initialize learning rate/tolerance
            trainingNetworkObj.learningRate =  1;
            trainingNetworkObj.tolerance = tolerance;
            trainingNetworkObj.maxIterations = maxIterations;
            
            %Initialize userDimensions/numlayers
            trainingNetworkObj.userDimmensions = userDimensions;
            trainingNetworkObj.numLayers = length(userDimensions);
            
            %Initialize layers
            trainingNetworkObj.layers = TrainingLayer.empty();
            for i = 1:trainingNetworkObj.numLayers
                trainingNetworkObj.layers(i) = TrainingLayer(trainingNetworkObj,i,userDimensions(i));
            end
            
            %Convert input to complex, if discreet, 
            %setup sectorsize/numsectors
            if(~isContinuous)
                trainingNetworkObj.numSectors = maxVal - minVal + 1; %max-min +1
                trainingNetworkObj.sectorSize = 2*pi / trainingNetworkObj.numSectors;
                trainingNetworkObj.sampleMatrix(:, trainingNetworkObj.numInputs+1:end) = trainingNetworkObj.sampleMatrix(:, trainingNetworkObj.numInputs+1:end) + trainingNetworkObj.sectorSize/2;
                trainingNetworkObj.sampleMatrix = exp((1i * 2*pi *trainingNetworkObj.sampleMatrix)/trainingNetworkObj.numSectors); %Ask about this
            else
                trainingNetworkObj.sampleMatrix = exp(1i*trainingNetworkObj.sampleMatrix); %Ask about this TODO
            end
            
            %Start learning after setting up network
            trainingNetworkObj.trainNetwork();
        end
                
        function validated = validateInputs(~, sampleMatrix, userDimensions, tolerance, maxIterations, minVal, maxVal)
            validated = true;
                        %Normalize the values between 0 and pi
            if(minVal >= maxVal)
                fprintf('minVal must be less than maxVal\n');
                validated = false;
            end
            
            nOut = userDimensions(length(userDimensions));
            if(nOut <= 0)
                fprintf('Cant have <= 0 outputs\n');
                validated = false;
            end
            
            if(length(sampleMatrix(1,:)) - nOut <= 0)
                fprintf('Cant have <= 0 inputs\n');
                validated = false;
            end
            
            if( length(sampleMatrix(:,1)) <= 0)
                fprintf('Cant have <= 0 samples\n');
                validated = false;
            end
            
            %Validate and initialize learning rate/tolerance
            if(tolerance <= 0)
                fprintf('Warning: Tolerance 0 or below, will only stop after reaching max iterations\n');
            end
            if(maxIterations <= 0)
                fprintf('Max iterations cant be <= 0\n');
                validated = false;
            end
            
            %Initialize userDimensions/numlayers
            if(length(userDimensions) < 2)
                fprintf('Network must have at least 1 hidden layer');
                validated = false;
            end
            
            %Validate and initialize layers
            for i = 1:length(userDimensions)
                if(userDimensions(i) <= 0)
                    fprintf('Cant have a layer with no neurons');
                    validated = false;
                end
            end
        end
        
        function trainNetwork(trainingNetwork)
            iterations = 0;
            
            while(calculateRMSE(trainingNetwork) > trainingNetwork.tolerance && iterations < trainingNetwork.maxIterations)
                iterations = iterations + 1;
                if(mod(iterations, 50) == 0)
                    fprintf('Iteration: %f , RMSE : %f \n', iterations,trainingNetwork.RMSE) %DEBUG
                    %trainingNetwork.layers(2).neurons(1).currentWeights
                end
                errorCorrect(trainingNetwork);
            end
            fprintf('Finish!\n');
            fprintf('Iterations: %f\n', iterations);
            fprintf('Tolerance: %f\n', trainingNetwork.tolerance);
            fprintf('RMSE: %f\n', trainingNetwork.RMSE);
        end
        
        function errorCorrect(trainingNetwork)
            for sampleIndex = 1:trainingNetwork.numSamples
                for layerIndex = trainingNetwork.numLayers:-1:1
                    trainingNetwork.layers(layerIndex).backProp(sampleIndex);
                end
                for layerIndex = 1:trainingNetwork.numLayers
                    %Go forward through each layer and update
                    %weights/outputs
                    trainingNetwork.layers(layerIndex).feedForward(sampleIndex);
                end
            end
        end
        
        function RMSE = calculateRMSE(trainingNetwork) %TODO: FINISH DISCRETE FUNCTION PATH
            systemTotal = 0;            
            for layerNum=1:trainingNetwork.numLayers%I tried to update every neurons current outputs
                 trainingNetwork.layers(layerNum).updateLayerOutputs();%might not work as intended   
            end
           
            for i = 1:trainingNetwork.numSamples
                sampleTotal = 0;
                
                %Get the error of each sample
                
                %For each output output neron
                for j = 1:trainingNetwork.numOutputs
                    desired =mod( angle(trainingNetwork.sampleMatrix(i, j + trainingNetwork.numInputs)),2*pi);
                    %fprintf('Value:%1.2f Angle:%1.2f\n',trainingNetwork.sampleMatrix(i, j + trainingNetwork.numInputs), angle(trainingNetwork.sampleMatrix(i, j + trainingNetwork.numInputs)));
   
                    weightedSum = trainingNetwork.layers(trainingNetwork.numLayers).neurons(j).currentOutputs(i);
                    
                    %This looks good!
                    %if(trainingNetwork.isContinuous)
                        actual = continuousActivation(weightedSum);
                    %else                       
                        %actual = disActivation(weightedSum, trainingNetwork.numSectors);
                    %end
                    
                    actual = mod(angle(actual),2*pi);%We need angles, since this is angular RMSe
                    
                    difference = abs(desired-actual);
                    if(difference > pi)                    %makes sure error is accurate
                        difference = (2*pi) - difference;
                    end
                    
                    sampleTotal = sampleTotal + (difference)^2; %Put ABS here without checking
                    if(trainingNetwork.isContinuous ~= true)
                        
                    end
                end
                sampleTotal = sampleTotal/trainingNetwork.numOutputs;
                
                systemTotal = systemTotal + sampleTotal;
                
            end
            systemTotal = systemTotal / trainingNetwork.numSamples;
            
            RMSE = systemTotal^0.5;
            
            trainingNetwork.RMSE = RMSE;
        end
    end
end
            