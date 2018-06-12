classdef FinishedNetwork < handle
    properties(SetAccess = public)
        valid;
        
        layers;
        numLayers;
        userDimmensions;
        
        isContinuous;
        minVal;
        maxVal;
        numSectors;
        sectorSize;
        
        numInputs;
        numOutputs;
        
    end
    methods
        function networkObj = FinishedNetwork(sampleMatrix, userDimensions, tolerance, maxIterations, minVal, maxVal, isContinuous)
            
            networkObj.valid = true;
            trainingNetwork = TrainingNetwork(sampleMatrix, userDimensions, tolerance, maxIterations, minVal, maxVal, isContinuous);
            if(~trainingNetwork.valid)
                networkObj.valid = false;
                return
            end
            
            networkObj.numLayers = trainingNetwork.numLayers;
            networkObj.isContinuous = trainingNetwork.isContinuous;
            networkObj.minVal = trainingNetwork.minVal;
            networkObj.maxVal = trainingNetwork.maxVal;
            networkObj.numSectors = trainingNetwork.numSectors;
            networkObj.sectorSize = trainingNetwork.sectorSize;
            networkObj.numInputs = trainingNetwork.numInputs;
            networkObj.numOutputs = trainingNetwork.numOutputs;
            
            networkObj.layers= FinishedLayer.empty();
            for i = 1:trainingNetwork.numLayers
                networkObj.layers(i) = FinishedLayer(trainingNetwork.layers(i), networkObj);
            end
            
        end
        
        %Input array is 1 dimmension
        function getOutput(thisNetwork, inputArray)
            nLayers = thisNetwork.numLayers;
            
            %Handle layer 1
            for i = 1:thisNetwork.layers(1).numNeurons  %For each neuron in layer 1
                neuron = thisNetwork.layers(1).neurons(i);
                sum = neuron.weights(1);
                
                
                for j = 2:length(neuron.weights)
                   
                    sum = sum + neuron.weights(j)*inputArray(j-1);
                end
               
                neuron.currentOutput = sum;
            end
            
            for i = 2:nLayers
                thisNetwork.layers(i).updateOutputs();
            end
        end
        
        %Pass a set of known data, check answers against those provided in
        %the data set
        function RMSE = getError(thisNetwork, samples)
            samples = (samples - thisNetwork.minVal)/(thisNetwork.maxVal - thisNetwork.minVal + 1)*2*pi;
            
            if(thisNetwork.isContinuous)
                samples = exp(1i*samples);
            else
                samples = exp((1i * 2*pi *samples)/thisNetwork.numSectors);
            end
            
            desiredPoints = [];
            actualPoints = [];
            desiredPoints(length(samples(:,thisNetwork.numInputs + 1:end)) + 1) = 0;
            actualPoints(length(samples(:,thisNetwork.numInputs + 1:end)) + 1) = 0;
            
            RMSE = 0; 
            for i = 1:length(samples(:,1)) %For each sample
               sampleTotal = 0;
               thisNetwork.getOutput(samples(i, 1:thisNetwork.numInputs));
               
               %For each output neuron
               for j = 1:thisNetwork.numOutputs
                   desired = mod(angle(samples(i, j + thisNetwork.numInputs)),2*pi);
                   weightedSum = thisNetwork.layers(thisNetwork.numLayers).neurons(j).currentOutput; %TODO make sure currentout is updated
                  
                   
                   
                   %fprintf('Value:%1.3f Angle:%1.3f  Mod:%1.3f\n', samples(i, j + thisNetwork.numInputs), angle(samples(i, j + thisNetwork.numInputs)), mod(angle(samples(i, j + thisNetwork.numInputs)),2*pi));
                   
                   %Todo: Is this working? NOPE
                   %if(thisNetwork.isContinuous)
                       actual = continuousActivation(weightedSum);
                   %else                       
                       %actual = disActivation(weightedSum,thisNetwork.numSectors);
                   %end
                   actual = mod(angle(actual),2*pi);%We need angles, since this is angular RMSe
                   
                
                   
                   %For plotting, only works with 1 output
                   actualPoints(i) = actual;
                   desiredPoints(i) = desired;
                    
                   error = (desired - actual)^2; 
                   sampleTotal =  sampleTotal + error;
               end
               sampleTotal = sampleTotal / thisNetwork.numOutputs; %We do this to get the average of the errors.
               RMSE = RMSE + sampleTotal; %When we finish this loop, RMSE will be the average squared error.
            end
            
            fprintf('poop\n')
            disp(actualPoints)
            disp(desiredPoints)
            fprintf('poop\n')
            figure(1);
            hold off
            plot(actualPoints, 'or');
            hold on
            plot(desiredPoints, '*b');
            
            RMSE = RMSE / length(samples(:,1));
            RMSE = RMSE^0.5;
        end
    end
end
            