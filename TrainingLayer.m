classdef TrainingLayer < handle
    properties(SetAccess = public)
        parent; %The network we are part of
        index; %The index of our layer in the network
        
        prevLayer; %pointer to previuous layer
        nextLayer; %pointer to next layer
        
        neurons; %Out list of neurons
        numNeurons; %Number of neurons we have
        
        isOutputLayer; %Is this the last layer 
    end
    methods
        function trainingLayerObj = TrainingLayer(parent, index, numNeurons)
            %layerObj.neurons(1,numOfNeurons) = mvNeuron(neuronAmountPrev);
            trainingLayerObj.parent = parent;
            trainingLayerObj.index = index;
            trainingLayerObj.numNeurons = numNeurons;
            trainingLayerObj.isOutputLayer = false;
            
            %Get the previous layer if we are not the first layer
            %Set ourself as that layers next layer
            if(index ~= 1)
                trainingLayerObj.prevLayer = parent.layers(index - 1);
                trainingLayerObj.prevLayer.nextLayer = trainingLayerObj;
            end
            
            %Get the number of neurons in the next layer
            if(index == parent.numLayers)
                trainingLayerObj.isOutputLayer = true;
            end
            
            %Initialize the size of the neurons array 
            %and fill it with new neurons
            %trainingLayerObj.neurons = zeros(1, numNeurons);
            trainingLayerObj.neurons = TrainingNeuron.empty();
            for i = 1 : numNeurons
                trainingLayerObj.neurons(i) = TrainingNeuron(trainingLayerObj, i);
            end
            
        end
        
        function backProp(thisLayer, sample)
            for i = 1:thisLayer.numNeurons
                thisLayer.neurons(i).backProp(sample);
            end
        end
        
        function feedForward(thisLayer, sample)
            for i = 1:thisLayer.numNeurons
                thisLayer.neurons(i).feedForward(sample);
            end
        end
        
        function updateLayerOutputs(thisLayer)
            for neuronIndex=1: thisLayer.numNeurons
                for sampleRowIndex=1: thisLayer.parent.numSamples
                    weightedSum=thisLayer.neurons(neuronIndex).getOutput(sampleRowIndex);
                    thisLayer.neurons(neuronIndex).currentOutputs(sampleRowIndex)=weightedSum;
                end
            end
        end
        
    end
end