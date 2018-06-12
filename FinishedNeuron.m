classdef FinishedNeuron < handle
    properties(SetAccess = public)
      parent; %The layer this neuron is part of. We can get the network from layer
      index; %The index of this Neuron in its parent layer
      weights; %A vector of weights, one for each input, + w0. Updated during feed forward
      currentOutput;
      isOutputNeuron;
    end
    methods
        function thisNeuron = FinishedNeuron(oldNeuron, parent)
            thisNeuron.parent = parent;
            thisNeuron.index = oldNeuron.index;
            thisNeuron.isOutputNeuron = oldNeuron.isOutputNeuron;
            thisNeuron.weights = oldNeuron.currentWeights;
        end
        
        function updateOutput(thisNeuron)
            prevLayerIndex = thisNeuron.parent.index-1;
            
            if(thisNeuron.parent.index ~=1)
                sum = thisNeuron.weights(1);
                
                for i = 2:length(thisNeuron.weights)
                    sum = sum + thisNeuron.weights(i)*continuousActivation(thisNeuron.parent.parent.layers(prevLayerIndex).neurons(i-1).currentOutput);
                end
                thisNeuron.currentOutput = sum;
            end
        end
    end
end