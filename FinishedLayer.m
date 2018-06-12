classdef FinishedLayer < handle
    properties(SetAccess = public)
        parent; %The network we are part of
        index; %The index of our layer in the network
        
        neurons; %Out list of neurons
        numNeurons; %Number of neurons we have
        
        isOutputLayer; %Is this the last layer 
    end
    methods
        function thisLayer = FinishedLayer(oldLayer, parent)
           thisLayer.parent = parent;
           thisLayer.index = oldLayer.index;
           thisLayer.numNeurons = oldLayer.numNeurons;
           thisLayer.isOutputLayer = oldLayer.isOutputLayer;
           
           thisLayer.neurons = FinishedNeuron.empty();
           for i = 1:thisLayer.numNeurons
               thisLayer.neurons(i) = FinishedNeuron(oldLayer.neurons(i), thisLayer);
           end
        end
        
        function updateOutputs(thisLayer)
            if(thisLayer.index ~= 1)
                for i = 1:thisLayer.numNeurons
                    thisLayer.neurons(i).updateOutput();
                end
            end
        end
    end
end
            