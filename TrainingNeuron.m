classdef TrainingNeuron < handle
   properties(Constant)
       MIN_WEIGHT = -0.5;
       MAX_WEIGHT = 0.5;
   end
   properties(SetAccess = public)
      parent; %The layer this neuron is part of. We can get the network from layer
      index; %The index of this Neuron in its parent layer
      currentErrors; %A vector of errors, one for each learning set. Updated during back prop
      currentWeights; %A vector of weights, one for each input, + w0. Updated during feed forward
      currentOutputs;  %A vector of outputs, one for each learning set. Updated during feed forward
      isOutputNeuron; %If this Neuron is in last layer, getErros and gerOutputs are handled differently (ex: discreet/contuous)  
   end
   methods
      function newTNeuron = TrainingNeuron(parent, index)
            
         
          newTNeuron.parent = parent;
          newTNeuron.index = index;
         
          
          %Keep track of EACH SAMPLES error/outputs
          newTNeuron.currentErrors = zeros(1, newTNeuron.parent.parent.numSamples);
          newTNeuron.currentOutputs = zeros(1, newTNeuron.parent.parent.numSamples);
            
           
          
          if(~parent.isOutputLayer)
            newTNeuron.isOutputNeuron = false;
          else
            newTNeuron.isOutputNeuron = true;
          end
          
          %Generate starting weights
          if(parent.index == 1)
            numWeights = parent.parent.numInputs + 1;
          else
            numWeights = parent.prevLayer.numNeurons + 1;
          end
            
          
          MAX_WEIGHT = newTNeuron.MAX_WEIGHT;
          MIN_WEIGHT = newTNeuron.MIN_WEIGHT;
          realpart = ((MAX_WEIGHT-MIN_WEIGHT).*rand(1,numWeights)+(MIN_WEIGHT));%Real part of the weights
          imagpart = (1i.*(MAX_WEIGHT-MIN_WEIGHT).*rand(1, numWeights)+ 1i*(MIN_WEIGHT)); %Imaginary part of the weights   
          
          newTNeuron.currentWeights = realpart + imagpart;
          
          
          %Generate starting outputs, bascially random
          for i = 1:newTNeuron.parent.parent.numSamples
            newTNeuron.currentOutputs(i) = newTNeuron.getOutput(i);
          end
      end
      
      function feedForward(thisNeuron, learningSample)
          
          %Update weights, handled differently for first, normal and last layer.
          if(thisNeuron.parent.index == 1)
              thisNeuron.currentWeights(1) = thisNeuron.currentWeights(1) + thisNeuron.currentErrors(learningSample)/(length(thisNeuron.currentWeights)/abs(thisNeuron.currentOutputs(learningSample)));
              for i = 2:length(thisNeuron.currentWeights)
                thisNeuron.currentWeights(i) = thisNeuron.currentWeights(i) + thisNeuron.currentErrors(learningSample)/(length(thisNeuron.currentWeights)/abs(thisNeuron.currentOutputs(learningSample)))*conj(thisNeuron.parent.parent.sampleMatrix(learningSample, i-1));
              end
          elseif(thisNeuron.parent.isOutputLayer)
              thisNeuron.currentWeights(1) = thisNeuron.currentWeights(1) + thisNeuron.currentErrors(learningSample)/length(thisNeuron.currentWeights);
              for i = 2:length(thisNeuron.currentWeights)
                thisNeuron.currentWeights(i) = thisNeuron.currentWeights(i) + thisNeuron.currentErrors(learningSample)/length(thisNeuron.currentWeights)*conj(continuousActivation(thisNeuron.parent.prevLayer.neurons(i-1).currentOutputs(learningSample)));
              end
          else
              thisNeuron.currentWeights(1) = thisNeuron.currentWeights(1) + thisNeuron.currentErrors(learningSample)/abs(thisNeuron.currentOutputs(learningSample))/length(thisNeuron.currentWeights);
              for i = 2:length(thisNeuron.currentWeights)
                thisNeuron.currentWeights(i) = thisNeuron.currentWeights(i) + thisNeuron.currentErrors(learningSample)/abs(thisNeuron.currentOutputs(learningSample))/length(thisNeuron.currentWeights)*conj(continuousActivation(thisNeuron.parent.prevLayer.neurons(i-1).currentOutputs(learningSample)));
              end
          end
          
          %Update our current output after adjusting the weights
          thisNeuron.currentOutputs(learningSample) = getOutput(thisNeuron, learningSample);
      end
      
      function backProp(thisNeuron, learningSample) 
          %Get the error of the neuron based on next neuron
          if thisNeuron.parent.isOutputLayer
              desired=thisNeuron.parent.parent.sampleMatrix(learningSample, thisNeuron.index + thisNeuron.parent.parent.numInputs);
              
              
              %if(thisNeuron.parent.parent.isContinuous == true)                  
                  actual = continuousActivation(thisNeuron.currentOutputs(learningSample)); 
              %else                 
                 %get the actual output, by using activation function on weighted sum
                  %actual = disActivation(thisNeuron.currentOutputs(learningSample),thisNeuron.parent.parent.numSectors);
              %end
              
               backError = desired - actual;
                
              thisNeuron.currentErrors(learningSample) = backError/length(thisNeuron.currentWeights);
          else
            total = 0;
              for i = 1:thisNeuron.parent.nextLayer.numNeurons
                  total = total + thisNeuron.parent.nextLayer.neurons(i).currentErrors(learningSample)/thisNeuron.parent.nextLayer.neurons(i).currentWeights(thisNeuron.index + 1);
              end
              
              thisNeuron.currentErrors(learningSample) = total / length(thisNeuron.currentWeights);
          end
      end
     
      function output = getOutput(thisNeuron, learningSample) %this function created the weighted sum for a neuron, based on the sample index
          %Use the previous neurons outputs and this neurons weights to get
          %this neurons output. 
          
          
          output = thisNeuron.currentWeights(1);
          if(thisNeuron.parent.index ~= 1)
              for i = 2:length(thisNeuron.currentWeights)
                  output = output + thisNeuron.currentWeights(i)*continuousActivation(thisNeuron.parent.prevLayer.neurons(i-1).currentOutputs(learningSample));%Hack 1 
              end
          else
              for i = 2:length(thisNeuron.currentWeights)
                                    
                  output = output + thisNeuron.currentWeights(i)*thisNeuron.parent.parent.sampleMatrix(learningSample, i-1); 
              end
          end
      end
   end
end