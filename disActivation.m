function [ comOutput ] = disActivation( weightedSum,valueAmount )
    sizeOfSector = 2*pi/valueAmount;

    %Changed: discrete output is no longer returned; could not think of a
    %use for it.
    arg = angle(weightedSum);  %I find the position of the weigted sum on the unit circle, through it's phase angle
    arg = mod(arg,2*pi);       %and perform modulo 2pi, on the phase angle, to fit it from 0 to 2pi                                                                                                                                    
    
    disOutput = floor(arg/sizeOfSector);    %then I find the sector which the phase angle belongs to
                                            %along with the corresponding
                                            %discrete value for the sector
    
    comOutput = exp( (1i * 2*pi * disOutput )/valueAmount);
                                            %I send back the discrete value
                                            %converted to a complex number
                                            %for use in weight adjustment
    
end
