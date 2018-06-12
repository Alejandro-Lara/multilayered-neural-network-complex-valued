input = [3 4 5 6 7 8]; 
samples = [0  1  2  3  4  5 6;
 1  2  3  4  5  6 7;
 2  3  4  5  6  7 8;
 3  4  5  6  7  8 9;
 4  5  6  7  8  9 10;
 5  6  7  8  9 10 11;
 6  7  8  9 10 11 12];

%mappingFile = load('samples.mat', 'MackeyGlass600Samples');
mapping = load('learnArith.txt');
%pumpernuckle = load('IrisMVN-3_Test-90-60.txt');

%mapping=mappingFile.MackeyGlass600Samples;
%mapping(5,:)
%mapping = exp(mapping *1i);
%mapping(5,:)
%mapping = mod(angle(mapping),2*pi);
%mapping(5,:)
dimensions = [4 1];
tolerance = 0.001; 
maxIterations = 200;

testNet = FinishedNetwork(mapping, dimensions, tolerance, maxIterations, 0, 30, true);
if(testNet.valid)
    fprintf('Test Error(angular) : %f \n',testNet.getError(samples));
end
