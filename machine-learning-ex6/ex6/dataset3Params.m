function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

C_set = [0.01, 0.03, 0.1, 0.3, 1 , 3, 10, 30];
sigma_set = [0.01, 0.03, 0.1, 0.3, 1 , 3, 10, 30];

pred_error = zeros( length(C_set)  , length(sigma_set) );

for m=1:length(C_set)
   
    C = C_set(m);
    
    for n=1:length(sigma_set)
        
    sigma = sqrt( sigma_set(n) );
    
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

    predictions = svmPredict(model, Xval);

    pred_error(m, n) = mean(double(predictions ~= yval));
     
    end
    
end

[m, n] = min( pred_error(:) );

[x ,  y] = ind2sub(size( pred_error ), n);

C = C_set(x);
sigma = sqrt( sigma_set(y) );

end
