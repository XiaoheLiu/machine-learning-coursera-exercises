%% ========== ex6, Part 7: Select (C, sigma) ==========

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Nrange = length(range);

errors = zeros(Nrange.^2, 1); 
% 1st column: C; 2nd column: sigma; 3rd column: error

for n = 1:Nrange
    C = range(n);
    for m = 1:Nrange
        sigma = range(m);
        
        % Train the SVM
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        % Generate prediactions for Xval
        predictions = svmPredict(model, Xval);
        
        % Calculate error on the validation set
        error = mean(double(predictions ~= yval));
        errors((n-1)*Nrange + m) = error;
        
        % Print the result
        fprintf('C=%f, sigma=%f, Error=%f', C, sigma, error);
        
    end
end

%% Report best sigma and C values:

[min_error, idx] = min(errors);
if mod(idx, Nrange) == 0
    best_sigma = range(end)
else
    best_sigma = range(mod(idx, Nrange))
end
best_C = range(floor(idx/Nrange)+1)

