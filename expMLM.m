clear; clc;
%% Setup
load('./DATA/CLASSIFICATION/iris.mat');
nRuns = 10;
nFoldsCV = 10;
%%
accuracy = zeros(nRuns,1);
for r = 1:nRuns
    %% Split data
    [Xl, Yl, Xt, Yt] = shuffle(X,y,0.8);
    data.x      = Xl;
    data.y      = MLMUtil.outputEncoding(Yl);
    testData.x  = Xt;
    testData.y  = MLMUtil.outputEncoding(Yt);    
    %% Train model
    K = modelSelection(data, 0.1:0.1:1, nFoldsCV);
    model = train(data, K);
    %% Test model
    yhat = predict(model, testData, 'nn');
    accuracy(r,1) = MLMUtil.getAccuracy(testData.y, yhat);        
end
disp(mean(accuracy));