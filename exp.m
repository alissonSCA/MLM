clear; clc;
%% Setup
nRuns = 10;
nFoldsCV = 10;
P = 0.8;
% dbPathClass = '../uci-classification-data/';
dbPathClass = './DATA/CLASSIFICATION/';
dbPathRegre = '../uci-regression-data/';
%% Classification
fprintf('Data\tMLM\tV-MLM\tRSV-MLM\tWV-MLM\tRSWV-MLM\n');
dbNames = dir(strcat(dbPathClass, '*-paper.mat'));
for dsId = 1:numel(dbNames)
    fprintf('\n%s',dbNames(dsId).name);
    dataset = load(strcat(dbPathClass, dbNames(dsId).name));
    
    accuracy = zeros(nRuns, 5);
    for r = 1:nRuns
        data.x      = dataset.xtrain{r};
        data.y      = MLMUtil.outputEncoding(dataset.ytrain{r});
        testData.x  = dataset.xtest{r};
        testData.y  = MLMUtil.outputEncoding(dataset.ytest{r});    
        
        %MLM
        K = modelSelection(data, 0.1:0.1:1, nFoldsCV);
        model = train(data, K);
        yhat = predict(model, testData, 'nn');
        accuracy(r,1) = MLMUtil.getAccuracy(testData.y, yhat);
        %V-MLM
        ensemble = ensembleGeneration( data, 10, 0, 0, nFoldsCV );
        yhat = ensambleIntegration( ensemble,  testData, 'c', 'voting');
        accuracy(r,2) = MLMUtil.getAccuracy(testData.y, yhat);
        %RSV-MLM
        ensemble = ensembleGeneration( data, 10, 1, P );
        yhat = ensambleIntegration( ensemble,  testData, 'c', 'voting');
        accuracy(r,3) = MLMUtil.getAccuracy(testData.y, yhat);
        %V-MLM
        ensemble = ensembleGeneration( data, 10, 0, 0, nFoldsCV );
        yhat = ensambleIntegration( ensemble,  testData, 'c', 'w-voting');
        accuracy(r,4) = MLMUtil.getAccuracy(testData.y, yhat);
        %RSV-MLM
        ensemble = ensembleGeneration( data, 10, 1, P );
        yhat = ensambleIntegration( ensemble,  testData, 'c', 'w-voting');
        accuracy(r,5) = MLMUtil.getAccuracy(testData.y, yhat);        
    end
    
    for i = 1:5
        fprintf('\t%1.4f %1.4f',mean(accuracy(:,i)), std(accuracy(:,i)));
    end
    
end
fprintf('\nDone.\n');
%% Regression
fprintf('Data\tMLM\tA-MLM\tRSA-MLM\tJ-MLM\tRSJ-MLM\n');
dbNames = dir(strcat(dbPathRegre, '*-paper.mat'));
for dsId = 1:numel(dbNames)
    fprintf('\n%s',dbNames(dsId).name);
    dataset = load(strcat(dbPathRegre, dbNames(dsId).name));
    
    mse = zeros(nRuns, 5);
    for r = 1:nRuns
        data.x      = dataset.xtrain{r};
        data.y      = dataset.ytrain{r};
        testData.x  = dataset.xtest{r};
        testData.y  = dataset.ytest{r};    
        
        %MLM
        K = modelSelection(data, 0.1:0.1:1, nFoldsCV);
        model = train(data, K);
        yhat = predict(model, testData, 'cubic');
        mse(r,1) = MLMUtil.getMSE(testData.y, yhat);
        %V-MLM
        ensemble = ensembleGeneration( data, 10, 0, 0, nFoldsCV );
        yhat = ensambleIntegration( ensemble,  testData, 'r', 'mean');
        mse(r,2) = MLMUtil.getMSE(testData.y, yhat);
        %RSV-MLM
        ensemble = ensembleGeneration( data, 10, 1, P );
        yhat = ensambleIntegration( ensemble,  testData, 'r', 'mean');
        mse(r,3) = MLMUtil.getMSE(testData.y, yhat);
        %V-MLM
        ensemble = ensembleGeneration( data, 10, 0, 0, nFoldsCV );
        yhat = ensambleIntegration( ensemble,  testData, 'r', 'J');
        mse(r,4) = MLMUtil.getMSE(testData.y, yhat);
        %RSV-MLM
        ensemble = ensembleGeneration( data, 10, 1, P );
        yhat = ensambleIntegration( ensemble,  testData, 'r', 'J');
        mse(r,5) = MLMUtil.getMSE(testData.y, yhat);        
    end
    
    for i = 1:5
        fprintf('\t%1.4f %1.4f',mean(mse(:,i)), std(mse(:,i)));
    end
    
end
fprintf('\nDone.\n');