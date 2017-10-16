function [ ensemble ] = ensembleGeneration( data, M, bagging, r, nFolds )

    if nargin < 5
        nFolds = 10;
    end

    if nargin < 4
        r = 0.8;
    end

    if nargin < 3
        bagging = 1;
    end

    if nargin < 2
        M = 10;
    end

    ensemble = cell(M, 1);

    if bagging
        for i = 1:M
            sample = bag(data,r);
            model = train(sample, 1);
            ensemble{i} = model;           
        end    
    else
        Kopt = modelSelection(data, 0.1:0.1:1, nFolds);
        for i=1:M
            model = train(data, Kopt);
            ensemble{i} = model;           
        end    
    end
end

function [data] = bag(data, r)
    [N, ~] = size(data.x);
    ss = ceil(r*N);
    sel = randperm(N, ss);
    data.x = data.x(sel, :);
    data.y = data.y(sel, :);
end



