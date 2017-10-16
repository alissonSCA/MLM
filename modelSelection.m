function [Kopt, error] = modelSelection(data, param, nFolds, method, lambda)

    if (nargin < 3)
        nFolds = 10;
    end
    if (nargin < 4)
        method = 'random';
    end
    if (nargin < 5)
        lambda = 0;
    end

    N = size(data.x, 1);
    if (nFolds == N)
        CVO = cvpartition(N, 'Leaveout');
    else
        CVO = cvpartition(N, 'k', nFolds);
    end
    amseValues = zeros(N, length(param));

    for i = 1: nFolds
        learnPoints.x = data.x(training(CVO, i), :);
        learnPoints.y = data.y(training(CVO, i), :);
        testData.x = data.x(test(CVO, i), :);
        testData.y = data.y(test(CVO, i), :);

        for j = 1: length(param),
            [model] = train(learnPoints, param(j), method, lambda); 

            [amseValues(i, j)] = AMSE(testData, model);
        end
    end

    Ecv = mean(amseValues);
    [error, ind] = min(Ecv);
    Kopt = param(ind);
end

function [amse] =  AMSE(data, model)
    DX = pdist2(data.x, model.refPoints.x);
    DYh = DX*model.B;
    DY = pdist2(data.y, model.refPoints.y);
    errors = (DY - DYh);
    amse = mean(mean(errors.^2));
end