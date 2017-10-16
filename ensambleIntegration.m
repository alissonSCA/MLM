function [ yhat, error ] = ensambleIntegration( ensembleModels,  data, task, method)
%ENSAMBLEINTEGRATION Summary of this function goes here
%   Detailed explanation goes here

    if ((task == 'c') || (task == 1))
        if(strcmpi(method, 'voting'))
            y = getY(ensembleModels, data, task);
            yhat = mode(y,2);        
        else % weighted voting
            w = getWeights(ensembleModels, data);
            y = getY(ensembleModels, data, task);
            
            N = size(data.y,1);
            S = size(data.y, 2);
            yhat = zeros(N, S);
            for i = 1:N
                si = zeros(S,1);
                for j = 1:S
                    si(j) = sum(w(i, y(i,:) == j));
                end
                [~,ind] = max(si);
                yhat(i,ind) = 1;
            end
        end
    else
        if(strcmpi(method, 'mean'))
            y = getY(ensembleModels, data, task);
            yhat = mean(y,2);        
        else % Multilateration method
            M = size(ensembleModels,1);
            N = size(data.y,1);
            yhat = zeros(N, 1);
            for i = 1:N
                distances = [];
                references = [];
                for m = 1:M
                    B = ensembleModels{m}.B;
                    refPoints = ensembleModels{m}.refPoints;
                    Dx = pdist2(data.x(i,:), refPoints.x);
                    Dy = Dx*B;
%                     disp(size(distances));
%                     disp(size(Dy));
                    distances = [distances Dy];
                    references = [references; refPoints.y];
                end
                yhat(i,:) = outputEstimation(distances, references, 'cubic');
            end
        end        
    end
    
    error = [];
    if(isempty(data.y) == 0)
%         error = data.y - yhat;
    end    
end

function Y = getY(ensemble, data, task)
    M = size(ensemble,1);
    N = size(data.y,1);
    Y = zeros(N,M);
    for m = 1:M
        model = ensemble{m};
        yhat = predict(model, data, 'nn');
        if ((task == 'c') || (task == 1))
            Y(:,m) = MLMUtil.outputDecoding(yhat);
        else
            Y(:,m) = yhat;
        end
    end
end

function w = getWeights(ensemble, data)
    M = size(ensemble,1);
    N = size(data.y,1);
    w = zeros(N,M);
    for m = 1:M
        B = ensemble{m}.B;
        refPoints = ensemble{m}.refPoints;
        Dx = pdist2(data.x, refPoints.x);
        Dy = Dx*B;
        w(:,m) = max(1./Dy, [], 2)./sum(abs(1./Dy), 2);
    end  
end
