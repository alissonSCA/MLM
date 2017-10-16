classdef MLMUtil
    
    methods(Static)
        
        function [encoded_y, labels] = outputEncoding(y, labels)
            if (nargin < 2)
                labels = unique(y);
            end
            
            code = zeros(length(labels));
            for j = 1: length(labels),
                code(j, j) = 1;
            end
            
            encoded_y = zeros(length(y), length(labels));
            for j = length(labels):-1:1,
                ind = (y == labels(j));
                tam = length(find(ind==1));
                encoded_y(ind, :) = repmat(code(j, :), tam, 1);    
            end            
        end
        
        function [decoded_y] = outputDecoding(y, labels)            
            if (nargin < 2)
                labels = 1:1:size(y,2);
            end            
            [~, decoded_y] = max(y,[],2);           
            decoded_y = labels(decoded_y)';
        end
        
        function [accuracy] = getAccuracy(t, yhat)
            if (size(t,2) > 1)
                t = MLMUtil.outputDecoding(t);
            end
            if (size(yhat,2) > 1)
                yhat = MLMUtil.outputDecoding(yhat);
            end
            accuracy = sum(yhat==t)/length(yhat);
        end
        
        function [mse] = getMSE(t, yhat)
            mse = mean((t - yhat).^2);
        end
        
      
    end
end