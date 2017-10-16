function [yhat, error] = predict(model, data, method)
% Minimal Learning Machine --- Out-of sample estimation (test procedure).
%
% [Yh,Error] = predict( model , data, [method] )
%
%
% Inputs:
%          model     a struct containing the model
%                    previously obtained by the
%                    train function.
%
%          data      is a struct with:
%                    data.x  a Nxd matrix.
%                    (optional) data.y  a Nxn matrix 
%                    of outputs (can be multi-output)
%
%
%          [method]  (optional) is the method used for
%					  output estimation.
%                     Default is 'fsolve'.
%
% Outputs:
%          yhat      the estimated output by the model.
%
%          error     the mean square error (for regression problem)
%                    or classification error (for classification 
%					 problem). Computed only if data.y is specified.
%                    
%
% References: 
%
%
%
%

if (nargin < 3)
    method = 'fsolve';
end

DX = pdist2(data.x, model.refPoints.x);
Dyh = DX*model.B;

yhat = outputEstimation(Dyh, model.refPoints.y, method);

error = [];
if(isempty(data.y) == 0)
    error = data.y - yhat;
end

