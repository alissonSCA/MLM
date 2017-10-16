function [yhat] = outputEstimation(Dyh, y, method)
% Method for computing the output from estimated distances in the output
% space (Dyh) and the locating of output reference points Y. 
%
% [yhat] = outputEstimation(Dyh , y, [method])
%
% Inputs:
%
%          Dyh       matrix with estimated distances to
%                    output reference points (Y).
%                    dim(Dyh) = N x K
%
%
%          y         Outputs of the reference points.
%                    dim(y) = K x S
%
%
%          [method]  (optional) is the method used for
%					  estimating the output.
%                     Default is 'fsolve'.
%
% Outputs:
%          yhat        estimated output
%
% References: 
%
%
%
N = size(Dyh, 1);
S = size(y, 2);
K = size(y, 1);
yhat = zeros(N, S);

if(strcmpi(method, 'fsolve'))
    options_fsolve = optimset('Display', 'off', 'Algorithm','levenberg-marquardt', 'FunValCheck', 'on', 'TolFun', 10e-6 );
    yh0 = mean(y); % initial estimate for y
    for i = 1: N,   
        yhat(i, :) = fsolve(@(x)(sum((y - repmat(x, K, 1)).^2, 2) - Dyh(i,:)'.^2), yh0, options_fsolve);
    end
elseif(strcmpi(method, 'nn')) % only for classification
    [~,id] = min(Dyh, [], 2);
    yhat = y(id, :);
elseif (strcmpi(method, 'cubic')) % only for single-output regression
    if (S > 1)
        throw(MException('MULTILATERATION:CUBIC_UNIDIMENSIONAL','Cubic equation method is only avaliable to one-dimensional output'));
    end
    
    A = K; %repmat(K, N, 1);
    B = -3*sum(y); %repmat(-3*sum(y), N, 1);
    C = repmat(3*sum(y.^2), N, 1) - sum(Dyh.^2, 2);
    D = Dyh.^2*y - repmat(sum(y.^3), N, 1);
    
    for i = 1:N
        
        R = roots([A B C(i) D(i)]);
        r = [isreal(R(1)), isreal(R(2)), isreal(R(3))];
        
        if (sum(r) > 1)
            if (sum(r) == 3) % case 1
                J = @(x) sum( ((repmat(x,1,K) - y').^2 -  Dyh(i,:).^2).^2 );
                [~, id] = min([J(R(1)) J(R(2)) J(R(3))]);
                yhat(i) = R(id);
            else %case 2
                id = (r==1);
                yhat(i) = R(id(1));
            end
        else %case 3
            yhat(i) = R(r==1); 
        end       
    end
end




