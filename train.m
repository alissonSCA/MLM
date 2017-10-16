function [model] = train(data, K, method, lambda)
% Minimal Learning Machine --- Training procedure
%
% [model] = train( data , [K] , [method], 
%                        [lambda])
%
%
% Inputs:
%          data       is a struct comprised of:
%                     data.x  a NxD matrix of variables
%                     data.y  a NxS matrix of outputs
%                             (can be multi-output)
%
%          [K]   	  (optional) represents the number of reference
%					  points. If K is between 0 and 1, we use it 
%					  as a percentage of the number of the learning
%					  points; otherwise, K (integer) denotes the exact
%					  number of reference points.
%					  Default is '0.5' (50% of data points).
%
%          [method]  (optional) is the method used to
%						select the reference points from data.
%                     Default is 'random'.
%
%          [lambda]  (optional) coefficient for Tikonov 
%						regularization.
%                     Default is 0 (no regularization).
%%
%
% Output:
%          [model]   a struct containing the parameters model.B and 
%					 the set of reference points in the input and
%					output spaces model.refPoints.x 
%					 model.refPoints.y.
%
% References: 
%

%
if (nargin < 2)
    K = 0.5;
end
if (nargin < 3)
    method = 'random';
end
if (nargin < 4)
    lambda = 0;
end

% Removing duplicates on the data
[data.x, ia, ~] = unique(data.x, 'rows');
data.y = data.y(ia, :);

N = size(data.x, 1);
if (K <= 1)
    K = floor(N*K);
end

% Selection of reference points
refPoints = selectReferencePoints(data, K, method);

% Compute the pairwise distances
Dx = pdist2(data.x, refPoints.x);
Dy = pdist2(data.y, refPoints.y);

% Computing the linear model B
if(lambda ~= 0 )
    model.B = (Dx'*Dx + lambda.*eye(N))\(Dx'*Dy); 
else
    model.B =  pinv(Dx)*Dy;
end

% Saving the MLM model
model.refPoints = refPoints;