function [refPoints] = selectReferencePoints(data, K, method)
% Method for selecting the reference points in Minimal Learning Machines.
%
% [refPoints] = selectReferencePoints( data , K, [method] )
%
% Inputs:
%
%          data      is a struct with:
%                    data.x  a Nxd matrix (inputs)
%                    data.y  a Nxn matrix (outputs).
%
%          K         Number of reference points:
%                    Either a real number between [0, 1]
%                    or an integer > 1.
%
%          [method]  (optional) is the method used for
%					  selecting the reference points.
%                     Default is 'random'.
%
% Outputs:
%          refPoints struct comprised of reference points 
%                    in the input space (refPoints.x) and output
%                    space (refPoints.y).
%
% References: 
%
%
%
%
if(strcmpi(method, 'random')),
    ind = randperm(size(data.x,1));
    refPoints.x = data.x(ind(1:K), :);
    refPoints.y = data.y(ind(1:K), :);
end
