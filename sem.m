function out = sem(in,dim)

% function out = sem(in,dim)
% 
% compute sem of input vector or matrix (1st dim)


if nargin ==1
  out = nanstd(in,[],1)/sqrt(size(in,1));
else
  if dim > ndims(in)
    dim = 1;
  end
  out = nanstd(in,[],dim)/sqrt(size(in,dim));
end


