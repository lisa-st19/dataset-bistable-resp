function out = ck_time_shifting(in)


% randomly shifts each row along 1st dimension independetly
% input (time,trials)

S = size(in);

% the original index for each column
standard_ind = repmat([1:S(1)]',[1,S(2)]);

% random shift for each S(2)
R = ceil(rand(1,S(2)) * S(1));

% we add shift to original index
ind = standard_ind + repmat(R,[S(1),1]);
% now we fold indizes back
ind(ind>S(1)) = ind(ind>S(1))-S(1);


% now we add column number
ind = ind + repmat([1:S(2)]-1,[S(1),1])*S(1);
out = in(ind);
