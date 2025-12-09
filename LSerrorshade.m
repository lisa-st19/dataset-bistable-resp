function hout = LSerrorshade(tax,M,SEM,cvec)

% Error-bar lines with shaded background
%
% function h = ckerrorshade(tax,M,SEM,cvec)
% 
% colors are specified from this list  'rbgcmk'
% or [] for red-blue shading of multiple inputs

if nargin<4
  cvec = 'rbgcmk';

else
    cvec = cvec;
end

hold on;
n = size(M,1);

  
  for k=1:n
    %cuse =  strfind(list,cvec(k)); % find matching color entry
    h = fill([tax(1:end) tax(end:-1:1)],[M(k,:)-SEM(k,:) M(k,[end:-1:1])+SEM(k,[end:-1:1])],cvec(n,:),'FaceAlpha',0.4);
    set(h,'EdgeColor',cvec(n,:));
    set(h,'HandleVisibility','off');
  end
  
  for k=1:n
    %cuse =  strfind(list,cvec(k));
    hout(k) =  plot(tax(1:end),M(k,:),'Color',cvec(n,:));
    set(hout(k),'LineWidth',2.5);
  end
  