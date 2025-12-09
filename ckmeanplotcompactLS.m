function h = ckmeanplotcompactLS(data,errortype,showunit,connectdata,Colorvector,offset,x,symvec)

% function h = ckmeanplotcompact(data,errortype,showunit,connectdata,Colorvector,xpos,symvec)
% compact mean / sem plot with defined colors
% and indvidual data, possibly connected by lines
%S
% errortype
% 1) STD
% 2) SEM
% 3) 95 BootCI
% 4) 99 BootCI

if nargin<3
  showunit = 0;
end
if nargin < 4
  connectdata = 0;
end
if nargin<5
  Colorvector(1,:) = [250,150,0]/250; %
  Colorvector(2,:) = [0,250,150]/250; %
  Colorvector(3,:) = [0,150,250]/250; %
  Colorvector(4,:) = [150,0,250]/250; %
end
if isempty(Colorvector)
  Colorvector(1,:) = [250,150,0]/250; %
  Colorvector(2,:) = [0,250,150]/250; %
  Colorvector(3,:) = [0,150,250]/250; %
  Colorvector(4,:) = [150,0,250]/250; %
end
n = size(data,2);
if nargin<7
  x = [1:n];
end
if isempty(x)
  x=[1:n];
end

if nargin<8
  symvec = '.............................................................';
end

if isempty(showunit), showunit=0; end
if isempty(connectdata), connectdata=0; end

%--------------------------------------------------
% repeat colors if required
if size(Colorvector,1)<size(data,2)
  Colorvector = cat(1,Colorvector,Colorvector);
  Colorvector = cat(1,Colorvector,Colorvector);
  Colorvector = cat(1,Colorvector,Colorvector);
  Colorvector = cat(1,Colorvector,Colorvector);
  Colorvector = cat(1,Colorvector,Colorvector);
end


%--------------------------------------------------
% make mean plots
axpos = get(gca,'Position');
hold on


Mval = nanmean(data);
if errortype==1
  Sval = nanstd(data,[],1);
  Eval(1,:) = Mval-Sval;
  Eval(2,:) = Mval+Sval;
elseif errortype==2
  Sval = nanstd(data,[],1)./sqrt(sum(~isnan(data),1));
  Eval(1,:) = Mval-Sval;
  Eval(2,:) = Mval+Sval;
elseif errortype==3
  Eval = local_boot(data,0.05);
elseif errortype==4
  Eval = local_boot(data,0.01);
end
  
%offset=0.25;
% indiv data
if showunit
  for k=1:n
    xpos = x(k)+offset+randn(size(data,1),1)*offset/4;
    plot(xpos,data(:,k),symvec(k),'Color',Colorvector(k,:),'MarkerSize',25);
    keepX(:,k) = xpos;
  end
end
% connect lines
if connectdata
  for k=1:size(data,1)
    plot(keepX(k,:),data(k,:),'color',[0.6 0.6 0.6]);
  end
end
  
% show data  
for k=1:n
  % mean
  plot(x(k),Mval(k),'.','color',Colorvector(k,:),'MarkerSize',50);
  % sem
  h = line([x(k) x(k)],[Eval(1,k) Eval(2,k)]);
  set(h,'LineWidth',3,'color',Colorvector(k,:));
  plot(x(k),Mval(k),'.','color',[0 0 0],'MarkerSize',25);

end

a = axis;
if showunit==0
  axis([0.5 n+0.5 a(3) a(4)]);
else
  axis([0.5 n+0.75 a(3) a(4)]);
end
set(gca,'Box','off')
set(gca,'XTickLabel',[])

return;



function out = local_boot(x,alpha)
statfun = @nanmean;
B1=2000;
vhat=feval(statfun,x);
vhatstar=bootstrp(B1,statfun,x);

q1=floor(B1*alpha*0.5);
q2=B1-q1+1;
st=sort(vhatstar);
Lo=st(q1,:);
Up=st(q2,:);
out(1,:) = Lo;
out(2,:) = Up;
return
