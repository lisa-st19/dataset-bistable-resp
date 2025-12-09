%% Analysis of AK01

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Respiration phase coherence in window around switch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
load('Resp_data.mat');

addpath(genpath('colormaps'));

% Saving figures when running the script or not 
saving_figures = 0; % 0 = not saving, 1 = saving

% cyclic color map
c = crameri('vikO');

% Number of permutations
reps = 4000;
% switch-free window in seconds
d = 0;

% costumized colors 
blue1 = [59, 128, 151]./255;
blue3 = [90 137 175]./255;

% Exclude all participants with >900 & <120 button presses

for k = 1:32
    if Resp_data.Resp_data(k).TotalSwitch > 900 || length(Resp_data.Resp_data(k).All_button_type) < 120
        store(k) = 0;
    else
        store(k) = 1;
    end
    tmp(k,1) = Resp_data.Resp_data(k).TotalSwitch;
    tmp(k,2) = length(Resp_data.Resp_data(k).All_button_type);
end

% figure(1); 
% bar(tmp)
% yline(120,'LineWidth',2,'LineStyle','--'); yline(900,'LineWidth',2,'LineStyle','--');
% title('AK01','FontSize',30,FontWeight='bold');
% t = text(8,1450, 'In all subjects >50% of trials survive resp cleaning');
% t.FontSize = 25;
% legend({'Original trial number','Trial number after resp cleaning','120','900'});

store = logical(store);
Resp_data.Resp_data = Resp_data.Resp_data(store);

for S = 1:length(Resp_data.Resp_data)

    tmp = Resp_data.Resp_data(S).All_button_type;
    idx = find(tmp(:,3)>= d & tmp(:,4)>= d);
    % Storing only the trials in the dataset that meet the criterion
    Resp_data.Resp_data(S).All_button_type = Resp_data.Resp_data(S).All_button_type(idx,:);
    Resp_data.Resp_data(S).All_resp_button = Resp_data.Resp_data(S).All_resp_button(idx,:,:);

    no_kept(S,1) = length(idx);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preparing data to have matching trials of respiration & eye data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('Eye_data.mat');
% Adjust the eye data
for k = 1:32
    if Eye_data.Eye_data(k).TotalSwitch > 900 || length(Eye_data.Eye_data(k).All_button_type) < 120
        store(k) = 0;
    else
        store(k) = 1;
    end
    tmp(k,1) = Eye_data.Eye_data(k).TotalSwitch;
    tmp(k,2) = length(Eye_data.Eye_data(k).All_button_type);
end
store = logical(store);
Eye_data.Eye_data = Eye_data.Eye_data(store);


% Finding all subjects that are in resp AND eye data 
resp_nr = length(Resp_data.Resp_data);
eye_nr = length(Eye_data.Eye_data);

for l = 1:resp_nr
    A(l) = Resp_data.Resp_data(l).SubjectID;
end
for m = 1:eye_nr
    B(m) = Eye_data.Eye_data(m).SubjectID;
end
common_values = intersect(A,B);

% Create logical arrays initialized to zero
new_idx_eye = zeros(size(B));
new_idx_resp = zeros(size(A));

% Mark indices that should be kept with 1
for j = 1:length(common_values)
    new_idx_resp = new_idx_resp | (A == common_values(j));
    new_idx_eye = new_idx_eye | (B == common_values(j));
end

% These are the new datasets that contain the shared subjects
eye = Eye_data.Eye_data(new_idx_eye);
resp = Resp_data.Resp_data(new_idx_resp);

Nsub = length(resp);

for tt = 1:length(resp)
    trlnr(tt,1) = size(resp(tt).All_button_type,1);
    raw_trlnr(tt,1) = resp(tt).TotalSwitch;

   resp_dur = vertcat(resp(tt).Resp.Respiration.Cycle_durations{:});
   avg_resp_dur(tt,1) = nanmean(resp_dur(:,1));
end

group_avg_resp_dur = mean(avg_resp_dur)
group_sd_resp_dur = std(avg_resp_dur)
% translate to frequency
test = 1./avg_resp_dur;
group_avg_resp_dur_Hz = mean(test)
group_sd_resp_dur_Hz = std(test)

avg_trlnr = mean(trlnr)
sd_avg = std(trlnr)

% Average percept duration
for tt = 1:length(resp)
    percept_dur(tt,:) = mean(resp(tt).All_button_type(:,[3 4]));
    percept_dur_sd(tt,:) = std(resp(tt).All_button_type(:,[3 4]));
end
tmp1 = mean(percept_dur); avg_percept = mean(tmp1);
tmp1 = mean(percept_dur_sd); sd_percept = mean(tmp1);


%% Phase coherences 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Respiration phase coherence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % window to cut data around button press
% ARG.fsample = 100;
% ARG.Trange = 4;
% 
% ARG.win = [-ARG.Trange*ARG.fsample:ARG.Trange*ARG.fsample];
% ARG.Accept_range = [3 71]; % range of accepted button press
% tax = ARG.win*10;
% tax_real = tax/1000;
% 
% % Permutation test for phase coherence 
% %PCShuf = zeros(length(Resp_data.Resp_data),size(Resp_data.Resp_data(1).All_resp_button,3),reps);
% PCShuf = zeros(length(resp),size(resp(1).All_resp_button,3),reps);
% 
% % Applying switch-free window around the button press (in seconds, adjust
% % with d)
% 
% for S = 1:length(resp)
%     
%     Phase = squeeze(resp(S).All_resp_button(:,3,:))/100;
% 
%     Phaseangle(S,:) = angle(nanmean(exp(2*pi*i*Phase(:,:))));
%     PC(S,:) = abs(nanmean(exp(2*pi*i*Phase(:,:))));
% 
%     for r=1:reps
%         PCShuf(S,:,r) = abs(nanmean(exp(2*pi*i*ck_time_shifting(Phase')'))); 
%     end
% 
% end
% thr99 = max(prctile(mean(PCShuf,1),99,3));
% thr95 = max(prctile(mean(PCShuf,1),95,3));
% 
% % Within participants PC
% tmp = exp(i*Phaseangle);
% Phaseconsistencyacrossparticipants = abs(nanmean(tmp));
% 
% % % Between-participants PC
% % for r=1:reps
% %     Phaseconsistencyacrossparticipants_Shuf(:,r) = abs(nanmean(ck_time_shifting(tmp'),2));
% % end
% % thr2 = max(prctile(mean(Phaseconsistencyacrossparticipants_Shuf,1),99,3));
% 
% save('PC.mat','PC'); save('Phaseangle.mat','Phaseangle'); 
% save('Phaseconsistencyacrossparticipants.mat','Phaseconsistencyacrossparticipants');
% save('tax_real.mat','tax_real');save('thr99.mat','thr99');save('thr95.mat','thr95');save('tmp.mat','tmp');


%% Plotting

load('PC.mat');load('Phaseangle.mat'); load('Phaseconsistencyacrossparticipants.mat');
load('tax_real.mat');load('thr95.mat');load('thr99.mat');load('tmp.mat');

f1 = figure(1);
set(f1, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]); % [left bottom width hight]

% Panel A
axes('Position',[0.044 0.55 0.275 0.35]); % [left bottom width hight]
tmp1 = PC;
x = tax_real;
y = nanmean(tmp1);
SE = sem(tmp1);
LSerrorshade(x,y,SE,blue3);
hold on;
grid on;
ylim([0.08 0.14]);
yticks(0.08:0.02:0.14);
xlim([-4 4]);
xticks(-4:0.5:4);
xticklabels({'-4','','-3','','-2','','-1','','0','','1','','2','','3','','4'}); 
xtickangle(0);
yline(thr99,':','LineWidth',3,'color','k');
yline(thr95,':','LineWidth',3,'color',[0.6 0.6 0.6]);
set(gca,'FontSize',26);
xlabel('Time around button press [s]');
title({'Within-participant','phase coherence'},'FontWeight','normal');
text(-4.8,0.155,'A','FontSize',30,'FontWeight','bold');

sig_win = find(y>thr95);

% Panel B 
% max phase coherence (0.1174) is at index 512 -> 1.1 s 
max_pc = PC(:,512);
avg_max_pc = mean(max_pc);
ax = axes('Position',[0.38 0.55 0.1 0.35]); % [left bottom width hight]
ckmeanplotcompactLS(max_pc,2,1,0,[70 117 155]./255,0.05);
ylim([0 0.3]);
yticks(0:0.1:0.3);
hold on
yline(thr99,':','LineWidth',3,'color','k');
yline(thr95,':','LineWidth',3,'color',[0.6 0.6 0.6]);
xlim([0.85 1.15]);
set(gca,'FontSize',26);
ax.XGrid = 'off';      
ax.YGrid = 'on';
text(0.77,0.3745,'B','FontSize',30,'FontWeight','bold'); 
title({'Individual','PC'},'FontWeight','normal');

% Panel C
axes('Position',[0.56 0.55 0.28 0.35]); % [left bottom width hight]
imagesc(tax_real,1:Nsub,Phaseangle); colormap(c); 
xticks(-4:1:4);
xticklabels(-4:1:4); 
xtickangle(0);
set(gca,'FontSize',26);
xlabel('Time around button press [s]');
ylabel('Participants');
title({'Epoch-averaged','respiration phase'},'FontWeight','normal');
text(-5.2,-4.72,'C','FontSize',30,'FontWeight','bold');
cbar = colorbar;
ylabel(cbar,'Inspiration              Expiration','FontSize',22,'Rotation',90);
cbar.Label.Position(1) = 1.2;
cbar.TickLength = 0;
cbar.TickLabels = '';

if saving_figures == 1
    % Saving image
    snamef = sprintf('%s/phase_coherence%s.png',figuredir);
    print('-dpng','-r400',snamef);
    snamef = sprintf('%s/phase_coherence%s.tiff',figuredir);
    print('-dtiffn','-r400',snamef);
end

%---------------------------------------------------------------------------------


trialVectors = {raw_trlnr, trlnr};
subplotTitles = {'Raw', 'After preprocessing'};
nPlots = numel(trialVectors);

f10 = figure(10);

for i = 1:nPlots
    x = trialVectors{i};
    nPoints = numel(x);
    % colors for each dot in the plot
    colors = hsv(nPoints); 
    % Correlation
    [r,p] = corrcoef(x,max_pc);
    fprintf('Plot %d: r = %.3f, p = %.3f\n', i, r(1,2), p(1,2));
    % Linear model
    mdl = fitlm(x,max_pc);
    slope = mdl.Coefficients.Estimate(2);   % slope
    pval  = mdl.Coefficients.pValue(2);     % p-value 
    % Polyfit 
    coeffs = polyfit(x,max_pc,1);
    fitValues = polyval(coeffs, x);

    subplot(1, nPlots, i);
    hold on;
    % plot each dot with specific color 
    for j = 1:nPoints
        scatter(x(j), max_pc(j), 50, colors(j,:), 'filled');
    end
    plot(x, fitValues, 'k', 'LineWidth', 2); 

    % Adding text 
    xText = min(x) + 0.05*range(x);  
    yText = max(max_pc) - 0.05*range(max_pc); 
    text(xText, yText, sprintf('Slope = %.6f\np = %.3f', slope, pval), ...
        'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k');

    hold off;
    xlabel('Reversals');
    ylabel('Coherence');
    title(subplotTitles{i});
end



%% Setting up master matrix with behavioural & respiratory data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating 'Master-Matrix' for AIC models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Alldata = [];
count = 1;
nBins = 6; % Binning for RESP FREQUENCY 

tmp = [];

norm_resp_freq = [];
for S = 1:length(resp)
    % Average duration to next button press to normlise trial-wise stability duration
    avg_dur_pre = nanmean(resp(S).All_button_type(:,3));
    avg_dur_next = nanmean(resp(S).All_button_type(:,4));
    % exclude unnormal resp freq. in the window before button press to facilitate binning of frequency
    tmp_resp_dur = resp(S).All_button_type(:,5);
    resp_dur_median = nanmedian(tmp_resp_dur);
    resp_dur_sd = nanstd(tmp_resp_dur);
    ex = tmp_resp_dur > resp_dur_median + 3*resp_dur_sd | tmp_resp_dur < resp_dur_median - 3*resp_dur_sd;
    outlier_cycles(S,1) = sum(ex);
    resp(S).All_button_type(ex,5) = NaN;

    % right now the respiration 'frequency' is given in seconds -> let's
    % convert into actual frequncy values

    in_Hz = 1./resp(S).All_button_type(:,5);
    norm_resp_freq = normalize(in_Hz,'zscore');
    %norm_resp_freq = normalize(resp(S).All_button_type(:,5),'zscore');
    upper_lim = max(norm_resp_freq);
    lower_lim = min(norm_resp_freq);

    spacing = linspace(lower_lim, upper_lim,9);
    bin_idx = discretize(norm_resp_freq,spacing);

    for k = 1:size(resp(1).All_resp_button,3)

        for t = 1:length(resp(S).All_button_type)
            % respiration parameters
            tmp{S}(t,1,k) = resp(S).All_resp_button(t,3,k); % Phase
            tmp{S}(t,2,k) = sin(2*pi*tmp{S}(t,1,k)/100); % Sine
            tmp{S}(t,3,k) = cos(2*pi*tmp{S}(t,1,k)/100); % Cosine
            tmp{S}(t,4,k) = 1/resp(S).All_button_type(t,5); % Respiration frequency shortly before switch report (last 2 resp cycles pre switch)
            tmp{S}(t,5,k) = norm_resp_freq(t,1); % normalized resp. frequency (normalization across all trials within participant)
            tmp{S}(t,6,k) = 0; %bin_idx(t,1); % binned normalized resp. freq.

            % normalized & raw pre- & post-duration
            %-------------------------------------------------------------------------------
            tmp{S}(t,7,k) = resp(S).All_button_type(t,3)/ avg_dur_pre; % normalised stability duration to previous button press
            tmp{S}(t,8,k) = resp(S).All_button_type(t,4)/ avg_dur_next; % normalised stability duration to next button press
            tmp{S}(t,9,k) = resp(S).All_button_type(t,3); % raw stability duration to previous button press
            tmp{S}(t,10,k) = resp(S).All_button_type(t,4); % raw stability duration to next button press

            % normalized pupil size
            %tmp{S}(t,11,k) = eye(S).All_eye_button(t,4,k);
            tmp{S}(t,11,k) = 0;

            % Response button and participant info
            tmp{S}(t,12,k) = resp(S).All_button_type(t,1); % button type (left or right)
            tmp{S}(t,13,k) = resp(S).All_button_type(t,6); % Trial number
            tmp{S}(t,14,k) = resp(S).SubjectID; % subject ID

        end %t
    end %k

    % Allocating binning values to the normalized resp. freq.-> the goal is to
    % have the same number of trials in each bin rather than binning by equally
    % spaced resp freq values - this would lead to having uneven amounts of
    % trials in each bin because there are fewer trials that belong to the
    % outer bins, the 'extreme' resp. frequencies. Most trials will fall into
    % the middle, to the average resp freq.
    x = tmp{S}(:,[5 13]);
    B = sortrows(x,1);
    keep = ~isnan(B(:,1));
    B = B(keep,:);
    kept_trials = length(B);
    bin_length = kept_trials/ (nBins);
    y = repmat(bin_length,1,nBins);
    result = cumsum(y);
    result = round(result);

    % Preallocate vector
    idx = zeros(1, result(end));
    startIdx = 1;
    for ii = 1:length(result)
        idx(startIdx:result(ii)) = ii;
        startIdx = result(ii) + 1;
    end

    idx = idx';
    binIDs = idx(1:kept_trials);
    % Step 1: Get the sorting order you used for B
    [~, sortIdx] = sort(x(:,1));
    % Step 2: Initialize bin assignment vector with NaNs (same length as x)
    bin_column = nan(size(x,1),1);
    % Step 3: Fill only the non-NaN (kept) positions
    nonNanIdx = ~isnan(x(sortIdx,1));  % which ones in sorted x were kept
    bin_column(sortIdx(nonNanIdx)) = binIDs;  % place them back in original order

    tmp{S}(:,6,:) = repmat(bin_column, 1, 1, size(tmp{S},3));

end %S

Alldata = vertcat(tmp{:});

% Alldata contains [RespPhase, Sine, Cosine, RespFreq, NormRespFreq, BinnedNormRespFreq, NormPreDur, NormPostDur, 
% RawPreDur, RawPostDur, NormPupil, Button, TrialID, SubjID]
% concatenated for all participants in a +/- 4 second (800 sample points -> 10ms steps) window around each trial 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Respiration parameters (frequency & phase) ~ Percept stability duration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-------------------------------------------------------------------------------------------------------------
% resp. phase at time of button press sorted into 6 bins 
%-------------------------------------------------------------------------------------------------------------
nbins = 6;
sid = unique(Alldata(:,14));
edges = 0:100/nbins:100;

for S = 1:length(sid)
    idx = Alldata(:,14)==sid(S);
    tmp = Alldata(idx,:,400);

    bin_idx = discretize(tmp(:,1),edges);

    for bin = 1:nbins
        idx = bin_idx == bin;

        NormStabPhase(S,bin,1) = nanmean(tmp(idx,7)); %NormPreDur
        NormStabPhase(S,bin,2) = nanmean(tmp(idx,8)); %NormPostDur

        RawStabPhase(S,bin,1) = nanmean(tmp(idx,9)); %RawPreDur
        RawStabPhase(S,bin,2) = nanmean(tmp(idx,10)); %RawPostDur

    end %bin

end %S



edges2 = 1:nBins;
for S = 1:length(sid)
    idx = Alldata(:,14)==sid(S);
    tmp = Alldata(idx,:,400);
    for freq_bin = 1:length(edges2)

        tmpidx = tmp(freq_bin,6);
        if isnan(tmpidx)
            DurPerPhase(S,freq_bin,2) = NaN;
            DurPerPhase(S,freq_bin,1) = NaN;
        else
            DurPerPhase(S,freq_bin,2) = nanmean(tmp(tmpidx,8));
            DurPerPhase(S,freq_bin,1) = nanmean(tmp(tmpidx,7));
        end

    end % resp freq bin 

end %S

%% plotting
f2 = figure(2);
set(f2, 'Units', 'normalized', 'OuterPosition', [0 0.4 1 0.6]); % [left bottom width hight]
ax1 = axes('Position',[0.378 0.16 0.27 0.7]); % [left bottom width hight] [0.08 0.11 0.38 0.37]

% Panel B
x = angle(exp(i*2*pi*[0:5:100]/100));
cm = crameri('vikO',length(x));
load('ToyTrace.mat','T','tax');
hold on
plot(tax,T)
t0 = find(tax==0);
bins = round([0:t0/9:t0]); bins(1) = 1;
for k=1:length(bins)-1
    j = [bins(k):bins(k+1)];
    p1 = plot(tax(j),T(j),'color',cm(k,:), 'LineWidth',2);
end
bins = round([t0:(length(tax)-t0-1)/9:length(tax)]);
size(bins)
for k=1:length(bins)-1
    j = [bins(k):bins(k+1)];
    p2 = plot(tax(j),T(j),'color',cm(k+11,:), 'LineWidth',2);
end
set(gca,'XTick',[])
set(gca,'YTick',[])
hold on

ax2 = axes('Position',[0.378 0.16 0.27 0.7]); % [left bottom width hight] [0.08 0.11 0.38 0.37]
ax2.Color = 'none';
tmp = NormStabPhase(:,:,2);
cm = crameri('vikO',nbins);
ckmeanplotcompactLS(tmp,2,1,0,cm,0.25);
axis([0.5 0.5+nbins  0.75 1.25]);
grid on
set(gca,'XTick',[]);
text(-0.5,0.63,{'Subsequent percept duration [z]'} ,'Rotation',90,'FontSize',26);
set(gca,'FontSize',26);
ylim([0.65 1.25]);
yticks(0.7:0.1:1.25);
text(-0.75,1.34,'B','FontSize',30,'FontWeight','bold');

text(3.45,0.6355,'|','FontSize',24,'color',[0.5 0.5 0.5]);
text(1.22,0.61,'Inspiration','FontSize',26,'color',[0.5 0.5 0.5]);
text(4.2,0.61,'Expiration','FontSize',26,'color',[0.5 0.5 0.5]);
text(1.9,0.545,'Respiration phase bins','FontSize',26);

% Panel C
ax4 = axes('Position',[0.718 0.16 0.27 0.7]); % [left bottom width hight][0.53 0.27 0.46 0.5]
ax4.Color = 'none';
tmp = DurPerPhase(:,:,1);
cm = crameri('vikO',length(edges2));
ckmeanplotcompactLS(tmp,2,1,0,cm,0.25);
axis([0.5 0.5+length(edges2) 0 5]);
ylim([0 5]);
yticks(0:1:5);
grid on
set(gca,'XTick',[]);
text(-0.3,0,{'Preceding percept duration [z]'} ,'Rotation',90,'FontSize',26);
text(1.3,-0.865,'Respiration frequency [z] bins','FontSize',26);
set(gca,'FontSize',26);
text(-0.65,5.75,'C','FontSize',30,'FontWeight','bold'); % -0.65


%% AIC models respiration ~ reversal behaviour 
%-------------------------------------------------------------------------------------------------------------
% AIC model comparison testing the predictive power of respirartion phase and respiration frequency on percept
% stability duration
%-------------------------------------------------------------------------------------------------------------
% Alldata contains [RespPhase, Sine, Cosine, RespFreq, NormRespFreq, BinnedNormRespFreq, NormPreDur, NormPostDur,
% RawPreDur, RawPostDur, NormPupil, Button, TrialID, SubjID]

c = 1;
for k = 1:20:size(Alldata,3)
    k
    tmp = Alldata(:,:,k);

    T = array2table(tmp, 'VariableNames',{'RespPhase', 'Sine', 'Cosine', 'RespFreq', 'NormRespFreq', 'BinnedNormRespFreq','NormPreDur', ...
        'NormPostDur', 'RawPreDur', 'RawPostDur', 'NormPupil', 'Button', 'TrialID', 'SubjID'});

    % Resp. phase being predictive of PreDur?
    F{1} = 'NormPreDur ~ Sine + Cosine + (1|SubjID) + (1|TrialID)';
    F{2} = 'NormPreDur ~ 1 + (1|SubjID) + (1|TrialID)';
    M1{1,c} = fitlme(T, F{1});
    M2{1,c} = fitlme(T, F{2});

    aics = [M1{1,c}.ModelCriterion.AIC M2{1,c}.ModelCriterion.AIC];
    aics2 = aics-min(aics);
    waic{1}(c,:) =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
    AIC_diff(1,c) = aics(2)-aics(1);

    % Resp. phase being predictive of PostDur?
    F{1} = 'NormPostDur ~ Sine + Cosine + (1|SubjID) + (1|TrialID)';
    F{2} = 'NormPostDur ~ 1 + (1|SubjID) + (1|TrialID)';
    M1{2,c} = fitlme(T, F{1});
    M2{2,c} = fitlme(T, F{2});

    aics = [M1{2,c}.ModelCriterion.AIC M2{2,c}.ModelCriterion.AIC];
    aics2 = aics-min(aics);
    waic{2}(c,:) =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
    AIC_diff(2,c) = aics(2)-aics(1);

    c = c + 1;

    % Since the respiration frequency is for all data points in the window around the button press the same, the model
    % comparison can be done for any of those data points and the result being the same for all data points in window
    if k == round(size(Alldata,3)/2)

        % Resp. frequency being predictive of PreDur?
        F{1} = 'NormPreDur ~ NormRespFreq + (1|SubjID) + (1|TrialID)';
        F{2} = 'NormPreDur ~ 1 + (1|SubjID) + (1|TrialID)';
        M_freqPre1 = fitlme(T, F{1});
        M_freqPre2 = fitlme(T, F{2});

        aics = [M_freqPre1.ModelCriterion.AIC M_freqPre2.ModelCriterion.AIC];
        aics2 = aics-min(aics);
        waic_freq1 =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
        AIC_diff_freq(1) = aics(2)-aics(1);

        % Resp. freqeuncy being predicitve of PostDur?
        F{1} = 'NormPostDur ~ NormRespFreq + (1|SubjID) + (1|TrialID)';
        F{2} = 'NormPostDur ~ 1 + (1|SubjID) + (1|TrialID)';
        M_freqPost1 = fitlme(T, F{1});
        M_freqPost2 = fitlme(T, F{2});

        aics = [M_freqPost1.ModelCriterion.AIC M_freqPost2.ModelCriterion.AIC];
        aics2 = aics-min(aics);
        waic_freq2 =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
        AIC_diff_freq(2) = aics(2)-aics(1);
    end % if k
end %k

% f3 = figure(3);
% plot(AIC_diff','Linewidth',2); 
% xlim([3.5 38.5]);
% xticks(6:5:36);
% xticklabels(-3:1:3);
% grid on
% xlabel('Time around button press [s]');
% ylabel('Delta AIC');
% set(gca,'FontSize',26);
% legend({'PreDur ~ Resp. Phase','PostDur ~ Resp. Phase'});

%% plotting
ax = axes('Position',[0.044 0.16 0.265 0.7]); % [left bottom width hight] [0.08 0.58 0.38 0.4]
plot(AIC_diff(1,:),'LineWidth',2.5,'color',[0.7 0.7 0.7]);
hold on 
plot(AIC_diff(2,:),'LineWidth',2.5,'color','k');
xlim([1 41]);
%ylim([-4 2.5]);
xticks(1:5:41);
xticklabels(-4:1:4);
grid on
xlabel('Time around button press [s]','FontSize',26);
text(-4,-0.6,'Delta AIC','Rotation',90,'FontSize',26);
set(gca,'FontSize',26);
text(-5,7.5,'A','FontSize',30,'FontWeight','bold');

if saving_figures == 1
    snamef = sprintf('%s/PerceptDur_RespParameters%s.png',figuredir);
    print('-dpng','-r400',snamef);
    snamef = sprintf('%s/PerceptDur_RespParameters%s.tiff',figuredir);
    print('-dtiffn','-r400',snamef);
end

%% Matching trials of eyetracking data and resp data

% For the following analysis steps we need the pupil data, so we first have
% to match the trials in both datasets.

% Now we have to identify the single trials that are shared between resp and eye WITHIN a subject

for k = 1:length(eye)

    A = eye(k).All_button_type(:,5);
    B = resp(k).All_button_type(:,6);
    common_values = intersect(A,B);

    % Create logical arrays initialized to zero
    new_idx_eye = zeros(size(A));
    new_idx_resp = zeros(size(B));

    % Mark indices that should be kept with 1
    for j = 1:length(common_values)
        new_idx_eye = new_idx_eye | (A == common_values(j));
        new_idx_resp = new_idx_resp | (B == common_values(j));
    end

    % Convert logical indexing to actual filtering
    eye(k).All_button_type = eye(k).All_button_type(new_idx_eye, :);
    eye(k).All_eye_button =  eye(k).All_eye_button(new_idx_eye, :, :);
    resp(k).All_button_type = resp(k).All_button_type(new_idx_resp, :);
    resp(k).All_resp_button = resp(k).All_resp_button(new_idx_resp, :, :);

    clean_switches(k,1) = length(resp(k).All_button_type);
end

avg_trnl_pupil = mean(clean_switches)
sd_trnl_pupil = std(clean_switches)

% We also need again the Alldata matrix, but now also column 11 filled with
% pupil size data
% For easier distinction and to not overwrite Alldata (only with resp
% data), the new matrix is called Alldata_pupil

Alldata_pupil = [];
count = 1;

tmp = [];

norm_resp_freq = [];
for S = 1:length(resp)
    % Average duration to next button press to normlise trial-wise stability duration
    avg_dur_pre = nanmean(resp(S).All_button_type(:,3));
    avg_dur_next = nanmean(resp(S).All_button_type(:,4));
    % exclude unnormal resp freq. in the window before button press to facilitate binning of frequency
    tmp_resp_dur = resp(S).All_button_type(:,5);
    resp_dur_median = nanmedian(tmp_resp_dur);
    resp_dur_sd = nanstd(tmp_resp_dur);
    ex = tmp_resp_dur > resp_dur_median + 3*resp_dur_sd | tmp_resp_dur < resp_dur_median - 3*resp_dur_sd;
    resp(S).All_button_type(ex,5) = NaN;

    norm_resp_freq = normalize(resp(S).All_button_type(:,5),'zscore');
    upper_lim = max(norm_resp_freq);
    lower_lim = min(norm_resp_freq);

    spacing = linspace(lower_lim, upper_lim,9);
    bin_idx = discretize(norm_resp_freq,spacing);

    for k = 1:size(resp(1).All_resp_button,3)

        for t = 1:length(resp(S).All_button_type)
            % respiration parameters
            tmp{S}(t,1,k) = resp(S).All_resp_button(t,3,k); % Phase
            tmp{S}(t,2,k) = sin(2*pi*tmp{S}(t,1,k)/100); % Sine
            tmp{S}(t,3,k) = cos(2*pi*tmp{S}(t,1,k)/100); % Cosine
            tmp{S}(t,4,k) = 1/resp(S).All_button_type(t,5); % Respiration frequency shortly before switch report (last 2 resp cycles pre switch)
            tmp{S}(t,5,k) = norm_resp_freq(t,1); % normalized resp. frequency (normalization across all trials within participant)
            tmp{S}(t,6,k) = bin_idx(t,1); % binned normalized resp. freq.
            % normalized & raw pre- & post-duration

            %-------------------------------------------------------------------------------
            tmp{S}(t,7,k) = resp(S).All_button_type(t,3)/ avg_dur_pre; % normalised stability duration to previous button press
            tmp{S}(t,8,k) = resp(S).All_button_type(t,4)/ avg_dur_next; % normalised stability duration to next button press
            tmp{S}(t,9,k) = resp(S).All_button_type(t,3); % raw stability duration to previous button press
            tmp{S}(t,10,k) = resp(S).All_button_type(t,4); % raw stability duration to next button press

            % normalized pupil size
            tmp{S}(t,11,k) = eye(S).All_eye_button(t,4,k);

            % Response button and participant info
            tmp{S}(t,12,k) = resp(S).All_button_type(t,1); % button type (left or right)
            tmp{S}(t,13,k) = resp(S).All_button_type(t,6); % Trial number
            tmp{S}(t,14,k) = resp(S).SubjectID; % subject ID

        end %t
    end % k

    % For later stats on pupil size devition from zero
    PupilForStats(S,:) = nanmean(squeeze(tmp{S}(:,11,:)));
end %S

Alldata_pupil = vertcat(tmp{:});

%% Model comparison: Resp - Pupil, Percept duration - Pupil

% Alldata_pupil contains [RespPhase, Sine, Cosine, RespFreq, NormRespFreq,
% BinnedNormRespFreq, NormPreDur, NormPostDur, RawPreDur, RawPostDur, 
% NormPupil, Button, TrialID, SubjID]

c = 1;
for k = 1:20:size(Alldata_pupil,3)
    k
    tmp = Alldata_pupil(:,:,k);

    T = array2table(tmp, 'VariableNames',{'RespPhase', 'Sine', 'Cosine', 'RespFreq', 'NormRespFreq', 'BinnedNormRespFreq','NormPreDur', ...
        'NormPostDur', 'RawPreDur', 'RawPostDur', 'NormPupil', 'Button', 'TrialID', 'SubjID'});

    % Pupil size being predictive of PreDur?
    F{1} = 'NormPreDur ~ NormPupil + (1|SubjID) + (1|TrialID)';
    F{2} = 'NormPreDur ~ 1 + (1|SubjID) + (1|TrialID)';
    M1 = fitlme(T, F{1});
    M2 = fitlme(T, F{2});

    aics = [M1.ModelCriterion.AIC M2.ModelCriterion.AIC];
    aics2 = aics-min(aics);
    waic{3}(c,:) =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
    AIC_diff(3,c) = aics(2)-aics(1);

    % Pupil size being predictive of PreDur?
    F{1} = 'NormPostDur ~ NormPupil + (1|SubjID) + (1|TrialID)';
    F{2} = 'NormPostDur ~ 1 + (1|SubjID) + (1|TrialID)';
    M1 = fitlme(T, F{1});
    M2 = fitlme(T, F{2});

    aics = [M1.ModelCriterion.AIC M2.ModelCriterion.AIC];
    aics2 = aics-min(aics);
    waic{4}(c,:) =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
    AIC_diff(4,c) = aics(2)-aics(1);

    % Resp. phase being predictive of Pupil size?
    F{1} = 'NormPupil ~ Sine + Cosine + (1|SubjID) + (1|TrialID)';
    F{2} = 'NormPupil ~ 1 + (1|SubjID) + (1|TrialID)';
    M1 = fitlme(T, F{1});
    M2 = fitlme(T, F{2});

    aics = [M1.ModelCriterion.AIC M2.ModelCriterion.AIC];
    aics2 = aics-min(aics);
    waic{5}(c,:) =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
    AIC_diff(5,c) = aics(2)-aics(1);

    c = c + 1;
end %k


%% Arranging pupil data for model pupil ~ resp phase & plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting pupil trace around button press & comparing 'active' vs. 'passive' windows 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1) Using the whole subblock to sort into phase bins and than filtering pupil size by those bins
% 2) Using +/-3 seconds around reversals to determine resp phase and pupil size 

% In resp.Resp.Resp.DispOn is a cell for each block that contains for each subblock (75s sessions) the 
% respiratory signal (col1), the mask (col2), and the resp. phase (col3)
% rows = subblocks, columns = resp. signal properties, z-Dimension = time course of the whole block (100 Hz)

% Pupil size in respiration phase bins for whole recording

nbins = 8;
edges = 0:100/nbins:100;

Alltime = [];
for S = 1:length(eye)
    AllTrialsResp = [];
    AllTrialsEye = [];
    all_trnum = [];
    all_bnum = [];
    for block = 1:4
        TR = [];
        TE = [];
        for trial = 1:6
            tmpResp = squeeze(resp(S).Resp.RespDispOn{block}(trial,3,:));
            tmpResp = resample(tmpResp',1,10,'Dimension',2);
            tmpResp = tmpResp';

            TR = [TR;tmpResp];

            tmpEye = eye(S).Eye{block}(trial,:);
            tmpEye = resample(tmpEye,1,10,'Dimension',2);
            tmpEye = tmpEye';

            TE = [TE;tmpEye];

            % adding the trial number for later AICs
            trnum = repmat(trial,length(tmpEye),1);
            all_trnum = [all_trnum; trnum];

            % adding block number for later AICs
            bnum = repmat(block,length(tmpEye)*6,1);

        end % trial
        AllTrialsResp = [AllTrialsResp; TR];
        AllTrialsEye = [AllTrialsEye; TE];

        all_bnum = [all_bnum; bnum];
        
    end % block
    tmp_s = repmat(resp(S).SubjectID,length(all_bnum),1);

    tmp_time = [AllTrialsResp, AllTrialsEye, all_bnum, all_trnum, tmp_s];
    
    Alltime = [Alltime; tmp_time];

    bin_idx = discretize(AllTrialsResp(:,1),edges);
    for bin = 1:nbins
        idx = bin_idx == bin;
        PupilSize(S,bin) =  nanmean(AllTrialsEye(idx,1));
    end % bin


    resp(S).All_resp_button = resample(resp(S).All_resp_button,1,10,'Dimension',3);
    eye(S).All_eye_button = resample(eye(S).All_eye_button,1,10,'Dimension',3);
    win_phase = [];
    win_pupil = [];

    for k = 1:size(resp(S).All_resp_button,1)
        win_tmp_R = squeeze(resp(S).All_resp_button(k,3,:));
        win_tmp_P = squeeze(eye(S).All_eye_button(k,4,:));

        win_phase = [win_phase; win_tmp_R];
        win_pupil = [win_pupil; win_tmp_P];

    end

    win_bin = discretize(win_phase(:,1),edges);
    for bin = 1:nbins
        idxW = win_bin == bin;
        WinPupilSize(S,bin) =  nanmean(win_pupil(idxW,1));
    end % bin

    clear idx idxW;

end % S

%% AIC model pupil ~resp phase
% In Alltime each row resembles a 100 Hz sample point. Similar to previous
% AICs we will go in 200 Hz steps -> every second row is not used 

new = Alltime(1:2:end,:);
new(find(sum(isnan(new),2)),:)=[];

for i_time = 1:length(new)
    ph(i_time,1) = sin(2*pi*new(i_time,1)/100); % Sine
    ph(i_time,2) = cos(2*pi*new(i_time,1)/100); % Cosine
end

tmp = [];
tmp = [ph, new(:,2:end)];
T = array2table(tmp, 'VariableNames',{'Sine', 'Cosine', 'Pupil', 'BlockID', 'TrialID', 'SubjID'});

F{1} = 'Pupil ~ Sine + Cosine + (1|SubjID) + (1|TrialID)'; % + (1|BlockID)';
F{2} = 'Pupil ~ 1 + (1|SubjID) + (1|TrialID)'; % + (1|BlockID)';
M1 = fitlme(T, F{1});
M2 = fitlme(T, F{2});
aics = [M1.ModelCriterion.AIC M2.ModelCriterion.AIC];
aics2 = aics-min(aics);
waic_time =  exp(-0.5*aics2)./(sum(exp(-0.5*aics2),2));
AIC_diff_time = aics(2)-aics(1);


%% Plotting
f4 = figure(4);
set(f4, 'Units', 'normalized', 'OuterPosition', [0 0.4 1 0.6]); % [left bottom width hight]

% % Panel C 
ax1 = axes('Position',[0.735 0.16 0.26 0.7]); % [left bottom width hight]
x = angle(exp(i*2*pi*[0:5:100]/100));
cm = crameri('vikO',length(x));
load('ToyTrace.mat','T','tax');
hold on
plot(tax,T)
t0 = find(tax==0);
bins = round([0:t0/9:t0]); bins(1) = 1;
for k=1:length(bins)-1
    j = [bins(k):bins(k+1)];
    p1 = plot(tax(j),T(j),'color',cm(k,:), 'LineWidth',2);
end
bins = round([t0:(length(tax)-t0-1)/9:length(tax)]);
size(bins)
for k=1:length(bins)-1
    j = [bins(k):bins(k+1)];
    p2 = plot(tax(j),T(j),'color',cm(k+11,:), 'LineWidth',2);
end
set(gca,'XTick',[]);
set(gca,'YTick',[]);
hold on

ax2 = axes('Position',[0.735 0.16 0.26 0.7]); % [left bottom width hight]
ax2.Color = 'none';
tmp = PupilSize;
cm = crameri('vikO',nbins);
ckmeanplotcompactLS(tmp,2,1,0,cm,0.25);
axis([0.5 0.5+nbins  -0.4 0.2]);
grid on
set(gca,'XTick',[]);
text(-0.9,-0.25,{'Pupil Size [z]'} ,'Rotation',90,'FontSize',26);
set(gca,'FontSize',26);
ylim([-0.45 0.25]);
yticks(-0.4:0.1:0.2);
text(-1.2,0.33,'C','FontSize',30,'FontWeight','bold');
%
text(4.4,-0.465,'|','FontSize',24,'Color',[0.5 0.5 0.5]);
text(1.3,-0.48,'Inspiration','FontSize',26,'Color',[0.5 0.5 0.5]);
text(5.5,-0.48,'Expiration','FontSize',26,'Color',[0.5 0.5 0.5]);
text(2.15,-0.57,'Respiration phase bins','FontSize',26);

% % Panel B 
ax3 = axes('Position',[0.41 0.16 0.26 0.7]); % [left bottom width hight]
x = angle(exp(i*2*pi*[0:5:100]/100));
cm = crameri('vikO',length(x));
load('ToyTrace.mat','T','tax');
hold on
plot(tax,T)
t0 = find(tax==0);
bins = round([0:t0/9:t0]); bins(1) = 1;
for k=1:length(bins)-1
    j = [bins(k):bins(k+1)];
    p1 = plot(tax(j),T(j),'color',cm(k,:), 'LineWidth',2);
end
bins = round([t0:(length(tax)-t0-1)/9:length(tax)]);
size(bins)
for k=1:length(bins)-1
    j = [bins(k):bins(k+1)];
    p2 = plot(tax(j),T(j),'color',cm(k+11,:), 'LineWidth',2);
end
set(gca,'XTick',[]);
set(gca,'YTick',[]);
hold on

ax4 = axes('Position',[0.41 0.16 0.26 0.7]); % [left bottom width hight]
ax4.Color = 'none';
tmp = WinPupilSize;
cm = crameri('vikO',nbins);
ckmeanplotcompactLS(tmp,2,1,0,cm,0.25);
axis([0.5 0.5+nbins  -0.4 0.2]);
grid on
set(gca,'XTick',[]);
text(-0.9,-0.25,{'Pupil Size [z]'} ,'Rotation',90,'FontSize',26);
set(gca,'FontSize',26);
ylim([-0.45 0.25]);
yticks(-0.4:0.1:0.2);
text(-1.2,0.33,'B','FontSize',30,'FontWeight','bold');
%
text(4.4,-0.465,'|','FontSize',24,'Color',[0.5 0.5 0.5]);
text(1.3,-0.48,'Inspiration','FontSize',26,'Color',[0.5 0.5 0.5]);
text(5.5,-0.48,'Expiration','FontSize',26,'Color',[0.5 0.5 0.5]);
text(2.15,-0.57,'Respiration phase bins','FontSize',26);

%% Stats on diviation of pupil size from zero in +/- 4 s window around reversal
% 
% %pupil (participants, time)
% Pupil = PupilForStats;
% Pupil = Pupil-mean(Pupil,2);
% cfg=[];
% cfg.p1level = 0.05;
% cfg.critvaltype ='par';% prctile' % type of threshold to apply. Usual 'par'
% cfg.critval = abs(tinv(cfg.p1level/2,size(Pupil,1)-1)) ;% critical cutoff value for cluster members if parametric
% cfg.clusterstatistic ='maxsum';
% cfg.minsize = 4; % minimal cluster size 
% cfg.pval = 0.05; % threshold to select signifciant clusters
% cfg.conn = 4;
% cfg.Nsample = size(Pupil,1);
% n_perm = 20000;
% T_true = squeeze(mean(Pupil)./sem(Pupil));
% nsub = size(Pupil,1);
% T_Perm = zeros(size(Pupil,2),n_perm);
% for b = 1:n_perm
% rs = 2*double(rand(nsub,1)>0.5)-1; % random sign
% rs = repmat(rs,[1,size(Pupil,2)]);
% T_Perm(:,b) = squeeze(mean(Pupil.*rs)./sem(Pupil.*rs));
% end
% [PosClus,NegClus] = eegck_clusterstats(cfg,T_true,T_Perm);
% 
% 
% % Extrahiere signifikante Zeitfenster (mehrere Cluster pro Richtung)
% sig_thresh = 0.04;
% AllClusWindows = [];
% clusStructs = {};
% 
% if exist('PosClus','var') && ~isempty(PosClus)
%     clusStructs{end+1} = PosClus;
% end
% if exist('NegClus','var') && ~isempty(NegClus)
%     clusStructs{end+1} = NegClus;
% end
% 
% for c = 1:numel(clusStructs)
%     Clus = clusStructs{c};
%     
%     if isstruct(Clus) && all(isfield(Clus, {'p','mask'}))
%         for i = 1:numel(Clus.p)
%             if Clus.p(i) < sig_thresh
%                 mask_i = (Clus.mask == i);  % Cluster i
%                 
%                 dmask = diff([0; mask_i(:); 0]);
%                 start_idx = find(dmask == 1);
%                 end_idx   = find(dmask == -1) - 1;
%                 
%                 if ~isempty(start_idx)
%                     clus_windows = [start_idx end_idx];
%                     AllClusWindows = [AllClusWindows; clus_windows];
%                 end
%             end
%         end
%     end
% end
% 
% if ~isempty(AllClusWindows)
%     AllClusWindows = sortrows(AllClusWindows,1);
%     disp('Signifikante Zeitfenster (Indices):');
%     disp(AllClusWindows);
% else
%     disp('Keine Cluster unter p < 0.04 gefunden.');
% end

%% Plotting
% Pupil size in respiration phase in window around button press

% Alldata contains [RespPhase, Sine, Cosine, RespFreq, NormRespFreq, BinnedNormRespFreq, NormPreDur, NormPostDur,
% RawPreDur, RawPostDur, NormPupil, Button, TrialID, SubjID]
tmp = [];
for k = 1:size(Alldata_pupil,3) 
    tmp(1,k) = mean(Alldata_pupil(:,11,k));
    tmp(2,k) = sem(Alldata_pupil(:,11,k));
end
tax = 1:size(Alldata_pupil,3);
min_y = min(tmp(1,:));
step = 0.05;
min_y = floor(min_y/step)*step;
max_y = max(tmp(1,:)); 
max_y = ceil(max_y/step)*step;

load('AllClusWindows.mat');

ax = axes('Position',[0.06 0.16 0.28 0.7]); % [left bottom width hight]
ax.Color = 'none';
yl = [min_y max_y];
hold on
%--- Schattierte Bereiche für alle Clusterfenster zeichnen --- %%
if exist('AllClusWindows','var') && ~isempty(AllClusWindows)
    for i = 1:size(AllClusWindows,1)
        % Clusterfenster-Indizes
        clus_idx = AllClusWindows(i,:);
        % Zeitachsenwerte, die zu diesen Indizes gehören
        realtime_clus = tax(clus_idx(1):clus_idx(2));
        % Fläche zeichnen (über gesamte y-Achse)
        fill([realtime_clus(1) realtime_clus(end) realtime_clus(end) realtime_clus(1)], ...
             [yl(1) yl(1) yl(2) yl(2)], ...
             [0.6 0.6 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    end
end

LSerrorshade(tax,tmp(1,:),tmp(2,:),blue1);
xlim([1 801]);
xticks(1:100:801);
xticklabels(-4:1:4);
ylim(yl);
%yticks(-0.1:0.05:0.1)
xlabel('Time around button press [s]');
ylabel('Pupil size [z]');
grid on
set(gca,'FontSize',26);
text(-167,0.129,'A','FontSize',30,'FontWeight','bold');%-167

if saving_figures == 1
    snamef = sprintf('%s/PupilSize_RespPhase%s.png',figuredir);
    print('-dpng','-r400',snamef);
    snamef = sprintf('%s/PupilSize_RespPhase%s.tiff',figuredir);
    print('-dtiffn','-r400',snamef);
end
