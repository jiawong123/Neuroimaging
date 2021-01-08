% Solve the problem 1 of Midterm Neuroimaging
% Name: Jia Wang
% ID: 120082910046

clc, clear, close all

%% Generate normally-distributed random matrix
samples = 1000;
groups = 10000;
NA = normrnd(10, 2, [groups,samples]);
NB = normrnd(10.5, 2, [groups,samples]);

%% compute the mean and standard deviation
NAM = mean(NA, 2);
NAstd = std(NA, 0, 2);
NBM = mean(NB, 2);
NBstd = std(NB, 0, 2);

%% display histfit of NAM and NAstd
figure; set(gcf, 'outerposition', get(0,'screensize'));
subplot(121), h = histfit(NAM); h(1).FaceColor = [0 0.75 0.75];
n = get(gca, 'Ylim'); zy = linspace(n(1),n(2));
hold on, p = plot(mean(NAM)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0,0,1];
text(mean(NAM)+0.0125, zy(end-2),['mean = ',num2str(mean(NAM))],'FontSize',14), 
set(gca,'FontSize',16); title('Histogram of NAM', 'Fontsize', 20);

subplot(122), h = histfit(NAstd); h(1).FaceColor = [1 0.85 0];
n = get(gca, 'Ylim'); zy = linspace(n(1),n(2));
hold on, p = plot(mean(NAstd)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [1,0,0];
text(mean(NAstd)+0.0125, zy(end-2),['mean = ',num2str(mean(NAstd))],'FontSize',14), 
set(gca,'FontSize',16); title('Histogram of NAstd', 'Fontsize', 20);
saveas(gcf, '1-Histogram of NAM and NAstd.png')

%% display histfit of NBM and NBstd
figure; set(gcf, 'outerposition', get(0,'screensize'));
subplot(121), h = histfit(NBM); h(1).FaceColor = [0 0.75 0.75];
n = get(gca, 'Ylim'); zy = linspace(n(1),n(2));
hold on, p = plot(mean(NBM)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0,0,1];
text(mean(NBM)+0.0125, zy(end-2),['mean = ',num2str(mean(NBM))],'FontSize',14), 
set(gca,'FontSize',16); title('Histogram of NBM', 'Fontsize', 20);

subplot(122), h = histfit(NBstd); h(1).FaceColor = [1 0.85 0];
n = get(gca, 'Ylim'); zy = linspace(n(1),n(2));
hold on, p = plot(mean(NBstd)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [1,0,0];
text(mean(NBstd)+0.0125, zy(end-2),['mean = ',num2str(mean(NBstd))],'FontSize',14), 
set(gca,'FontSize',16); title('Histogram of NBstd', 'Fontsize', 20);
saveas(gcf, '1-Histogram of NBM and NBstd.png')

%% display Mean in a diagram window
figure; set(gcf, 'outerposition', get(0,'screensize'));
h = histfit(NAM); h(1).FaceColor = [0 0.75 0.75]; h(2).Color = [0,0,1];
n = get(gca, 'Ylim'); zy = linspace(n(1),n(2));
hold on, h = histfit(NBM); h(1).FaceColor = [1 0.85 0]; h(2).Color = [1,0,0];
hold on, p = plot(mean(NAM)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0,0.5,0.5];
hold on, p = plot(mean(NBM)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0.5,0.5,0];
text(mean(NAM)+0.0125, zy(end-2),['mean = ',num2str(mean(NAM))],'FontSize',14), 
text(mean(NBM)+0.0125, zy(end-2),['mean = ',num2str(mean(NBM))],'FontSize',14), 
legend('NAM','NAM-fitted','NBM','NBM-fitted','mean(NAM)','mean(NBM)'), 
set(gca,'FontSize',16); title('Histogram of NAM and NBM', 'Fontsize', 20),
saveas(gcf, '1-Histogram of NAM and NBM.png')

%% display Std in a diagram window
figure; set(gcf, 'outerposition', get(0,'screensize'));
h = histfit(NAstd); h(1).FaceColor = [0 0.75 0.75]; h(2).Color = [0,0,1];
n = get(gca, 'Ylim'); zy = linspace(n(1),n(2));
hold on, h = histfit(NBstd);  h(1).FaceColor = [1 0.85 0]; h(2).Color = [1,0,0]; 
hold on, p = plot(mean(NAstd)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0,0.5,0.5];
hold on, p = plot(mean(NBstd)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0.5,0.5,0];
legend('NAstd','NAstd-fitted','NBstd','NBstd-fitted','mean(NAstd)','mean(NBstd)'), 
set(gca,'FontSize',16); title('Histogram of NAstd and NBstd', 'Fontsize', 20),
saveas(gcf, '1-Histogram of NAstd and NBstd.png')

%% t-test
tN = zeros(groups,1);

for i = 1:groups
    % equal sample size, equal variance
    tN(i) = (NBM(i) - NAM(i)) / sqrt((NAstd(i)^2 + NBstd(i)^2)) / sqrt(1/samples);
end

tN_gt7 = tN > 7;
frac_gt7 = sum(tN_gt7) / groups;

%% display tN
figure; set(gcf, 'outerposition', get(0,'screensize'));
h = histfit(tN); h(1).FaceColor = [0 0.75 0.75];
n = get(gca, 'Ylim'); zy = linspace(n(1),n(2));
hold on, p = plot(mean(tN)*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0,0,1];
text(mean(tN)+0.0125,zy(end-2),['mean = ',num2str(mean(tN)),', std = ',num2str(std(tN))],...
    'FontSize',14),
hold on, p = plot(7*ones(1,numel(zy)),zy); p.LineWidth = 2; p.Color = [0,0.5,0.5];
text(7+0.0125, zy(end-10), ['%(tN > 7) = ', num2str(frac_gt7*100)],'FontSize',14), 
set(gca,'FontSize',16); title('Histogram of tN', 'Fontsize', 20);
saveas(gcf, '1-Histogram of tN.png')

