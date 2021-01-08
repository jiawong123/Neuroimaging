% Solve the problem 2 of Midterm Neuroimaging
% Name: Jia Wang
% ID: 120082910046

clc, clear, close all

%% Initialization
f = @(a,phi,d,f,t) a * exp(1i * phi) * exp((-d + 1i * 2*pi * f) * t);
K = 5;  % the number of types of metabolite components
MB = cell(K,5);
MB(:,1) = {'NAA', 'Cr', 'Cho', 'MI', 'Lipid'};        % Metabolite
MB(:,2) = {10.3, 4.8, 3.2, 1.5, 0.8};                 % a, amplitude (a.u)
MB(:,3) = {0, pi, pi/2, 0, pi/6};                     % phi, phase (rad)
MB(:,4) = {0.025, 0.02, 0.015, 0.015, 0.01};          % d, damping factor (Hz)
MB(:,5) = {0.8285, 0.8925, 0.9053, 0.9232, 0.7504};   % f, frequency (Hz)

sigma2 = [2, 5];% noise level
delta = 1;      % sample interval (s)
N = 1024;       % data length

%% Compute synthesized MRS time series
time = (0:N-1) * delta;
y1 = zeros(1,N);    % with noise level of sigma^2 = 2 

%% Lorentzian model
for n = 1:N
    for k = 1:K
        y1(n) = y1(n) + f(MB{k,2},MB{k,3},MB{k,4},MB{k,5},time(n));
    end
end

y2 = y1;            % with noise level of sigma^2 = 5

y1 = y1 + sqrt(sigma2(1)/2) * (normrnd(0,1,[1,N]) + 1i * normrnd(0,1,[1,N]));
y2 = y2 + sqrt(sigma2(2)/2) * (normrnd(0,1,[1,N]) + 1i * normrnd(0,1,[1,N]));

%% display MRS in time domain
figure, set(gcf, 'outerposition', get(0,'screensize'));
subplot(121), plot(time, abs(y1)), xlim([-20,time(end)+20]),
set(gca,'FontSize',16); xlabel('sample points')
title(['MRS in time domain, sigma^2 = ', num2str(sigma2(1))], 'Fontsize', 18);

subplot(122), plot(time, abs(y2)), xlim([-20,time(end)+20]), 
set(gca,'FontSize',16); xlabel('sample points')
title(['MRS in time domain, sigma^2 = ', num2str(sigma2(2))], 'Fontsize', 18);
saveas(gcf, '2-MRS in time domain.png')

figure, set(gcf, 'outerposition', get(0,'screensize'));
plot(time, abs(y1)), hold on
plot(time, abs(y2)), xlim([-20,time(end)+20]),
set(gca,'FontSize',16); xlabel('sample points')
title('MRS in time domain', 'Fontsize', 18);
legend(['sigma^2 = ', num2str(sigma2(1))], ['sigma^2 = ', num2str(sigma2(2))])
saveas(gcf, '2-MRS in time domain - superimpose.png')

%% display MRS in frequency domain
yf1 = fft(y1);              % FFT of y1
yf2 = fft(y2);              % FFT of y2
fs = 1 / delta;             % sample frequency
freq = linspace(0, fs, N);  % frequency series

% amplitude
figure, set(gcf, 'outerposition', get(0,'screensize'));
subplot(121), plot(freq, abs(yf1)), xlim([-0.0125,freq(end)+0.0125]), 
set(gca,'FontSize',16); xlabel('frequency (Hz)');
title(['Amplitude of MRS in frequency domain, sigma^2 = ', num2str(sigma2(1))],...
    'Fontsize', 18);

subplot(122), plot(freq, abs(yf2)), xlim([-0.0125,freq(end)+0.0125]), 
set(gca,'FontSize',16); xlabel('frequency (Hz)');
title(['Amplitude of MRS in frequency domain, sigma^2 = ', num2str(sigma2(2))],...
    'Fontsize', 18);
saveas(gcf, '2-Amplitude of MRS in frequency domain.png')

% real value
figure, set(gcf, 'outerposition', get(0,'screensize'));
subplot(121), plot(freq, real(yf1)), xlim([-0.0125,freq(end)+0.0125]), 
set(gca,'FontSize',16); xlabel('frequency (Hz)');
title(['Real value of MRS in frequency domain, sigma^2 = ', num2str(sigma2(1))],...
    'Fontsize', 18);

subplot(122), plot(freq, real(yf2)), xlim([-0.0125,freq(end)+0.0125]),
set(gca,'FontSize',16); xlabel('frequency (Hz)');
title(['Real value of MRS in frequency domain, sigma^2 = ', num2str(sigma2(2))],...
    'Fontsize', 18);
saveas(gcf, '2-Real value of MRS in frequency domain.png')

% imaginary value
figure, set(gcf, 'outerposition', get(0,'screensize'));
subplot(121), plot(freq, imag(yf1)), xlim([-0.0125,freq(end)+0.0125]), 
set(gca,'FontSize',16); xlabel('frequency (Hz)');
title(['Imaginary value of MRS in frequency domain, sigma^2 = ', num2str(sigma2(1))],...
    'Fontsize', 18);

subplot(122), plot(freq, imag(yf2)), xlim([-0.0125,freq(end)+0.0125]), 
set(gca,'FontSize',16); xlabel('frequency (Hz)');
title(['Imaginary value of MRS in frequency domain, sigma^2 = ', num2str(sigma2(2))],...
    'Fontsize', 18);
saveas(gcf, '2-Imaginary value of MRS in frequency domain.png')
