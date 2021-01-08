# Neuroimaging: Midterm

</br>

1. Create 10000 different arrays of 1000 normally-distributed random numbers with a mean of 10 and a standard deviation of 2 (be sure to type a semi-colon at the end of the line or your screen will become cluttered with random numbers), Named as matrix NA. Create another array at the same size, NB, with normally distributed numbers having a mean of 10.5 and a standard deviation of 2. Find the mean and standard deviation of each row in NA and NB (10000 rows in all), named as NAM, NBM, NAstd and NBstd. Plot and send me the histograms of NAM, NBM, NAstd and NBstd. Notice that there is a range of both standard deviations and means. Calculate 10000 t statistics of the difference of means between NA and NB (saved as array tN). Plot and send me the histogram of tN. What is the mean of your experimentally determined tN? What is the standard deviation of tN? Approximately what fraction of the time would your experimentally measured t statistic be greater than 7? (25points)

   *Hint: Programming with MATLAB.*

   </br>

   **Solution:**

   &emsp;The histograms of NAM, NAstd, NBM, NBstd are as below. The mean of NAM, NAstd, NBM, NBstd is

   $$
   \begin{aligned}
   \overline{X}_{\mathrm{NAM}} &= 10.0008 \quad,\quad \overline{X}_{\mathrm{NAstd}} = 1.9990 \\
   \overline{X}_{\mathrm{NBM}} &= 10.5004 \quad,\quad \overline{X}_{\mathrm{NBstd}} = 1.9995
   \end{aligned}
   $$

   <img src=".//1-Histogram of NAM and NAstd.png"></img>

   <img src=".//1-Histogram of NBM and NBstd.png"></img>

   &emsp;Since that the variables NA and NB have the equal sample size and equal variance, then perform independent two-sample t-test to the difference of means between NA and NB,

   $$
   t = \frac{\overline{X}_1 - \overline{X}_2}{\sqrt{\frac{1}{2}(S^2_{X_1}+S^2_{X_2})}\sqrt{\frac{2}{n}}}
   $$

   &emsp;Then, the histogram of tN is as below. The mean and standard deviation of tN is

   $$
   \overline{X}_{\mathrm{tN}} = 5.5881 \quad,\quad S_{\mathrm{tN}} = 1.0004
   $$

   &emsp;And approximately 8.1% of tN are greater than 7.

   <img src=".//1-Histogram of tN.png"></img>

   &emsp;MATLAB Code (Next page)

   ```MATLAB{class=line-numbers}
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
   ```

   </br>

2. Construct a synthesized magnetic resonance spectroscopy (MRS) time series using the Lorentzian model:

   $$
   y_n = \sum_{k=1}^{K} a_k e^{j\phi_k} e^{(-d_k + j2\pi f_k)t_n} + e_n \quad,\quad n=0,1,\cdots,N-1
   $$

   Where $y_n$ is the $n$th measured data point, ($a_k,\;\phi_k,\;d_k,\;f_k$) are the amplitude, phase, damping factor and frequency of the $k$th component, $t_n = 1s$ as the data sampling interval, and $e_n$ is the circular complex white Gaussian noise. Here the data length is assumed as $N=1024$. The metabolite component parameters were assigned as in Table 1:

   <img src="./Table 1.bmp"></img>

   Using various noise level $\sigma^2 =2 \;\mathrm{and}\; 5$, respectively.
   Plot the corresponding absolute value of MRS signal in time domain.Perform a fast Fourier transform (FFT) to obtain the frequency domain representation of MRS signal (i.e. the spectrum). Plot the absolute value, real value and imaginary value of MRS spectrum separately. Send me the codes and plots. (35points)

   *Hint: Programming with MATLAB. Might use “fft” function for transformation.*

   </br>

   **Solution:**

   &emsp;The synthesized MRS signals in time domain are as below.
   &emsp;Since the circular complex white Gaussian noises are high enough, the MRS signal after 150s is almost submerged by white noise. MRS signal with noise level of $\sigma^2=5$ is more affected.

   <img src=".//2-MRS in time domain.png"></img>

   &emsp;After preform FFT to MRS signal, the absolute value, real value and imaginary value of MRS spectrum are as below.

   &emsp;In the figure of amplitude of MRS spectrum, NAA has the most significant spectral peak and the peaks of Cr and Cho are also obvious, but they are very close. In contrast, the spectral peaks of MI and Lipid are completely masked by white noise.

   <img src=".//2-Amplitude of MRS in frequency domain - labeled.png"></img>

   <img src=".//2-Real value of MRS in frequency domain.png"></img>

   <img src=".//2-Imaginary value of MRS in frequency domain.png"></img>

   &emsp;MATLAB Code (Next page)

   ```MATLAB{class=line-numbers}
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
   ```

   </br>

3. Given the raw speckle image files in the data folder ‘Dat’ (80 frames altogether), calculate the spatial LSI (using a 3*3 window) contrast for each frame. By averaging all the contrast frames, draw the final sLASCA image. Implement temporal LSI (using a 20-frame window) too and draw the final tLASCA image. Apply eLASCA to both images and plot the new figures (2 figures). Send me the codes and plots. All images in gray map. (40 points)

   *Hint: Programming with MATLAB. Might use “imread” and “imagesc” functions for image loading and plotting. Try to improve the computational efficiency by intelligent matrix manipulation.*

   </br>

   **Solution:**

   &emsp;The sLASCA image, enhanced sLASCA (EsLASCA) image, tLASCA image, and enhanced tLASCA (EtLASCA) image are as below.

   &emsp;After apply eLASCA to both images, the contrasts of both images are significantly improved. Small blood vessels are clearly visible.

   <img src=".//3-sLASCA and EsLASCA.png"></img>

   <img src=".//3-tLASCA and EtLASCA.png"></img>

   &emsp;MATLAB Code (Next page)

   ```MATLAB{class=line-numbers}
   % Solve the problem 3 of Midterm Neuroimaging
   % Name: Jia Wang
   % ID: 120082910046

   clc, clear, close all

   %% Initialization
   path = 'Dat';
   filelist = dir(path);
   filelist = filelist(~ismember({filelist.name},{'.','..'}));
   len = length(filelist);
   imgS = cell(1, len);            % the raw speckle image series

   swinS = 3;                      % Size of spatial window
   swh = round((swinS-1) / 2);     % Half of spatial window

   twinS = 20;                     % Size of temporal window
   twh = round(twinS / 2);         % Half of temporal window

   %% Load data
   for k = 1:len
      path_sub = strcat(filelist(k).folder, '\', filelist(k).name);
      imgS{k} = imread(path_sub);
   end

   [rows, cols] = size(imgS{1});
   imgS = reshape(cell2mat(imgS), [rows,cols,len]);
   imgS = double(imgS);

   %% compute the spatial LSI contrast (LASCA)
   LASCA = zeros(rows, cols, len);                  % spatial LSI contrast (LASCA)

   for m = 1+swh : rows-swh
      for n = 1+swh : cols-swh
         sw = imgS(m-swh:m+swh, n-swh:n+swh, :);    % pixels in a spatial window
         p1 = squeeze(sum(sum(sw.*sw)) / swinS^2);
         p3 = squeeze(sum(sum(sw)) / swinS^2);
         p2 = p3.^2;
         LASCA(m,n,:) = sqrt(p1 - p2) ./ p3;
      end
   end

   %% compute sLASCA
   sLASCA = sum(LASCA,3) / len;

   % display sLASCA
   figure, imshow(sLASCA, []), title('sLASCA', 'Fontsize', 20)
   saveas(gcf, '3-sLASCA.png')
   figure, imagesc(sLASCA), title('sLASCA', 'Fontsize', 20), colorbar
   saveas(gcf, '3-sLASCA-color.png')

   %% compute temporal LSI (tLSI), without zeros padding
   tLSI = zeros(rows, cols, len-twinS+1);  % temporal LSI (tLSI)

   for k = twh : len-twh
      tw = imgS(:, :, k-twh+1:k+twh);     % pixels in a temporal window
      p1 = sum(tw.*tw, 3) / twinS;
      p3 = sum(tw, 3) / twinS;
      p2 = p3.^2;
      tLSI(:,:,k-twh+1) = sqrt(p1 - p2) ./ p3;
   end

   %% compute tLASCA
   tLASCA = sum(tLSI,3) / size(tLSI,3);

   % display
   figure, imshow(tLASCA, []), title('tLASCA', 'Fontsize', 20)
   saveas(gcf, '3-tLASCA.png')
   figure, imagesc(tLASCA), title('tLASCA', 'Fontsize', 20), colorbar
   saveas(gcf, '3-tLASCA-color.png')

   %% apply eLASCA to sLASCA
   EsLASCA = zeros(rows,cols);    % Enhanced sLASCA
   for m = 1:rows
      for n = 1:cols
         EsLASCA(m,n) = sum(sum(sLASCA < sLASCA(m,n))) / rows / cols;
      end
   end

   % display EsLASCA
   figure, imshow(EsLASCA, []), title('EsLASCA', 'Fontsize', 20)
   saveas(gcf, '3-EsLASCA.png')
   figure, imagesc(EsLASCA), title('EsLASCA', 'Fontsize', 20), colorbar
   saveas(gcf, '3-EsLASCA-color.png')

   %% apply eLASCA to tLASCA
   EtLASCA = zeros(rows,cols);    % Enhanced sLASCA
   for m = 1:rows
      for n = 1:cols
         EtLASCA(m,n) = sum(sum(tLASCA < tLASCA(m,n))) / rows / cols;
      end
   end

   % display EsLASCA
   figure, imshow(EtLASCA, []), title('EtLASCA', 'Fontsize', 20)
   saveas(gcf, '3-EtLASCA.png')
   figure, imagesc(EtLASCA), title('EtLASCA', 'Fontsize', 20), colorbar
   saveas(gcf, '3-EtLASCA-color.png')

   %% compare
   figure, set(gcf, 'outerposition', get(0,'screensize'));
   subplot(121), set(gca,'FontSize',16);colorbar
   imshow(sLASCA, []), title('sLASCA', 'Fontsize', 20),
   subplot(122), set(gca,'FontSize',16);colorbar
   imshow(EsLASCA, []), title('EsLASCA', 'Fontsize', 20)
   saveas(gcf, '3-sLASCA and EsLASCA.png')

   figure, set(gcf, 'outerposition', get(0,'screensize'));
   subplot(121), set(gca,'FontSize',16);colorbar
   imshow(tLASCA, []), title('tLASCA', 'Fontsize', 20)
   subplot(122), set(gca,'FontSize',16);colorbar
   imshow(EtLASCA, []), title('EtLASCA', 'Fontsize', 20)
   saveas(gcf, '3-tLASCA and EtLASCA.png')
   ```
