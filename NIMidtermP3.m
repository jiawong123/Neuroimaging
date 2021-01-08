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
