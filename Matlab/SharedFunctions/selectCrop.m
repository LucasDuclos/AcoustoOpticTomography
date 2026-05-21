function [mask] = selectCrop(path)
addpath('D:\AO--commons\common Data Analysis');
addpath('D:\AO--commons\shared functions folder');

image = double(importdata(path));
imgFFT = fftshift(fft2(image));

% Créer une figure
fig = figure(1);

% Obtenir la taille de l'écran
screensize = get(groot, 'ScreenSize');

% Définir la position et la taille de la figure pour couvrir tout l'écran
fig_position = [1, 1, screensize(3), screensize(4)];
set(fig, 'Position', fig_position);

% Afficher l'image avec imagesc
imagesc(abs(imgFFT));
axis image; % pour afficher les axes avec la même échelle
clim([0 2*mean2(abs(imgFFT))])
% Titre et labels
title('FFT of a raw frame');
xlabel('$\nu_x$');
ylabel('$\nu_z$');
set(gca, 'FontSize', 24);

% Indiquer à l'utilisateur de sélectionner deux points
disp('Cliquez sur deux points sur l''image.');
[x, y] = ginput(2);

filter = ImageFilter([abs(x(1)+x(2))/2 abs(y(1)+y(2))/2 abs(x(1)-x(2)) abs(y(1)-y(2))]);
filter.DrawROI;
pause(1)
close all

mask = filter.getROI(size(imgFFT,2),size(imgFFT,1));
end