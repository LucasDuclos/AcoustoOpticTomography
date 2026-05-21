addpath('D:\AO--commons\common Data Analysis');
addpath('D:\AO--commons\shared functions folder');

path = "\\bazar\acousto-optique\ExperimentalData\2024-04-10-HOLLANDE-EtalonnageSignalHolographique\Ahmat\Holo\Holo1.tif";

% Charger votre image
mask = selectCrop(path);

imagesc(mask.*abs(imgFFT))
clim([0 2*mean2(abs(imgFFT))])
