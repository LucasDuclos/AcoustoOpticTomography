%% 
clearvars

Fmax = 50e6;
N = 2^7;
MyImage = TF_t(N,Fmax);


% signal
sigma = 5e-3;
I = exp(-(MyImage.t).^2/sigma^2) + 0.1*rand(1,N);
Ifft = MyImage.fourier(I);

%% plot images FFT
figure(1);
hold on
plot(MyImage.t*1e3,I,'o-')

figure(2);
hold on
semilogy(MyImage.f/1000,abs(Ifft),'-o')
%% pareval relations
% trapz(MyImage.f,abs(Ifft).^2)
% trapz(MyImage.t,abs(I).^2)

clearvars
N = 2^7;
I = exp(-((1:N) - 30).^2/10^2);
Fs = 100;
S = stats_t(Fs);


y = S.PowSpecDens( I );

figure;plot(y)

S.Energy_t( I )
S.Energy_psd( y )














