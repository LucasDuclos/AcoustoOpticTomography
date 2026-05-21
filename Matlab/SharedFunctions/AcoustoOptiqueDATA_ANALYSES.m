%% =============== add subfolders ===============  %%

 addpath('D:\AO--commons\shared functions folder')
 addpath('D:\AO--commons\common subfunctions')
%  addpath('gui');
%  addpath('sequences');
%  addpath('subfunctions');

%( for simulated datas: go to D:\AO--simulations\Noise and run "main.m" file )
 
 %% =============== load datas ===============  %%
 
 % data : 13/11/2019:
 % importdata('Q:\datas\2019-11-08\Signal_modeTwicking_RepHz_100_17h32_35.mat')
 
 % detector menlo PD: frequenc BD: 150 MHz BW
 
 %% =============== screen out datas ===============  %%
 n_channel = 2 ; % active channel to plot
 
 
     h = 6.6e-34;
     lambda = 1064e-9;
     Ephoton = h*(3e8/lambda); % J
         
     Fs1 = SampleRate ;     % gage sampling frequncy
     Fs2 = Frep ;           % triggerbox sampling frequency
     
     % definition of 2 structure for data analyses:
     MyStat1 = stats_t( Fs1 );
     MyStat2 = stats_t( Fs2 );

     %alpha = 1/(0.45*1e5) ;        % convertion W/Volt - thorlabs
     %alpha = (200e-6)/0.3057 ;     % convertion W/Volt - Menlo
     
     
     % alpha = (18e-6)/0.3057 ;     % Thorlabs InGa DET10N/M + transimpedance 
     alpha = 1/(50*0.62) ;          % Thorlabs InGa DET20C2 termination 50Ohm W/Volt 
     alpha = 1e-6;
     % alpha = 1e6/(2*pi);          % Sylentsys conversion factor => MHz/Volt conversion
     %BW = 150e6 ;                  % Menlo photo-detector bandwith
     %BW = 70e6 ;                   % Thorlabs InGa DET10N/M 
     BW = 11.7e6 ;                  % Thorlabs InGa DET20C2 
     
     unit = 'W';    
     
    %% 1d-noise plots
    
    Hmu = figure(2); hold on ; %clf(Hmu,'reset');
    set(Hmu,'color','w')
    % set(Hmu,'units','normalized','outerposition',[0 0 1 1])
    %set(Hmu,'WindowStyle','docked'); 
    % raw/(0.45*1e5)  [V]unit x [W/V] = [W]unit - SI PD
    % raw/(0.45*1e5)  [V]unit x [W/V] = [W]unit - Menlo PD - saturation 0.7529 V

     
     % average and standard deviation
     Datas_mu1   = MyStat1.average(     alpha*raw(:,:,n_channel)  ); %Fs1 = SampleRate-> size t1
     Datas_std1  = MyStat1.standard_dev( alpha*raw(:,:,n_channel)  );%Fs1 = SampleRate-> size t1   
     Datas_mu2   = MyStat2.average(     alpha*raw(:,:,n_channel)'  );
     Datas_std2  = MyStat2.standard_dev( alpha*raw(:,:,n_channel)' );
     
     
%      [freq1 , PSDx1 , unit_psdx1  ] = MyStat1.PowSpecDens( alpha*raw  , unit )  ; % acquisition quick
%      [freq2 , PSDx2 , unit_psdx2  ] = MyStat2.PowSpecDens( alpha*raw' , unit ) ;  % acquisition long
      
     [freq1 , PSDx1 , unit_psdx1  ] = MyStat1.NoiseSpecDens( alpha*raw(:,:,n_channel)  , unit )  ; % acquisition quick
     [freq2 , PSDx2 , unit_psdx2  ] = MyStat2.NoiseSpecDens( alpha*raw(:,:,n_channel)' , unit ) ;  % acquisition long
  
     % power spectral density in [W]^2/Hz
     psdx1 = mean(PSDx1,2)'; %Fs1 = SampleRate-> size t1 /2
     psdx2 = mean(PSDx2,2)';
     
     % overall averaged power
     fprintf('\n Average power = %.2f mW \n',1e3*mean( Datas_mu1 ))
%%
     %D = ReadHighFinessFile('D:\datas\2021-10-29\HIghfiness_PSD_13mW.txt');

     % power spectral density variance:
     psdx1_std = sqrt(var(PSDx1,0,2))';
     psdx2_std = sqrt(var(PSDx2,0,2))';
     
%      figure 
%      hold on
%      loglog(freq1(2:end),1e6*psdx1(2:end));
%      xlabel('frequency(Hz)')
%      ylabel('psd(Hz/Hz')
  
     % power spectral density in dBm/sqrt(Hz)
     s1 = 10*log(sqrt( psdx1(2:end) )/(1e-3) ) ; % dBm/sqrt(Hz) unit
     s2 = 10*log(sqrt( psdx2(2:end) )/(1e-3) ) ; % dBm/sqrt(Hz) unit
        
     s1_std = 10*log( sqrt(psdx1_std(2:end))/(1e-3) ) ; % dBm/sqrt(Hz) unit
     s2_std = 10*log( sqrt(psdx2_std(2:end))/(1e-3) ) ; % dBm/Hz unit
     
     % shot noise level:
     psdn1_std = 2*(abs(mean(Datas_mu1))*Ephoton*BW)/(MyStat1.Fs) ; % in[W^2/Hz]
     psdn2_std = 2*(abs(mean(Datas_mu2))*Ephoton*BW)/(MyStat2.Fs) ; % in[W^2/Hz]
     
     n1_std = 10*log(sqrt( psdn1_std )/(1e-3) ) ; % dBm/sqrt(Hz) unit
     n2_std = 10*log(sqrt( psdn2_std )/(1e-3) ) ; % dBm/sqrt(Hz) unit
     
    t1 = (1/Fs1)*(1:size(raw,1)); % s
    t2 = (1/Fs2)*(1:size(raw,2)); % s
    
    subplot(221)
    title(sprintf('%d sampled points at %d Hz', length(t2) , Fs2 ))
    l = line(t2*1e3,1e6*Datas_mu2,'Color','k','marker','x'); hold on ;
    ylabel(strcat('\mu (\mu ',unit,')over short '))
    xlabel('time (ms)','fontsize',10)
    ax1 = gca; % current axes
    set(ax1,'Ylim',sort([ 0.9*min(min(1e6*Datas_mu2),min(1e6*Datas_mu1)) 1.2*max(max(1e6*Datas_mu2),max(1e6*Datas_mu1)) ]));
    set(ax1,'XColor','k');
    set(ax1,'YColor','k');
    set(ax1,'YGrid','on');
    set(ax1,'XAxisLocation','bottom','YAxisLocation','left');
    hold on
    patch = fill([t2*1e3,fliplr(t2*1e3)],[1e6*( Datas_mu2 + Datas_std2 ),fliplr(1e6*( Datas_mu2 - Datas_std2 ))],...
      get(l,'color'),'Parent',ax1,'FaceAlpha',0.2, 'EdgeColor','none');
%   
      subplot(222)
      title(sprintf('%d sampled points at %d MHz', length(t1) , Fs1*1e-6 ))
l = line(t1*1e6,1e6*Datas_mu1,'Color','r'); hold on
ax2 = gca; % current axes
patch = fill([t1*1e6,fliplr(t1*1e6)],[1e6*( Datas_mu1 + Datas_std1 ),fliplr(1e6*( Datas_mu1 - Datas_std1 ))],...
    get(l,'color'),'Parent',ax2,'FaceAlpha',0.2, 'EdgeColor','none');
    set(ax2,'Ylim',sort([ 0.9*min(min(1e6*Datas_mu2),min(1e6*Datas_mu1)) 1.2*max(max(1e6*Datas_mu2),max(1e6*Datas_mu1)) ]));
    set(ax2,'XColor','r');
    set(ax2,'YColor','r');
    set(ax2,'YGrid','on');
    ylabel('\mu (\mu W)over  long','fontsize',10)
    xlabel('time (\mu s)')
    
    
    subplot(223)
line(freq2(2:end),s2,'Color','k'); hold on ;
%yline(n2_std,'Color','g','LineStyle','--','linewidth',3);
title('Spectral Density relative to shot noise')
    xlabel('Frequency (Hz)')
    ylabel('PSD dBm/$$\sqrt{Hz}$$','Interpreter','latex');
    ax1 = gca; % current axes
    set(ax1,'XAxisLocation','bottom','YAxisLocation','left','Color','none');
    %set(ax1,'XScale','log');
    % patch = fill([freq2(2:end)*1e-6,fliplr(freq2(2:end)*1e-6)],[s2 + s2_std,fliplr(s2 - s2_std)],...
%      'k','Parent',ax1,'FaceAlpha',0.2, 'EdgeColor','none');
    ax1_pos = get(ax1,'Position'); % position of first axes

    subplot(224)
    % psdx unit: [raw^2/Hz] = [W^2/Hz] (cf Equation 3)
% test avec noise VA directement:

% plot(sqrt(1:10)); 
% h = legend(['$$\sqrt{blah}$$'])
% set(h,'Interpreter','latex','fontsize',24) 

%  figure;plot(freq1,10*log10(Fs1*psdx1)),
%  figure;plot(freq2,10*log10(Fs2*psdx2),'Color','r')
% e1  = MyStat1.Energy_t( raw/(0.45*1e5) );
% ee1 = MyStat1.Energy_psd( psdx1 ); 

line(freq1(2:end),s1,'Color','r'); hold on ;
%yline(n1_std,'Color','g','LineStyle','--','linewidth',3);
     ax2 = gca;
     %set(ax2,'Xlim',[min(freq1*1e-6) max(freq1*1e-6)]);
     title('Spectral Density relative to shot noise')
     set(ax2,'XColor','r');
     set(ax2,'YColor','r');
     %set(ax2,'XScale','log')
     xlabel('Frequency (Hz)')
     ylabel('PSD dBm/$$\sqrt{Hz}$$','Interpreter','latex');
    
%  patch = fill([freq1(2:end)*1e-6,fliplr(freq1(2:end)*1e-6)],[s1 + s1_std,fliplr(s1 - s1_std)],...
%      get(l,'color'),'Parent',ax2,'FaceAlpha',0.2, 'EdgeColor','none');
    %set(ax2,'Ylim',[ 0.9*min(min(s1),min(s2)) 1.2*max(max(s1),max(s2)) ]);
    
    
    set(findall(Hmu,'-property','FontSize'),'FontSize',11) 
    
%    figure(2); hold on ; plot( 1e6*Datas_mu1 , Datas_std1./sqrt(Ephoton*BW*abs(Datas_mu1)) , 'o') ;  
%    %figure; plot( 1e6*Datas_mu1 , Datas_std1./Datas_mu1  , 'o') ;
%     xlabel('Power PD (\mu W)')
%     ylabel('\sigma / \sigma_{sn} over sort ')

    
    %% 2-d raw datas
%     Hmu = figure(2); clf(Hmu,'reset');
%     set(Hmu,'color','w')
%     imagesc(1e3*t2,1e6*t1,1e3*raw);
%     set(gca,'XAxisLocation','top');
%     xlabel('t2( ms )')
%     ylabel('t1( \mu s )')
%     cb = colorbar;
%     ylabel(cb,'mVolt')
%     colormap(parula)
%     title(sprintf('Number of traces = %d',size(raw,2)))
    
    
    %%
    