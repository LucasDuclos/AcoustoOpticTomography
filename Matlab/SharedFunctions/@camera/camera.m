classdef camera
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % constant parameters
        FPS
        QE
        FWC % Full Well Capacity
        AD
        Idark
        ReadNoise
        bit
        Nx_cam 
        Ny_cam 
        dpixel
        x_cam
        y_cam
        
        % current frame
        wavelength
        IntegrationTime
        format
        frame
        frame_bg
    end
    
    methods
        function obj = camera(model)
            % UNTITLED5 Construct an instance of this class
            % Detailed explanation goes here
            switch model
                case 'PCO.edge'
            obj.FPS = 50;
            obj.AD = 4.3 ; 
            obj.Idark = 1 ;     % electron/px/s
            obj.ReadNoise = 1 ; % electrons
            obj.bit = 16 ;
            obj.Nx_cam = 2048;
            obj.Ny_cam = 2048;
            obj.dpixel = 5e-6;
            obj.QE = 0.5;
            
                case 'xiB-64'
            % sensor : LUX160
            % product ref : CB160MG-LX-X8G3-EF
            obj.FPS = 300;
            obj.AD = 9.77 ; 
            obj.FWC = 1e4;          % electrons
            obj.Idark = 1e4 ;       % electron/px/s
            obj.ReadNoise = 10 ;    % electrons
            obj.bit = 10 ;
            obj.Nx_cam = 4704;
            obj.Ny_cam = 3424;    
            obj.dpixel = 3.9e-6;   % pixels dimension in m
            obj.QE = 0.3;
            obj.x_cam = 1:(obj.Nx_cam);
            obj.y_cam = 1:(obj.Ny_cam);
            
                case 'DMK33GR0134'
            
            obj.FPS = 70;
            obj.AD = 0 ; 
            obj.FWC = 0;          % electrons
            obj.Idark = 0;        % electron/px/s
            obj.ReadNoise = 0;    % electrons
            obj.bit = 8 ;
            obj.Nx_cam = 1280;
            obj.Ny_cam = 960;    
            obj.dpixel = 3.75e-6;   % pixels dimension in m
            obj.QE = 0.3; %0.3 at 780nm / 1% at 1064nm
            obj.x_cam = 1:(obj.Nx_cam);
            obj.y_cam = 1:(obj.Ny_cam);
            
                case 'PhotronSA4'
                    
            obj.FPS = 1000;
            obj.AD = 0 ; 
            obj.FWC = 0;          % electrons
            obj.Idark = 0;        % electron/px/s
            obj.ReadNoise = 0;    % electrons
            obj.bit = 12 ;
            obj.Nx_cam = 1024;
            obj.Ny_cam = 1024;    
            obj.dpixel = 20e-6;   % pixels dimension in m
            obj.QE = 0.3; %0.3 at 780nm / 1% at 1064nm
            obj.x_cam = 1:(obj.Nx_cam);
            obj.y_cam = 1:(obj.Ny_cam);         
            
            end
            
        end
        
        function obj = ResizePixels(obj,Nx,Ny)
           obj.Nx_cam = Nx;
           obj.Ny_cam = Ny;  
           obj.x_cam = 1:(obj.Nx_cam);
           obj.y_cam = 1:(obj.Ny_cam);

        end
        
        function obj = ConvertFrame2electrons(obj)
            obj.frame = (obj.frame)*( obj.FWC /(2^(obj.format)-1) ) ;
        end
        
        function SIG = GetShotNoise(obj,mu)
            h = 6.6260e-34; % m^2 kg / s
            Ephoton = h*(3e8/obj.wavelength) ; % J
            alpha = (Ephoton)/(obj.QE*obj.IntegrationTime); %W
            SIG = sqrt(alpha)*sqrt(mu);
        end
        
        function obj = ConvertFrame2photons(obj)
            Electrons = (obj.frame)*( obj.FWC /2^(obj.format) ) ;
            obj.frame = Electrons/(obj.QE) ;
        end
        
        function obj = ConvertFrame2Intensity(obj)
            Electrons = (obj.frame)*( obj.FWC /(2^(obj.format)-1) ) ;
            Photons = Electrons/(obj.QE) ;
            h = 6.6260e-34; % m^2 kg / s
            Ephoton = h*(3e8/obj.wavelength) ; % J
            obj.frame = (Ephoton*Photons)/(obj.IntegrationTime*obj.dpixel*obj.dpixel) ; % W /m^2
        end

    end
end

