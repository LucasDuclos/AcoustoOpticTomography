classdef FourierMap
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        ScanParam   % coefficient scanned in Fourier Domain
        nuX0
        nuZ0
        In          % Inpout Image Line Vector
        If          % Inpout Image Spectrum

            
    end
    
    methods
        function obj = FourierMap(ScanParam,nuX0,nuZ0)
            % UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.ScanParam = ScanParam ;
            obj.nuX0 = nuX0 ;
            obj.nuZ0 = nuZ0 ;
            NbXNbz = unique(ScanParam(:,1:2),'row');
            obj.Nx = Nx;
            obj.Nz = Nz;
            obj.dx = 1/Fmax_x; % in m
            obj.dz = 1/Fmax_z; % in m
            obj.x = (-Nx/2:1:Nx/2-1)*obj.dx;
            obj.z = (-Nz/2:1:Nz/2-1)*obj.dz;
            obj.xRange = (Nx-1)*obj.dx;
            obj.zRange = (Nz-1)*obj.dz;
            obj.dfx = 1/obj.xRange;
            obj.dfz = 1/obj.zRange;
            obj.fx = 0:Nx*nuX0;
            obj.fz = 0:Nz*nuZ0;

        end
        
        function obj = DefineImage(obj,In)
           obj.In = In(:) ;  
        end
        
        function obj = Fourier(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            obj.If = fft(In(:));
            
        end
    end
end

