classdef TF2D
    %TF2D Summarz of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        Nx       % number of point for fourier transform
        Nx0      % index of zero frequency along x direction
        xRange  
        x        % position in m
        Fsx
        fx       % frequencz varaiable
        kx       % frequencz variable in rad
        lambdax
        dx
        dfx
        
        Nz
        Nz0
        zRange  
        z        % position in m
        Fsz
        fz       % frequencz varaiable
        kz       % frequencz variable in rad
        lambdaz
        dz
        dfz
        
    end
    
    methods
        
        function obj = TF2D(varargin)
            
            % TF2D(x,z)
            if nargin == 2
            obj.x = varargin{1}; 
            obj.z = varargin{2}; 
            
            obj.Nx     = length(obj.x); 
            obj.dx = obj.x(2) - obj.x(1) ;
            obj.Fsx = 1/(obj.dx) ;         
            obj.xRange = (obj.Nx)*(obj.dx) ;
            obj.dfx = (obj.Fsx)/(obj.Nx) ; % because it should match (N/2)df = Fs/2 according to Nyquist in DFT
            obj.fx = (-floor((obj.Nx)/2):1:(ceil((obj.Nx)/2)-1))*obj.dfx;
            obj.Nx0 = floor( obj.Nx/2 ) + 1 ;
            obj.kx = 2*pi*obj.fx;
            obj.lambdax(1:floor((obj.Nx)/2)) = -(3e8)./obj.fx(1:floor((obj.Nx)/2));  % wavelength
            obj.lambdax(floor((obj.Nx)/2)+1) = 1e15;                                % non zero value at origin
            obj.lambdax(floor((obj.Nx)/2)+2:(obj.Nx))= (3e8)./obj.fx(floor((obj.Nx)/2)+2:(obj.Nx));
            
            obj.Nz     = length(obj.z); 
            obj.dz = obj.z(2) - obj.z(1) ;
            obj.Fsz = 1/(obj.dz) ;         
            obj.zRange = (obj.Nz)*(obj.dz) ;
            obj.dfz = (obj.Fsz)/(obj.Nz) ; % because it should match (N/2)df = Fs/2 according to Nyquist in DFT
            obj.fz = (-floor((obj.Nz)/2):1:(ceil((obj.Nz)/2)-1))*obj.dfz;
            obj.Nz0 = floor( obj.Nz/2 ) + 1 ;
            obj.kz = 2*pi*obj.fz;
            obj.lambdaz(1:floor((obj.Nz)/2)) = -(3e8)./obj.fz(1:floor((obj.Nz)/2));  % wavelength
            obj.lambdaz(floor((obj.Nz)/2)+1) = 1e15;                                % non zero value at origin
            obj.lambdaz(floor((obj.Nz)/2)+2:(obj.Nz))= (3e8)./obj.fz(floor((obj.Nz)/2)+2:(obj.Nz));
            
            elseif nargin == 4
            % TF2D(Nx,Nz,Fmax_x,Fmax_z)
            obj.Nx = varargin{1}; 
            obj.Nz = varargin{2}; 
            obj.Fsx = varargin{3}; 
            obj.Fsz = varargin{4}; 
            
            obj.dx = 1/(obj.Fsx);  % in m
            obj.x = ( (-floor(obj.Nx/2)):(ceil(obj.Nx/2)-1) )*obj.dx;
            obj.xRange = (obj.Nx)*obj.dx;
            obj.dfx = (obj.Fsx)/(obj.Nx) ; % because it should match (N/2)df = Fs/2 according to Nyquist in DFT
            obj.fx = (-floor(obj.Nx/2):1:(ceil(obj.Nx/2)-1))*obj.dfx;
            obj.Nx0 = floor( obj.Nx/2 ) + 1 ;
            obj.kx = 2*pi*obj.fx;
            obj.lambdax(1:floor((obj.Nx)/2)) = -(3e8)./obj.fx(1:floor((obj.Nx)/2));  % wavelength
            obj.lambdax(floor((obj.Nx)/2)+1) = 1e15;                                % non zero value at origin
            obj.lambdax(floor((obj.Nx)/2)+2:(obj.Nx))= (3e8)./obj.fx(floor((obj.Nx)/2)+2:(obj.Nx));
            

            obj.dz = 1/(obj.Fsz);  % in m
            obj.z = ( (-floor(obj.Nz/2)):(ceil(obj.Nz/2)-1) )*obj.dz;
            obj.zRange = (obj.Nz)*obj.dz;
            obj.dfz = (obj.Fsz)/(obj.Nz) ; % because it should match (N/2)df = Fs/2 according to Nyquist in DFT
            obj.fz = (-floor(obj.Nz/2):1:(ceil(obj.Nz/2)-1))*obj.dfz;
            obj.Nz0 = floor( obj.Nz/2 ) + 1 ;
            obj.kz = 2*pi*obj.fz;
            obj.lambdaz(1:floor((obj.Nz)/2)) = -(3e8)./obj.fz(1:floor((obj.Nz)/2));  % wavelength
            obj.lambdaz(floor((obj.Nz)/2)+1) = 1e15;                                % non zero value at origin
            obj.lambdaz(floor((obj.Nz)/2)+2:(obj.Nz))= (3e8)./obj.fz(floor((obj.Nz)/2)+2:(obj.Nz));
            
            end
            
        end
        
        function Ekxkz = fourier(obj, Exz)
            %fftshift(Et);    %real(F) sera toujours positif pour phi=0
            Ekxkz=fft2(ifftshift(Exz))*(obj.xRange/obj.Nx)*(obj.zRange/obj.Nz) ;
            %Ekxkz=fft2(Exz)*(obj.xRange/obj.Nx)*(obj.zRange/obj.Nz) ;
            Ekxkz=fftshift(Ekxkz);
        end
        
        function Exkz = fourierz(obj, Exz)
            %fftshift(Et);    %real(F) sera toujours positif pour phi=0
            Exkz = fft(ifftshift(Exz,1),obj.Nz,1)*(obj.zRange/obj.Nz) ;
            Exkz = fftshift(Exkz,1);
        end
         
        function Exz = ifourierz(obj, Exkz)
            %fftshift(Et);    %real(F) sera toujours positif pour phi=0
            Exz = ifft(fftshift(Exkz,1),obj.Nz,1)*(obj.Nz/obj.zRange) ;
            Exz = ifftshift(Exz,1);
        end
        
        function Ekxz = fourierx(obj, Exz)
            %fftshift(Et);    %real(F) sera toujours positif pour phi=0
            Ekxz = fft(ifftshift(Exz,2),obj.Nx,2)*(obj.xRange/obj.Nx) ;
            Ekxz = fftshift(Ekxz,2);
        end
        
        function Ekxz = ifourierx(obj, Exz)
            %fftshift(Et);    %real(F) sera toujours positif pour phi=0
            Ekxz = ifft(fftshift(Exz,2),obj.Nx,2)*(obj.Nx/obj.xRange)  ;
            Ekxz = ifftshift(Ekxz,2);
        end
        
        function Exz = ifourier(obj, Ekxkz)
            %fftshift(Et);    %real(F) sera toujours positif pour phi=0
            Exz=ifft2(ifftshift(Ekxkz))*(obj.Nz/obj.zRange)*(obj.Nx/obj.xRange)  ;
            Exz=fftshift(Exz);
        end
        
    end
    
end

