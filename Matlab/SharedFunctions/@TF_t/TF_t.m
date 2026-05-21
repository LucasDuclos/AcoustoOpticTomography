classdef TF_t
    %TFSpace: Creates an object with all the related variables in the time
    %frequency space
    
    properties
        N       % number of point for fourier transform
        N0
        T  
        t       % position in m
        Fs
        f       % frequency varaiable
        w       % frequency variable in rad
        l       % wavelength variable
        dt
        df
        dw
    end
    
    methods 
        % 2 METHODS are implemented to define the fourier paramater:
        % 1-> input N : number of point as powof 2 
        % Fmax : max sampling fequency 
        % 2-> intput : t : the time variable from raw data
        function obj = TF_t(varargin)
            
            if nargin == 1
                
            obj.t = varargin{1}; 
            obj.N = length(obj.t) ;
            obj.dt = obj.t(2) - obj.t(1) ;
            obj.Fs = 1/(obj.dt) ;
            obj.T = (obj.N)*(obj.dt) ;
            obj.df = (obj.Fs)/(obj.N) ; % because it should match (N/2)df = Fs/2 according to Nyquist in DFT
            obj.f = (-floor((obj.N)/2):1:(ceil((obj.N)/2)-1))*obj.df;
            obj.N0 = floor( obj.N/2 ) + 1 ;
            obj.w = 2*pi*obj.f;
            obj.l(1:floor((obj.N)/2))=-(3e8)./obj.f(1:floor((obj.N)/2)); % wavelength
            obj.l(floor((obj.N)/2)+1)=1e15;   % non zero value at origin
            obj.l(floor((obj.N)/2)+2:(obj.N))= (3e8)./obj.f(floor((obj.N)/2)+2:(obj.N)); 
            
            end
            
            if nargin == 2
            N = varargin{1};
            Fs = varargin{2};
            % Fmax corresponds to the max sampling of image
            
            obj.Fs = Fs ;   % sampling frequency imposed by user
            obj.N = N;      % number of points imposed by user
            obj.dt = 1/Fs;  % in m
            obj.t = ( (-floor(N/2)):(ceil(N/2)-1) )*obj.dt;
            obj.T = N*obj.dt;
            obj.df = Fs/N ; % because it should match (N/2)df = Fs/2 according to Nyquist in DFT
            obj.f = (-floor(N/2):1:(ceil(N/2)-1))*obj.df;
            obj.N0 = floor( obj.N/2 ) + 1 ;
            obj.w = 2*pi*obj.f;
            obj.l(1:floor(N/2))=-(3e8)./obj.f(1:floor(N/2)); % wavelength
            obj.l(floor(N/2)+1)=1e15;   % non zero value at origin
            obj.l(floor(N/2)+2:N)= (3e8)./obj.f(floor(N/2)+2:N); 
            
            end
            
            
        end
        function obj = Initialize(obj,N,Fmax)
            % Fmax corresponds to the mas sampling of image    
            % t is homogenous to a distance, not a time
            obj.N = N;
            obj.dt = 1/Fmax; % in m
            obj.t = (-N/2:1:N/2-1)*obj.dt;
            obj.T = N*obj.dt;
            obj.df = 1/obj.T;
            obj.f = (-N/2:1:N/2-1)*obj.df;
            obj.w = 2*pi*obj.f;
            obj.l(1:N/2)=  -(3e8)./obj.f(1:N/2); % wavelength
            obj.l(N/2+1)=   1e15;   % non zero value at origin
            obj.l(N/2+2:N)= (3e8)./obj.f(N/2+2:N);
            
        end
        function Ew  = fourier(obj, Et)
            %fftshift(Et);    %real(F) sera toujours positif pour phi=0
            % Ew=fft(ifftshift(Et),obj.N)*obj.tRange/obj.N;
            Ew=fft(ifftshift(Et),obj.N)*(1/obj.Fs);
            Ew=fftshift(Ew);
        end
        function Et  = ifourier(obj, Ew)
            %Et=ifft(ifftshift(Ew),obj.N)*obj.N/obj.tRange;
            Et=ifft(ifftshift(Ew),obj.N)*(obj.Fs) ;
            Et=fftshift(Et);
        end
        function e_t = Tintegral(obj,f)
        % using this function to check energy conservation. In that case f = |Fourier|^2    
            e_t = ( obj.dt )*sum(f) ;
            
        end
        function e_f = Fintegral(obj,f)
            
            e_f = ( obj.df )*sum(f) ; 
            
        end
        
        
        
    end
    
end

