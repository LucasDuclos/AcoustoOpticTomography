classdef OF < TF2D
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Data
        x_scan
        t
    end
    
    properties (Access = private)
        Lx              % [min max] : dimension of input image in z direction
        Lz              % [min max] : dimension of input image in z direction
        SamplingRate         % Samplinf frequency in Hz
        c                    % sound velocity in m/s 
    end
    
    methods
        function obj = OF(x_scan,t,Data,SamplingRate,c)
            %N: number of points for fourier transform
            N = 2^10 ;
            Fmax_x = 1/( x_scan(2) - x_scan(1) ) ;
            Fmax_z = 1/( c*t(2)- c*t(1) ) ; % convert time to distance through time of flight
            obj@TF2D(N,N,Fmax_x,Fmax_z);    % build TF2D structure for fourier transform (not used here)
            
            
            obj.x_scan = x_scan ;
            obj.t = t ;
            obj.Data = Data;
            obj.SamplingRate = SamplingRate ;
            obj.c = c ;
            
            % interpolation of raw data into fourier grid :
%             [X_scan,CT] = meshgrid(x_scan,c*t);
%             [X,Z] = meshgrid(obj.x,obj.z);
%             
%             obj.Data = interp2(X_scan,CT,Data,X,Z,'linear',0);
            
        end
        

    end
end

