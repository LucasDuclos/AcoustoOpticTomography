classdef ImageFilter
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        Coord
    end

    methods
        function obj = ImageFilter(points)
            % [centerX,centerZ,WidthX,LengthZ]
            obj.Coord = points ;
        end

        function H = getROI(obj,Nx,Ny)

            H = ones(Ny,Nx);


            H( : ,1:Nx > obj.Coord(1) + obj.Coord(3)/2 )  = 0;
            H( :,1:Nx < obj.Coord(1) - obj.Coord(3)/2 )  = 0;
            H( 1:Ny > obj.Coord(2) + obj.Coord(4)/2 , : ) = 0;
            H( 1:Ny < obj.Coord(2) - obj.Coord(4)/2, :  ) = 0;

        end

        function H = getROI_scale(obj,Nx,Ny)

            Nx_TAB = 4704; 
            Ny_TAB = 3424;
            Cx_TAB = floor(Nx_TAB/2)+1;
            Cy_TAB = floor(Ny_TAB/2)+1;

            Cx = floor(Nx/2)+1;
            Cy = floor(Ny/2)+1;
            Coord = zeros(1,4);
        
            nux = (obj.Coord(1)-Cx_TAB)/Nx_TAB;
            nuy = (obj.Coord(2)-Cy_TAB)/Ny_TAB;
            Coord(1) = nux*Nx+Cx;
            Coord(2) = nuy*Ny+Cy;                
            nux = obj.Coord(3)/Nx_TAB;
            nuy = obj.Coord(4)/Ny_TAB;
            Coord(3) = nux*Nx;
            Coord(4) = nuy*Ny;                
               
            H = ones(Ny,Nx);

            H( : ,1:Nx > Coord(1) + Coord(3)/2 )  = 1;
            H( :,1:Nx < Coord(1) - Coord(3)/2 )  = 0;
            H( 1:Ny > Coord(2) + Coord(4)/2 , : ) = 0;
            H( 1:Ny < Coord(2) - Coord(4)/2, :  ) = 0;


        end

        function [] = DrawROI(obj)
            BOX = [obj.Coord(1) - obj.Coord(3)/2,...
                obj.Coord(1) + obj.Coord(3)/2,...
                obj.Coord(2) - obj.Coord(4)/2,...
                obj.Coord(2) + obj.Coord(4)/2 ];
            rectangle('Position',[BOX(1)-0.5,BOX(3)-0.5,BOX(2)-BOX(1)+0.5,BOX(4)- BOX(3)+0.5])
        end
    end
end

