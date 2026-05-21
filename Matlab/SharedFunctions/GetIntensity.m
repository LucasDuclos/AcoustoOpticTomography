function [frame,P_tot] = GetIntensity( IM , BG , MyXimea )


%% define of camera buffer used for analysis:
MyXimea.frame = IM;%( 1200 + (1:2^10) , 1700 + (1:2^10) );
MyXimea.frame_bg = BG;%( 1200 + (1:2^10) , 1700 + (1:2^10) );


% remove BG
MyXimea.frame = MyXimea.frame - MyXimea.frame_bg ;
MyXimea.frame(MyXimea.frame<0) = 1e-22;
% MyXimea = MyXimea.ConvertFrame2electrons;
 %MyXimea = MyXimea.ConvertFrame2photons;
 MyXimea = MyXimea.ConvertFrame2Intensity; % in W/m^2

% evaluate total incident power
P_tot = sum( MyXimea.frame(:)*MyXimea.dpixel*MyXimea.dpixel );
frame = MyXimea.frame;

end

