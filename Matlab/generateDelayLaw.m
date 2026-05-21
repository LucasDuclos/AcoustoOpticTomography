function Delay = generateDelayLaw(Angles,NbElemts,Nscan)
Delay = zeros(NbElemts,Nscan); %(s)

for i = 1:length(Angles)
    Delay(:,i ) = 1000*(1/c)*sin(Angles(i))*(1:NbElemts)'*(pitch); %s
    Delay(:,i ) = Delay(:,i ) - min(Delay(:,i));
    
end

Delay = repmat( Delay(:,1:length(Angles)), 1 , Nscan/length(Angles) ) ;

end