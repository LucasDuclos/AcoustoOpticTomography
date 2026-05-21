function ActiveLIST = generateActiveList(Decimations,NbElemts,Nscan)

ActiveLIST = true(NbElemts,Nscan); 

for i = 1:length(Decimations)
    
    i_decimate = Decimations(i);
    i_cos      = 4*i_decimate - 3 ;
    i_ncos     = 4*i_decimate - 2 ;
    i_sin      = 4*i_decimate - 1 ;
    i_nsin     = 4*i_decimate - 0 ;
    
    I = (1:length(AlphaM)) + length(AlphaM) ;
    
    Icos  = I + ( i_cos-1 )*length(AlphaM) ;  % index of column with same decimate
    Incos = I + (i_ncos-1 )*length(AlphaM) ;  % index of column with same decimate
    Isin  = I + (i_sin-1  )*length(AlphaM) ;  % index of column with same decimate
    Insin = I + (i_nsin-1 )*length(AlphaM) ;  % index of column with same decimate
    
    ActiveLIST( : , Icos  )   = CalcMat_OS( Xelements - XMiddle , dFx*Nbx(i_decimate) , ActiveLIST(:,Icos) , 'cos' ) ;
    ActiveLIST( : , Incos )   = ~ActiveLIST( : , Icos );
    ActiveLIST( : , Isin  )   = CalcMat_OS( Xelements - XMiddle , dFx*Nbx(i_decimate) , ActiveLIST(:,Isin) , 'sin' ) ;
    ActiveLIST( : , Insin )   = ~ActiveLIST( : , Isin );
    
end

ActiveLIST(setdiff(Ielements,ElmtBorns(1):ElmtBorns(2)),:) = false ;

end