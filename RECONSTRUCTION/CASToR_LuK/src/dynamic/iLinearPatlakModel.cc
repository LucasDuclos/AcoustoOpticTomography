
/*!
  \file
  \ingroup  dynamic
  \brief    Implementation of class iLinearPatlakModel
  \todo     Checks regarding optimization on NestedEM loops when updating parametric images
*/

#include "iLinearPatlakModel.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn iLinearPatlakModel
  \brief Constructor of iLinearPatlakModel. Simply set all data members to default values.
*/
iLinearPatlakModel::iLinearPatlakModel() : iLinearModel() 
{
  m_nbTimeBF       = 2;
  m_nbModelParam   = 2;
  m_PatlakMethodFlag = PATLAK_METHOD_LS; // Least-square by default
  m_nnlsN = 2;
  m_nbRgateBF = 1; m_nbRGModelParam=1;
  m_nbCgateBF = 1; m_nbCGModelParam=1;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn ~iLinearPatlakModel
  \brief Destructor of iLinearPatlakModel
*/
iLinearPatlakModel::~iLinearPatlakModel() 
{
  if(m_initialized)
  {
    if(m_PatlakMethodFlag == PATLAK_METHOD_NNLS)
    {
      if(m3p_nnlsA && m2p_nnlsB && m2p_nnlsMat)
      {      
        for( int th=0 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++ )
        {
          //for( int n=0 ; n<m_nnlsN ; n++ )
          //  if(m3p_nnlsA[ th ][ n ]) delete[] m3p_nnlsA[ th ][ n ];
  
          if(m3p_nnlsA[ th ] )   delete[] m3p_nnlsA[ th ];
          if(m2p_nnlsB[ th ] )   delete[] m2p_nnlsB[ th ];
          if(m2p_nnlsMat[ th ] ) delete[] m2p_nnlsMat[ th ];
        }
  
        delete[] m3p_nnlsA;
        delete[] m2p_nnlsB;
        delete[] m2p_nnlsMat;
      }
      
      if( mp_w ) delete[] mp_w;
    }
  }
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn ShowHelp
  \brief Print out specific help about the implementation of the Patlak
         model and its initialization
*/
void iLinearPatlakModel::ShowHelp()
{
  cout << "-- This class implements the Patlak Reference Tissue Model : " << endl;
  cout << "-- Patlak CS, Blasberg RG: Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data" << endl;
  cout << "-- J Cereb Blood Flow Metab 1985, 5(4):5 84-590." << endl;
  cout << "-- DOI http://dx.doi.org/10.1038/jcbfm.1985.87" << endl;
  cout << "-- It is used to model radiotracers which follows as 2-tissue compartment model with irreversible trapping  " << endl;
  cout << "-- The Patlak temporal basis functions are composed of the Patlak slope (integral of the reference TAC from the injection time " << endl;
  cout << "   divided by the instantaneous reference activity), and intercept (reference tissue TAC)  " << endl;
  cout << endl;
  cout << " It can be initialized using either an ASCII file or a list of option with the following keywords and information :" << endl; 
  cout << "   As this class inherits from the iLinearModel class, the following parameters must be declared inside a couple of the following specific tags: " << endl;
  cout << "   - DYNAMIC FRAMING/ENDDF " << endl;
  cout << "   - The ASCII file must contain the following keywords :" << endl;
  cout << "   'Basis_functions:'       (mandatory) Enter the coefficients of Patlak plot and intercept for each time frame (tf) ";
  cout << "                                        on two successive lines, separated by ',' :" << endl;
  cout << "                            -> Patlak_functions: " << endl;
  cout << "                            -> coeff_Pplot_tf1,coeff_Pplot_tf2,...,coeff_Pplot_tfn" << endl;
  cout << "                            -> coeff_Pintc_tf1,coeff_Pintc_tf2,...,coeff_Pintc_tfn" << endl;
  cout << "   'Parametric_image_init: path ' (optional) path to an interfile image to be used as initialization for the parametric images." << endl;
  cout << "   'Patlak method        : x    ' (optional) optimization method: " << endl;
  cout << "                                    x=0: Least-Square (default) " << endl;
  cout << "                                    x=1: Iterative non-negative Least-Square " << endl;
  cout << "                                         (C.L. Lawson and R.J. Hanson, Solving Least Squares Problems)" << endl;
  cout << "                                    x=2: Nested EM " << endl;
  cout << endl;
  cout << " - The list of options must contain the coefficients of both Patlak functions separated by commas, with the following template :" << endl;
  cout << "   coeff_Pplot_tf1,coeff_Pplot_tf2,...,coeff_Pplot_tfn,";
  cout << "   coeff_Pintc_tf1,coeff_Pintc_tf2,...,coeff_Pintc_tfn "<< endl;
  cout << "   Parametric images will be initialized with 1.0 and 1000. for Patlak slope and intercept by default " << endl;
  cout << "   The parametric images estimations will be written on disk for each iteration" << endl;
  cout << "    " << endl;
  cout << "   The following keywords are common to all dynamic models :" << endl;
  cout << "   'Number of iterations before image update: x' Set a number 'x' of iteration to reach before using the model to generate the images at each frames/gates" << endl;
  cout << "   (Default x ==  0) " << endl;
  cout << "   'No image update: x'                          If set to 1, the reconstructed images for the next iteration/subset are not reestimated using the model" << endl;
  cout << "   (Default x ==  0)                              (the code just performs standard independent reconstruction of each frames/gates) " << endl;
  cout << "   'No parameters update: x'                     If set to 1, the parameters / functions of the model are not estimated with the image" << endl;
  cout << "   (Default x ==  0)                              (this could be used to test The EstimateImageWithModel() function with specific user-provided parametric images) " << endl;
  cout << "   'Save parametric images : x'                  Enable (1)/Disable(0) saving parametric images on disk" << endl;
  cout << "   (Default x == 1)  " << endl;
  cout << "   " << endl;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn ReadAndCheckConfigurationFileSpecific
  \param const string& a_configurationFile : ASCII file containing informations about a dynamic model
  \brief This function is used to read options from a configuration file.
  \return 0 if success, other value otherwise.
*/
int iLinearPatlakModel::ReadAndCheckConfigurationFileSpecific()
{
  if(m_verbose >=3) Cout("iLinearPatlakModel::ReadAndCheckConfigurationFileSpecific ..."<< endl); 
  
  // The file will be fully processed in the Initialize() function
  ifstream in_file(m_fileOptions.c_str(), ios::in);
  
  if( in_file)
  {
    // Number of iteration for NESTED EM method, in case it is selected
    if( ReadDataASCIIFile(m_fileOptions, "Number_model_iterations", &m_nbLinearModelCycles, 1, KEYWORD_OPTIONAL) == 1)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read 'Number_model_iterations' flag in " << m_fileOptions << endl);
      return 1;
    }
  }
  else
  {
    Cerr("***** iLinearPatlakModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read configuration file at: " << m_fileOptions << endl);
    return 1;
  }
  
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn ReadAndCheckOptionsList
  \brief This function is used to read parameters from a string.
  \return 0 if success, other value otherwise.
*/
int iLinearPatlakModel::ReadAndCheckOptionsList(string a_listOptions)
{
  if(m_verbose >=3) Cout("iLinearPatlakModel::ReadAndCheckOptionsList ..."<< endl); 
  
  // Just recover the string here, it will be processed in the Initialize() function
  m_listOptions = a_listOptions;
  
  // Normal end
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn CheckSpecificParameters
  \brief This function is used to check whether all member variables
         have been correctly initialized or not.
  \return 0 if success, positive value otherwise.
*/
int iLinearPatlakModel::CheckSpecificParameters()
{
  if(m_verbose >=3) Cout("iLinearPatlakModel::CheckSpecificParameters ..."<< endl); 
  
  
  // Check at least one basis function number has been initialized
  if (m_nbTimeBF<=0)
  {
    Cerr("***** iLinearPatlakModel::CheckParameters() -> Error, the variables corresponding to the number of basis function has not been initialized. There might be an error in the configuration process/file !" << endl);
    return 1;
  }
  else
  {
    // Set the other variables to 1 if not initialized
    if(m_nbTimeBF <0) m_nbTimeBF =1;
    if(m_nbRgateBF<0) m_nbRgateBF=1;
    if(m_nbCgateBF<0) m_nbCgateBF=1;
  }
  
  // Check if we have somehow both a file and a list of options for init...
  if(m_listOptions != "" && m_fileOptions != "")
  {
    Cerr("***** iLinearPatlakModel::Initialize -> Either a file or a list of options have to be selected to initialize the model, but not both ! " << endl);
    return 1;
  }

  // Normal end
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn InitializeSpecific
  \brief This function is used to initialize Patlak parametric images and basis functions
  \return 0 if success, other value otherwise.
*/
int iLinearPatlakModel::InitializeSpecific()
{
  if(m_verbose >=2) Cout("iLinearPatlakModel::InitializeSpecific ..."<< endl); 

  // Forbid initialization without check
  if (!m_checked)
  {
    Cerr("***** oDynamicModelManager::InitializeSpecific() -> Must call CheckParameters functions before Initialize() !" << endl);
    return 1;
  }
  
  // --- Memory Allocation --- //
  
  // Allocate memory for Parametric images and functions of Patlak model
  m2p_parametricImages = new FLTNB*[m_nbTimeBF];
  m2p_outputParImages  = new FLTNB*[m_nbTimeBF];
  m2p_modelTACs = new HPFLTNB*[m_nbTimeBF];
  
  for(int b=0 ; b<m_nbTimeBF ; b++)
  {
    m2p_modelTACs[b] = new HPFLTNB[mp_ID->GetNbTimeFrames()];
    m2p_parametricImages[b] = new FLTNB[mp_ID->GetNbVoxXYZ()];
  }

  m2p_RGModelTACs = new HPFLTNB*[m_nbRgateBF];
  m2p_RGParametricImages = new FLTNB*[m_nbRgateBF];

  for(int rb=0 ; rb<m_nbRgateBF ; rb++)
  {
    m2p_RGModelTACs[rb] = new HPFLTNB[mp_ID->GetNbRespGates()];
    m2p_RGParametricImages[rb] = new FLTNB[mp_ID->GetNbVoxXYZ()];
  }

  m2p_CGModelTACs = new HPFLTNB*[m_nbCgateBF];
  m2p_CGParametricImages = new FLTNB*[m_nbCgateBF];

  
  for(int cb=0 ; cb<m_nbCgateBF ; cb++)
  {
    m2p_CGModelTACs[cb] = new HPFLTNB[mp_ID->GetNbCardGates()];
    m2p_CGParametricImages[cb] = new FLTNB[mp_ID->GetNbVoxXYZ()];
  }
  
  // --- Default Initialization basis functions --- //
  for(int b=0 ; b<m_nbTimeBF ; b++)
    for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
      m2p_modelTACs[b][fr] = 1.;
      
  for(int rb=0 ; rb<m_nbRgateBF ; rb++)
    for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
      m2p_RGModelTACs[rb][rg] = 1.;
      
  for(int cb=0 ; cb<m_nbCgateBF ; cb++)
    for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
      m2p_CGModelTACs[cb][cg] = 1.;

            
  // Memory for correction images
  mp_corrBasisCoeffs = new FLTNB[mp_ID->GetNbVoxXYZ()];
  mp_corrBasisFunctions = new HPFLTNB[mp_ID->GetNbThreadsForProjection()];
  
  // --- Data Initialization with a configuration file --- //

  if(m_fileOptions != "")
  {
    ifstream in_file(m_fileOptions.c_str(), ios::in);
    
    if(in_file)
    {
      // Frame basis functions Initialization
      if(ReadDataASCIIFile(m_fileOptions,
                            "Basis_functions",
                            m2p_modelTACs,
                            mp_ID->GetNbTimeFrames(),
                            m_nbTimeBF,
                            KEYWORD_MANDATORY) )
        {
          Cerr("***** iLinearPatlakModel::Initialize -> Error while trying to read frame basis functions coefficients !" << endl);
          return 1;
        }
      
        
      
      // --- Parametric image initialization --- //
      
      // Frame model
      string input_image = "";
      int return_value = 0;
      
      return_value = ReadDataASCIIFile(m_fileOptions,
                                        "Parametric_images_init",
                                        &input_image,
                                        1,
                                        KEYWORD_OPTIONAL);
      
      if( return_value == 0) // Image have been provided
      {
        // Read image // INTF_LERP_DISABLED = interpolation disabled for input image reading
        if( IntfReadImgDynCoeffFile(input_image,
                                    m2p_parametricImages,
                                    mp_ID,
                                    m_nbModelParam,
                                    m_verbose,
                                    INTF_LERP_DISABLED) ) // Image have been provided
        {
          Cerr("***** iLinearPatlakModel::Initialize -> Error while trying to read the provided initialization parametric images : " << input_image << endl);
          return 1;
        }
      }
      else if( return_value == 1) // Error during reading
      {
        Cerr("***** iLinearPatlakModel::Initialize -> Error while trying to read dynamic frame model parametric images !" << endl);
        Cerr("                                  'Parametric_image_init' keyword in " << m_fileOptions << endl);
        return 1;
      }
      else //(return_value >= 1 ) // Keyword not found : no initialization provided
      {
        // Standard initialization
        for(int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
        {
          m2p_parametricImages[0][v] = 1.;
          m2p_parametricImages[1][v] = 1000.;
        }
      }
      
      // Patlak method
      if( ReadDataASCIIFile(m_fileOptions,
                            "Patlak_method",
                            &m_PatlakMethodFlag,
                            1,
                            KEYWORD_OPTIONAL) == KEYWORD_OPTIONAL_ERROR )
      {
        Cerr("***** iPatlakModel::Initialize -> Error while trying to read 'POSEM_mode' keyword in " << m_fileOptions << endl);
        return 1;
      }

    }

    else
    {
      Cerr("***** iLinearPatlakModel::Initialize() -> Error while trying to read configuration file at: " << m_fileOptions << endl);
      return 1;
    }
  }
  
  
  // --- Data Initialization with a list of options --- //

  if(m_listOptions != "")
  {
    // We expect here the coefficients of the Patlak functions for each time point
    // Patlak slope before Patlak intercept
  
    // Allocate memory to recover the elements in one tmp vector
    FLTNB *p_coeffs = new FLTNB[m_nbTimeBF * mp_ID->GetNbTimeFrames()];
    
    // Read them
    if (ReadStringOption(m_listOptions,
                         p_coeffs,
                         m_nbTimeBF * mp_ID->GetNbTimeFrames(),
                         ",",
                         "Patlak model configuration"))
    {
      Cerr("***** iPatlakModel::Initialize() -> Failed to correctly read the list of parameters in command-line options  !" << endl);
      return 1;
    }
    
    // Affect coeffs
    for(int c=0 ; c<m_nbTimeBF * mp_ID->GetNbTimeFrames() ; c++)
    {
      int bf = int(c/mp_ID->GetNbTimeFrames()); // Patlak basis function index
      int fr = int(c%mp_ID->GetNbTimeFrames()); // Frame index
      m2p_modelTACs[bf][fr] = p_coeffs[c];
    }
    
    // Delete the tmp vector
    delete[] p_coeffs;
    
    
    // Standard initialization for the parametric images
    for(int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
    {
      m2p_parametricImages[0][v] = 1.;
      m2p_parametricImages[1][v] = 1000.;
    }
  }


  // Allocate output image matrices
  for(int b=0 ; b<m_nbTimeBF ; b++)
    m2p_outputParImages[b] = new FLTNB[mp_ID->GetNbVoxXYZ()];

  // Initialize gate parametric images of linear model mother class 
  for(int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
  {
    m2p_RGParametricImages[ 0 ][ v ] = 1.;
    m2p_CGParametricImages[ 0 ][ v ] = 1.;
  }

  // NNLS variables
  if(m_PatlakMethodFlag == PATLAK_METHOD_NNLS)
  {
    m3p_nnlsA    = new HPFLTNB** [mp_ID->GetNbThreadsForImageComputation()];
    m2p_nnlsB    = new HPFLTNB*  [mp_ID->GetNbThreadsForImageComputation()];
    m2p_nnlsMat  = new HPFLTNB*  [mp_ID->GetNbThreadsForImageComputation()];
    
    for( int th=0 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++ )
    {
      // Init 2D coefficient matrix for NNLS estimation
      m3p_nnlsA[ th ] = new HPFLTNB*[ m_nnlsN ];
  
      for(int n=0 ; n<m_nnlsN ; n++)
        m3p_nnlsA[ th ][n] = new HPFLTNB[ mp_ID->GetNbTimeFrames() ];
    
      // Init solution vector and working matrix for NNLS estimation
      m2p_nnlsB[ th ]   = new HPFLTNB[ mp_ID->GetNbTimeFrames() ];
      m2p_nnlsMat[ th ] = new HPFLTNB[ (m_nnlsN+2) * mp_ID->GetNbTimeFrames() ];
    }
    
    // Compute weights for NNLS
    mp_w = new HPFLTNB[ mp_ID->GetNbTimeFrames() ];
  
    for(int t=0; t<mp_ID->GetNbTimeFrames(); t++) 
      mp_w[t] = sqrt(mp_ID->GetFrameDurationInSec(0,t)) ;  // (convert time in min);
  }
  
  
  
  
  // Display method and initial TACs
  if(m_verbose >=2)
  {
    if(m_PatlakMethodFlag == PATLAK_METHOD_LS)
      Cout("iLinearPatlakModel::InitializeSpecific() -> Selected optimization method : Least-square (0)" << endl);
    else if(m_PatlakMethodFlag == PATLAK_METHOD_NNLS)
      Cout("iLinearPatlakModel::InitializeSpecific() -> Selected optimization method : NNLS (1)" << endl);
    else if(m_PatlakMethodFlag == PATLAK_METHOD_NESTED_EM)
      Cout("iLinearPatlakModel::InitializeSpecific() -> Selected optimization method : Nested-EM (2)" << endl);
    
    
    if(m_nbTimeBF>1)
    {
      Cout("iLinearPatlakModel::InitializeSpecific() -> Frame dynamic model TAC coefficients :" << endl);
      for(int b=0 ; b<m_nbTimeBF ; b++)
      {
        Cout("                              ");
        Cout("Basis function["<<b+1<<"]");
        for(int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
          Cout(m2p_modelTACs[b][fr] << ", ");
        Cout(endl);
      }
    }
  }
  
  // Normal end
  m_initialized = true;
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn EstimateModelParameters
  \param ap_ImageS : pointer to the ImageSpace
  \param a_ite : index of the actual iteration (not used)
  \param a_sset : index of the actual subset (not used)
  \brief Estimate Patlak parametric images
  \return 0 if success, other value otherwise.
*/
int iLinearPatlakModel::EstimateModelParameters(oImageSpace* ap_ImageS, int a_ite, int a_sset) 
{
  #ifdef CASTOR_DEBUG
  if (!m_initialized)
  {
    Cerr("***** iLinearPatlakModel::EstimateModelParameters() -> Called while not initialized !" << endl);
    Exit(EXIT_DEBUG);
  }
  #endif

  // Use the nested NNLS method
  if(m_PatlakMethodFlag == PATLAK_METHOD_NNLS)
  {
    if(Patlak_NNLS(ap_ImageS, a_ite) )
      {
        Cerr("***** iLinearPatlakModel::EstimateModelParameters() -> An error occured while using the NNLS method !" << endl);
        return 1;
      }
  }
  
  // Use the nested EM parametric image estimation method
  else if(m_PatlakMethodFlag == PATLAK_METHOD_NESTED_EM)
  {
    if(NestedEM(ap_ImageS, a_ite) )
    {
      Cerr("***** iLinearPatlakModel::EstimateModelParameters() -> An error occured while using the nested EM parametric image estimation method !" << endl);
      return 1;
    }
  }
  
  // Least square method (default)
  else if(m_PatlakMethodFlag == PATLAK_METHOD_LS) 
  {
    if(Patlak_LS(ap_ImageS, a_ite) )
      {
        Cerr("***** iLinearPatlakModel::EstimateModelParameters() -> An error occured while using the least-square method !" << endl);
        return 1;
      }
  }
  else
  {
    Cerr("***** iLinearPatlakModel::EstimateModelParameters() -> Error : unknown method to estimate Patlak images ! !" << endl);
    return 1;
  }

  
  if(m_verbose >=3)
  {
      for (int fb=0 ; fb<m_nbTimeBF ; fb++) 
      { 
        HPFLTNB avg = 0;
        
        for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
          avg += m2p_parametricImages[fb][v];
        
        avg /= mp_ID->GetNbVoxXYZ();
        Cout( "iLinearPatlakModel::EstimateModelParameters() -> Frame parametric image["<<fb<<"] avg value:" << avg << endl);
      }
  }
  
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn Patlak_NNLS
  \param ap_ImageS : pointer to the ImageSpace
  \param a_ite : index of the actual iteration (not used)
  \brief Estimate Patlak parametric images using the NNLS method
  \return 0 if success, other value otherwise.
*/
int iLinearPatlakModel::Patlak_NNLS(oImageSpace* ap_ImageS, int a_ite) 
{
  if(m_verbose >=3) Cout("iLinearPatlakModel::Patlak_NNLS ..." <<endl);
  
  int v;
  #pragma omp parallel for private(v) schedule(guided)
  for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
  {
    
        int th=0;
    #ifdef CASTOR_OMP
    th = omp_get_thread_num();
    #endif
    
    HPFLTNB** pp_nnls_A = m3p_nnlsA[ th ];
    HPFLTNB* p_nnls_B = m2p_nnlsB[ th ];    
    HPFLTNB p_nnls_X[2];
    
    HPFLTNB nnls_rnorm;
    HPFLTNB p_nnls_wp[2];
    HPFLTNB *p_nnls_zz=NULL;
    int     p_nnls_index[3];
    
    for(int t=0; t<mp_ID->GetNbTimeFrames(); t++)
    {
      // Fill NNLS A matrix:
      // function #1: tissue integral x -1
      pp_nnls_A[0][t]=m2p_modelTACs[0][t];
      // function #2: integral of input
      pp_nnls_A[1][t]=m2p_modelTACs[1][t];
        
      // Fill NNLS B array: tissue
      p_nnls_B[t]=ap_ImageS->m4p_image[t][0][0][v];
    }


    if(NNLS(pp_nnls_A,
            mp_ID->GetNbTimeFrames(),
            2,
            p_nnls_B,
            p_nnls_X,
            &nnls_rnorm,
            p_nnls_wp,
            p_nnls_zz,
            p_nnls_index) )
      continue;
      
    m2p_parametricImages[0][v] = p_nnls_X[0];
    m2p_parametricImages[1][v] = p_nnls_X[1];


    delete[] p_nnls_B;
    
    for(int i=0 ; i<2 ; i++)
      delete[] pp_nnls_A[i];
      
    delete[] pp_nnls_A;
  }
    
  return 0;
}
  
  
  
  
// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn Patlak_LS
  \param ap_ImageS : pointer to the ImageSpace
  \param a_ite : index of the actual iteration (not used)
  \brief Estimate Patlak parametric images using the voxelwise least square method
  \return 0 if success, other value otherwise.
*/
int iLinearPatlakModel::Patlak_LS(oImageSpace* ap_ImageS, int a_ite) 
{
  if(m_verbose >=3) Cout("iLinearPatlakModel::Patlak_LS ..." <<endl);
  
  int v;
  #pragma omp parallel for private(v) schedule(guided)
  for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
  { 
    HPFLTNB xmean = 0.,
            ymean = 0.,
            K = 0.,
            Vd = 0.,
            Cov= 0.,
            Var = 0.;
  
  
    // Compute means for this voxel
    for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++) 
    {
      xmean += m2p_modelTACs[0][fr]/m2p_modelTACs[1][fr];
      ymean += ap_ImageS->m4p_image[fr][0][0][v]/m2p_modelTACs[1][fr];
    }

    xmean /= mp_ID->GetNbTimeFrames();
    ymean /= mp_ID->GetNbTimeFrames();
    
    HPFLTNB Yt,Xt ;

    // Compute covariance & variance
    for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
    {
      Yt = ap_ImageS->m4p_image[fr][0][0][v]/m2p_modelTACs[1][fr];
      Xt = m2p_modelTACs[0][fr]/m2p_modelTACs[1][fr];
      Cov += (Xt-xmean) * (Yt-ymean);
      Var += (Xt-xmean) * (Xt-xmean);
    }

    // Slope
    K = (Var != 0) ? Cov/Var : 0.;
    
    // Non-negativity constraint
    if(K<0) K = 0.;
    
    // Intercept
    Vd = ymean - K*xmean;

    // Non-negativity constraint
    if(Vd<0) Vd = 0.;

    m2p_parametricImages[0][v] = K;
    m2p_parametricImages[1][v] = Vd;
  }
  
  return 0;
}
