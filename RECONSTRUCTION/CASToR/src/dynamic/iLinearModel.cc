
/*!
  \file
  \ingroup  dynamic
  \brief    Implementation of class iLinearModel
  \todo     Checks regarding optimization on NestedEM loops when updating parametric images
*/

#include "iLinearModel.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn iLinearModel
  \brief Constructor of iLinearModel. Simply set all data members to default values.
*/
iLinearModel::iLinearModel() : vDynamicModel() 
{
  m_nbRgateBF      = -1;
  m_nbCgateBF      = -1;
  m_nbRGModelParam = -1;
  m_nbCGModelParam = -1;
  
  m2p_RGParametricImages = NULL;
  m2p_CGParametricImages = NULL;
  m2p_RGModelTACs        = NULL;
  m2p_CGModelTACs        = NULL;
  
  m_fileOptions = "";
  m_listOptions = "";

  mp_corrBasisCoeffs    = NULL;
  mp_corrBasisFunctions = NULL;
  m_nbLinearModelCycles       = 1;
  m_basisFunctionsUpdStartIte =-1;
  m_basisFunctionsUpdRatio    = 1;
  m_basisFunctionsUpdIdx      = 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn ~iLinearModel
  \brief Destructor of iLinearModel
*/
iLinearModel::~iLinearModel() 
{
  if(m_initialized)
  {
    for(int rb=0 ; rb<m_nbRgateBF ; rb++)
    {
      if (m2p_RGModelTACs[rb]) delete[] m2p_RGModelTACs[rb];
      if (m2p_RGParametricImages[rb]) delete[] m2p_RGParametricImages[rb];
    }
  
    for(int cb=0 ; cb<m_nbCgateBF ; cb++)
    {
      if (m2p_CGModelTACs[cb]) delete[] m2p_CGModelTACs[cb];
      if (m2p_CGParametricImages[cb]) delete[] m2p_CGParametricImages[cb];
    }

    if (mp_corrBasisCoeffs != NULL) delete[] mp_corrBasisCoeffs; 
    if (mp_corrBasisFunctions != NULL) delete[]  mp_corrBasisFunctions;
  }
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn ShowHelp
  \brief Print out specific help about the implementation of this model
         and its initialization
*/
void iLinearModel::ShowHelp()
{
  cout << "-- This class implements a general linear dynamic model applied between the images of a dynamic acquisition." << endl;
  cout << "-- The model is applied on a voxel-by-voxel basis between the images of the frames and/or respiratory/cardiac gates. " << endl;
  cout << "-- The main keywords 'DYNAMIC FRAMING', 'RESPIRATORY GATING' and 'CARDIAC GATING' ahead of the parameters allow to define at which level the model parameters must be applied. " << endl;
  cout << "-- Main parameters to define are:" << endl;
  cout << "   -> The number of basis functions / parametric images defined in the model" << endl;
  cout << "   -> Basis function initial values" << endl;
  cout << "   -> Parametric images initialization (optional)" << endl;
  cout << "   The optimization method is the Nested EM algorithm" << endl;
  cout << endl;
  cout << " It can be initialized using a configuration text file with the following keywords and information :" << endl; 
  cout << " - Mandatory keywords :" << endl;
  cout << "   The following keywords are mandatory for at least one dynamic image level (Dynamic frame, Respiratory gating or Cardiac gating) :" << endl;
  cout << "   'Number basis functions:'   Enter the number of basis function for each image of time frame or respiratory/cardiac gate ";
  cout << "   'Basis_functions:'          Enter the basis function (bf) coefficients for each image (im) of time frame or respiratory/cardiac gate ";
  cout << "                                 on successive lines, separated by ',' :" << endl;
  cout << "                                -> basis_functions: " << endl;
  cout << "                                -> coeff_bf1_im1,coeff_bf1_im2,...,coeff_bf1_imn" << endl;
  cout << "                                -> coeff_bf2_im1,coeff_bf2_im2,...,coeff_bf2_imn" << endl;
  cout << "                                -> etc..." << endl;
  cout << " - Optional keywords :" << endl;
  cout << "   'Parametric_image_init: image_file' Set an image file to initialize the parametric images of each dynamic model. Default initialization: '1.0 for each voxel." << endl;
  cout << "   " << endl;
  cout << "   The previous parameters must be declared inside a couple of the following specific tags: " << endl;
  cout << "   - DYNAMIC FRAMING/ENDDF for chronological frame model  (dynamic model applied to chronological frames of a dynamic aquisition)" << endl;
  cout << "   - RESPIRATORY GATING/ENDRG for respiratory model       (dynamic model applied to respiratory gates of a dynamic aquisition)" << endl;
  cout << "   - CARDIAC GATING/ENDCG for cardiac model               (dynamic model applied to cardiac gates of a dynamic aquisition)" << endl;
  cout << "   Different levels of dynamic model can be enabled simultaneously (i.e a dynamic frame model can be used simultaneously with respiratory and/or cardiac gating model) " << endl;
  cout << "   " << endl;
  cout << "   " << endl;
  cout << "   The following keywords are optional and common to each dynamic model (Dynamic frame, Respiratory gating or Cardiac gating) :" << endl;
  cout << "   'Number_model_iterations: x'        Number of iterations of the model parameters and basis functions updates in one cycle of Nested EM" << endl;
  cout << "   (Default x ==  1)                      (one cycle consists in x iterations in which either the parametric images or the basis functions are updated) " << endl;
  cout << "                                         The ratio of parametric images / basis functions updates depends on the following parameter:" << endl;
  cout << "   'Basis_function_update_ratio: x'    Ratio of update between parametric images and basis functions updates cycle " << endl;
  cout << "   (Default x ==  0)                      Cycles consist in x iterations of the parametric images, following by x iterations of the basis functions" << endl;
  cout << "                                         If x == 0, only the parametric images are updated" << endl;
  cout << "   'Basis_function_start_ite : x'      Starting iteration for the update of basis functions " << endl;
  cout << "   (Default x == -1)                     If negative, no update of the basis functions is performed (only parametric images are updated) " << endl;
  cout << "   " << endl;
  cout << "   " << endl;
  cout << "   ---------------------------" << endl;
  cout << "   Example of initialization: " << endl;
  cout << "   DYNAMIC FRAMING" << endl;
  cout << "   Number basis functions         : 2" << endl;
  cout << "   Basis_functions            :" << endl;
  cout << "   23682.79, 25228.74, 26636.99, 27923.61, 29101.16" << endl;
  cout << "   5.4, 4.91, 4.48, 4.1, 3.75" << endl;
  cout << "   ENDDF" << endl;
  cout << "   " << endl;
  cout << "   RESPIRATORY GATING" << endl;
  cout << "   Number basis functions         : 6" << endl;
  cout << "   Basis_functions            :" << endl;  
  cout << "   1, 0.8, 0.6, 0.4, 0.2, 0.01" << endl;
  cout << "   0.7, 0.9, 0.7, 0.5, 0.3, 0.1" << endl;
  cout << "   0.4, 0.6, 0.8, 0.6, 0.4, 0.2" << endl;
  cout << "   0.2, 0.4, 0.6, 0.8, 0.6, 0.4" << endl;
  cout << "   0.1, 0.3, 0.5, 0.7, 0.9, 0.7" << endl;
  cout << "   0.01, 0.2, 0.4, 0.6, 0.8, 1" << endl;
  cout << "   ENDRG" << endl;
  cout << "   " << endl;
  cout << "   Number_model_iterations : 1" << endl;
  cout << "   Basis_function_start_ite : -1" << endl;
  cout << "   Basis_function_update_ratio : 0" << endl;
  cout << "   ---------------------------" << endl;
  cout << "   " << endl;
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
int iLinearModel::ReadAndCheckConfigurationFileSpecific()
{
  if(m_verbose >=3) Cout("iLinearModel::ReadAndCheckConfigurationFileSpecific ..."<< endl); 
  
  // The file will be fully processed in the Initialize() function
  ifstream in_file(m_fileOptions.c_str(), ios::in);
  
  if(in_file)
  {
    // Check first the file contains the mandatory keyword(s)
    bool file_is_good = false;
    
    string line="", kword_to_search="";
    while(!in_file.eof())
    {
      getline(in_file, line);
  
      //remove comment
      if (line.find("#") != string::npos) line = line.substr(0, line.find_first_of("#")) ;
      
      if (line.find("DYNAMIC FRAMING") != string::npos)
      {
        if(kword_to_search != "")
        {
          Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error, found an end tag 'END**' before 'DYNAMIC FRAMING' in configuration file: " << m_fileOptions << endl);
          return 1;
        }
        else
          kword_to_search = "ENDDF";
      }
      
      else if (line.find("RESPIRATORY GATING") != string::npos)
      {
        if(kword_to_search != "")
        {
          Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error, found an end tag 'END**' before 'RESPIRATORY GATING' in configuration file: " << m_fileOptions << endl);
          return 1;
        }
        else
          kword_to_search = "ENDRG";
      }
      
      else if (line.find("CARDIAC GATING") != string::npos)
      {
        if(kword_to_search != "")
        {
          Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error, found an end tag 'END**' before 'CARDIAC GATING' in configuration file: " << m_fileOptions << endl);
          return 1;
        }
        else
          kword_to_search = "ENDCG";
      }
        
      else if (kword_to_search != "" 
       && line.find(kword_to_search) != string::npos)
      {
        kword_to_search = "";
        file_is_good = true;
      }
    }
    
    if(!file_is_good)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error, no mandatory tags ('DYNAMIC FRAMING'/'ENDDF', 'RESPIRATORY GATING'/'ENDRG', 'CARDIAC GATING'/'ENDCG' detected in configuration file: " 
            << m_fileOptions << ". Check help for more info" << endl);
      return 1;
    }
    
    if( ReadDataASCIIFile(m_fileOptions, "Number basis functions", &m_nbTimeBF, 1, KEYWORD_MANDATORY, "DYNAMIC FRAMING", "ENDDF") == 1)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read 'Number basis functions' flag in " << m_fileOptions << endl);
      return 1;
    }
    
    if( ReadDataASCIIFile(m_fileOptions, "Number basis functions", &m_nbRgateBF, 1, KEYWORD_OPTIONAL, "RESPIRATORY GATING", "ENDRG") == 1)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read 'Number basis functions' flag in " << m_fileOptions << endl);
      return 1;
    }
    
    if( ReadDataASCIIFile(m_fileOptions, "Number basis functions", &m_nbCgateBF, 1, KEYWORD_OPTIONAL, "CARDIAC GATING", "ENDCG") == 1)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read 'Number basis functions' flag in " << m_fileOptions << endl);
      return 1;
    }

    if( ReadDataASCIIFile(m_fileOptions, "Number_model_iterations", &m_nbLinearModelCycles, 1, KEYWORD_OPTIONAL) == 1)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read 'Number_model_iterations' flag in " << m_fileOptions << endl);
      return 1;
    }
    
    if( ReadDataASCIIFile(m_fileOptions, "Basis_function_start_ite", &m_basisFunctionsUpdStartIte, 1, KEYWORD_OPTIONAL) == 1)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read 'Basis_function_start_ite' flag in " << m_fileOptions << endl);
      return 1;
    }
    
    if( ReadDataASCIIFile(m_fileOptions, "Basis_function_update_ratio", &m_basisFunctionsUpdRatio, 1, KEYWORD_OPTIONAL) == 1)
    {
      Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read 'Basis_function_update_ratio' flag in " << m_fileOptions << endl);
      return 1;
    }
  }
  else
  {
    Cerr("***** iLinearModel::ReadAndCheckConfigurationFileSpecific -> Error while trying to read configuration file at: " << m_fileOptions << endl);
    return 1;
  }


  // Initialize the number of parameters of the linear model as the number of (frame) basis functions
  // (Variable used during parametric images writing on disk)
  m_nbModelParam = m_nbTimeBF;
  m_nbRGModelParam = m_nbRgateBF;
  m_nbCGModelParam = m_nbCgateBF;
  
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
int iLinearModel::ReadAndCheckOptionsList(string a_listOptions)
{
  if(m_verbose >=3) Cout("iLinearModel::ReadAndCheckOptionsList ..."<< endl); 
  
  // Just recover the string here, it will be processed in the Initialize() function
  m_listOptions = a_listOptions;
  
  // TODO
  
  // For now, just restricts the initialization using a configuration file as there are quite a lot of parameters to initialize for a use with command line options
  Cerr("***** iLinearModel::ReadAndCheckOptionsList() -> Initialization with command line options is not implemented for this class. Please use a configuration file instead" << endl);
  return 1;
  
  // Normal end
  //return 0;
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
int iLinearModel::CheckSpecificParameters()
{
  if(m_verbose >=2) Cout("iLinearModel::CheckSpecificParameters ..."<< endl); 
  
  
  // Check at least one basis function number has been initialized
  if (m_nbTimeBF<=0 && m_nbRgateBF<=0 && m_nbCgateBF<=0)
  {
    Cerr("***** iLinearModel::CheckParameters() -> Error, the variables corresponding to the number of basis function has not been initialized. There might be an error in the configuration process/file !" << endl);
    return 1;
  }
  else
  {
    // Set the other variables to 1 if not initialized
    if(m_nbTimeBF<0)  m_nbTimeBF =1;
    if(m_nbRgateBF<0) m_nbRgateBF=1;
    if(m_nbCgateBF<0) m_nbCgateBF=1;
  }
  
  // Check if we have somehow both a file and a list of options for init...
  if(m_listOptions != "" && m_fileOptions != "")
  {
    Cerr("***** iLinearModel::Initialize -> Either a file or a list of options have to be selected to initialize the model, but not both ! " << endl);
    return 1;
  }
  
  // Check if we have no file not list of options for some reason...
  if(m_listOptions == "" && m_fileOptions == "")
  {
    Cerr("***** iLinearModel::Initialize -> Either a file or a list of options should have been provided at this point ! " << endl);
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
  \brief This function is used to initialize the parametric images and basis functions
  \return 0 if success, other value otherwise.
*/
int iLinearModel::InitializeSpecific()
{
  if(m_verbose >=2) Cout("iLinearModel::InitializeSpecific ..."<< endl); 

  // Forbid initialization without check
  if (!m_checked)
  {
    Cerr("***** oDynamicModelManager::InitializeSpecific() -> Must call CheckParameters functions before Initialize() !" << endl);
    return 1;
  }
  
  // --- Memory Allocation --- //
  
  // Allocate memory for Parametric images and functions
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
      if(m_nbTimeBF>1
      && ReadDataASCIIFile(m_fileOptions,
                            "Basis_functions",
                            m2p_modelTACs,
                            mp_ID->GetNbTimeFrames(),
                            m_nbTimeBF,
                            KEYWORD_MANDATORY,
                            "DYNAMIC FRAMING",
                            "ENDDF") )
        {
          Cerr("***** iLinearModel::Initialize -> Error while trying to read frame basis functions coefficients !" << endl);
          Cerr("                                  'Basis_functions' keyword inside DYNAMIC FRAMING / ENDDF paragraph in " << m_fileOptions << endl);
          return 1;
        }
      
      // Resp gates basis functions Initialization
      if(m_nbRgateBF>1
      && ReadDataASCIIFile(m_fileOptions,
                            "Basis_functions",
                            m2p_RGModelTACs,
                            mp_ID->GetNbRespGates(),
                            m_nbRgateBF,
                            KEYWORD_MANDATORY,
                            "RESPIRATORY GATING",
                            "ENDRG") )
        {
          Cerr("***** iLinearModel::Initialize -> Error while trying to read respiratory gates basis functions coefficients !" << endl);
          Cerr("                                  'Basis_functions' keyword inside RESPIRATORY GATING / ENDRG paragraph in " << m_fileOptions << endl);
          return 1;
        }
        
      // Card gates basis functions Initialization
      if(m_nbCgateBF>1
      && ReadDataASCIIFile(m_fileOptions,
                            "Basis_functions",
                            m2p_CGModelTACs,
                            mp_ID->GetNbCardGates(),
                            m_nbCgateBF,
                            KEYWORD_MANDATORY,
                            "CARDIAC GATING",
                            "ENDCG") )
        {
          Cerr("***** iLinearModel::Initialize -> Error while trying to read cardiac gates basis functions coefficients !" << endl);
          Cerr("                                  'Basis_functions' keyword inside CARDIAC GATING / ENDCG paragraph in " << m_fileOptions << endl);
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
                                        KEYWORD_OPTIONAL,
                                        "DYNAMIC FRAMING",
                                        "ENDDF");
      
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
          Cerr("***** iLinearModel::Initialize -> Error while trying to read the provided initialization parametric images : " << input_image << endl);
          return 1;
        }
      }
      else if( return_value == 1) // Error during reading
      {
        Cerr("***** iLinearModel::Initialize -> Error while trying to read dynamic frame model parametric images !" << endl);
        Cerr("                                  'Parametric_image_init' keyword in " << m_fileOptions << endl);
        return 1;
      }
      else //(return_value >= 1 ) // Keyword not found : no initialization provided
      {
        // Standard initialization
        for(int b=0 ; b<m_nbTimeBF ; b++)
          for(int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
            m2p_parametricImages[b][v] = 1.;
      }
      
      
      // Resp gate model
      input_image = "";
      return_value = 0;
      
      return_value = ReadDataASCIIFile(m_fileOptions,
                                        "Parametric_images_init",
                                        &input_image,
                                        1,
                                        KEYWORD_OPTIONAL,
                                        "RESPIRATORY GATING",
                                        "ENDRG");
      
      if( return_value == 0) // Image have been provided
      {
        // Read image // INTF_LERP_DISABLED = interpolation disabled for input image reading
        if( IntfReadImgDynCoeffFile(input_image,
                                    m2p_RGParametricImages,
                                    mp_ID,
                                    m_nbRGModelParam,
                                    m_verbose,
                                    INTF_LERP_DISABLED) ) // Image have been provided
        {
          Cerr("***** iLinearModel::Initialize -> Error while trying to read the provided initialization parametric images : " << input_image << endl);
          return 1;
        }
      }
      else if( return_value == 1) // Error during reading
      {
        Cerr("***** iLinearModel::Initialize -> Error while trying to read respiratory gate model parametric images !" << endl);
        Cerr("                                  'Parametric_image_init' keyword in " << m_fileOptions << endl);
        return 1;
      }
      else //(return_value >= 1 ) // Keyword not found : no initialization provided
      {
        // Standard initialization
        for(int rb=0 ; rb<m_nbRgateBF ; rb++)
          for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
            m2p_RGParametricImages[rb][v] = 1.;
      }
      
      
      
      // Card gate model
      input_image = "";
      return_value = 0;
      
      return_value = ReadDataASCIIFile(m_fileOptions,
                                        "Parametric_images_init",
                                        &input_image,
                                        1,
                                        KEYWORD_OPTIONAL,
                                        "CARDIAC GATING",
                                        "ENDCG");
      
      if( return_value == 0) // Image have been provided
      {
        // Read image // INTF_LERP_DISABLED = interpolation disabled for input image reading
        if( IntfReadImgDynCoeffFile(input_image,
                                    m2p_CGParametricImages,
                                    mp_ID,
                                    m_nbCGModelParam,
                                    m_verbose,
                                    INTF_LERP_DISABLED) ) // Image have been provided
        {
          Cerr("***** iLinearModel::Initialize -> Error while trying to read the provided initialization parametric images : " << input_image << endl);
          return 1;
        }
      }
      else if( return_value == 1) // Error during reading
      {
        Cerr("***** iLinearModel::Initialize -> Error while trying to read cardiac gate model parametric images !" << endl);
        Cerr("                                  'Parametric_image_init' keyword in " << m_fileOptions << endl);
        return 1;
      }
      else //(return_value >= 1 ) // Keyword not found : no initialization provided
      {
        // Standard initialization
        for(int cb=0 ; cb<m_nbCgateBF ; cb++)
          for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
            m2p_CGParametricImages[cb][v] = 1.;
      }
    }

    else
    {
      Cerr("***** iLinearModel::Initialize() -> Error while trying to read configuration file at: " << m_fileOptions << endl);
      return 1;
    }
  }
  
  
  // --- Data Initialization with a list of options --- //

  if(m_listOptions != "")
  {
    // TODO
  }

  // Allocate output image matrices
  for(int b=0 ; b<m_nbTimeBF ; b++)
    m2p_outputParImages[b] = new FLTNB[mp_ID->GetNbVoxXYZ()];


  // Display initial TACs
  if(m_verbose >=2)
  {
    if(m_nbTimeBF>1)
    {
      Cout("iLinearModel::Initialize() -> Frame dynamic model TAC coefficients :" << endl);
      for(int b=0 ; b<m_nbTimeBF ; b++)
      {
        Cout("                              ");
        Cout("Basis function["<<b+1<<"]");
        for(int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
          Cout(m2p_modelTACs[b][fr] << ", ");
        Cout(endl);
      }
    }
    
    if(m_nbRgateBF>1)
    {
      Cout("iLinearModel::Initialize() -> Respiratory gate model TAC coefficients :" << endl);
      for(int rb=0 ; rb<m_nbRgateBF ; rb++)
      {
        Cout("                              ");
        Cout("Basis function["<<rb+1<<"]");
        for(int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
          Cout(m2p_modelTACs[rb][rg] << ", ");
        Cout(endl);
      }
    }
    
    if(m_nbCgateBF>1)
    {
      Cout("iLinearModel::Initialize() -> Cardiac gate model TAC coefficients :" << endl);
      for(int cb=0 ; cb<m_nbCgateBF ; cb++)
      {
        Cout("                              ");
        Cout("Basis function["<<cb+1<<"]");
        for(int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
          Cout(m2p_modelTACs[cb][cg] << ", ");
        Cout(endl);
      }
    }
  }
  
   // TODO : print output parametric images rg et cg
  
  
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
  \brief Estimate model parameters (parametric images and basis functions)
  \return 0 if success, other value otherwise.
*/
int iLinearModel::EstimateModelParameters(oImageSpace* ap_ImageS, int a_ite, int a_sset) 
{
  #ifdef CASTOR_DEBUG
  if (!m_initialized)
  {
    Cerr("***** iLinearModel::EstimateModelParameters() -> Called while not initialized !" << endl);
    Exit(EXIT_DEBUG);
  }
  #endif

  // Use the nested EM parametric image estimation method
  if(NestedEM(ap_ImageS, a_ite) )
  {
    Cerr("***** iLinearModel::EstimateModelParameters() -> An error occured while using the nested EM parametric image estimation method !" << endl);
    return 1;
  }
  
  return 0;
}
  
  
  

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn NestedEM
  \param ap_ImageS : pointer to the ImageSpace
  \param a_ite : index of the actual iteration (not used)
  \brief Estimate parametric images and basis functions (if enabled) using the nested EM method
  \return 0 if success, other value otherwise.
*/
int iLinearModel::NestedEM(oImageSpace* ap_ImageS, int a_ite) 
{
  if(m_verbose >=3) Cout("iLinearModel::NestedEM() ..." <<endl);
  
  for (uint32_t it=0 ; it<m_nbLinearModelCycles ; it++)
  {
    if(m_verbose >=3) Cout("iLinearModel::NestedEM() cycle "<< it+1 << "/" << m_nbLinearModelCycles <<endl);
    
    // Step 1 : Generate a difference image from voxel-by-voxel division of the current estimation of the image and the model image generated from the current parametric images / basis functions of the model
    // The backward image matrix (which is useless at this point of the reconstruction) is used as provisional image to gather the model image
    
    int v;
    #pragma omp parallel for private(v) schedule(static, 1)
    for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
      for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
        for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
          for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
          {
            // Reset this voxel to 0
            ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v] = 0;
  
            for (int fb=0 ; fb<m_nbTimeBF ; fb++)
              for (int rb=0 ; rb<m_nbRgateBF ; rb++)
                for (int cb=0 ; cb<m_nbCgateBF ; cb++)
                {
                  // Retrieve current estimation of image according to coeffs/basis functions
                  ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v] += m2p_parametricImages[fb][v] 
                                                                     * m2p_modelTACs[fb][fr] 
                                                                     * m2p_RGParametricImages[rb][v] 
                                                                     * m2p_RGModelTACs[rb][rg] 
                                                                     * m2p_CGParametricImages[cb][v] 
                                                                     * m2p_CGModelTACs[cb][cg] ;
                }
              
            // Recover correction images using the ratio of the current estimation of the image (m4p_image) and the image generated with coffs/basis functions
            // (Again, use backward image as provisional image to recover the result)
            ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v] = ap_ImageS->m4p_image[fr][rg][cg][v] / ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v];
          }

    
    // Step 2 : Estimate either basis functions or parametric images
    
    // Step 2a : Basis functions estimation
    if( m_basisFunctionsUpdStartIte >= 0      // if (m_basisFunctionsUpdStartIte < 0) --> no update of basis functions, just update the parametric images
    &&  m_basisFunctionsUpdStartIte <= a_ite // if (m_basisFunctionsUpdStartIte >= 0) --> Check condition of minimal iteration before starting to update the basis functions 
    && !(int((m_basisFunctionsUpdIdx-1)/m_basisFunctionsUpdRatio)&1) ) // Check if we must estimate basis functions or coefficients at this stage (depends on m_basisFunctionsUpdRatio)
    {

      HPFLTNB parametric_image_norm = 0;
    
      if( m_nbTimeBF>1 )
      {
        if(m_verbose >=3) Cout("iLinearModel::NestedEM() -> Estimate Basis Functions - Frame basis functions estimation step" <<endl);
          
        for (int fb=0 ; fb<m_nbTimeBF ; fb++) 
        { 
          // Compute normalization related to the vowelwise time basis functions coefficients.
          parametric_image_norm = 0;
    
          // Compute normalization related to the parametric images.
          for (int rb=0 ; rb<m_nbRgateBF ; rb++)
            for (int cb=0 ; cb<m_nbRgateBF ; cb++)
                for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
                  parametric_image_norm += m2p_parametricImages[fb][v]
                                         * m2p_RGParametricImages[rb][v]
                                         * m2p_CGParametricImages[cb][v];
                  
          parametric_image_norm *= mp_ID->GetNbRespGates()
                                 * mp_ID->GetNbCardGates();
                                 
          for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
          {
            // Initialization of corrections factor for time basis functions 
            for (int th=0 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++)
              mp_corrBasisFunctions[th]=0;
    
            for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
              for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                for (int rb=0 ; rb<m_nbRgateBF ; rb++)
                  for (int cb=0 ; cb<m_nbCgateBF ; cb++)
                  {
                    int v;
                    #pragma omp parallel for private(v) schedule(static, 1)
                    for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
                    {
                      int th = 0;
                      #ifdef CASTOR_OMP 
                      th = omp_get_thread_num();
                      #endif
                      
                      mp_corrBasisFunctions[th] += ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v] 
                                                 * m2p_parametricImages[fb][v] 
                                                 * m2p_RGParametricImages[rb][v]
                                                 * m2p_CGParametricImages[cb][v];
                    }
                  }
    
            // Reduce
            for (int th=1 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++)
              mp_corrBasisFunctions[0] += mp_corrBasisFunctions[th];
              
            // Apply corrections and normalization to the temporal basis functions values
            if (mp_corrBasisFunctions[0] > 0.)
              m2p_modelTACs[fb][fr] *= mp_corrBasisFunctions[0]/parametric_image_norm;
          }
          
        }
      }
      
      if( m_nbRgateBF>1 )
      {
        if(m_verbose >=3) Cout("iLinearModel::NestedEM() -> Estimate Basis Functions - Respiratory gate basis functions estimation step" <<endl); 
          
        for (int rb=0 ; rb<m_nbRgateBF ; rb++) 
        {
          // Compute normalization related to the vowelwise time basis functions coefficients.
          parametric_image_norm = 0;
          
          for (int fb=0 ; fb<m_nbTimeBF ; fb++)
            for (int cb=0 ; cb<m_nbRgateBF ; cb++)
                for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
                  parametric_image_norm += m2p_parametricImages[fb][v]
                                         * m2p_RGParametricImages[rb][v]
                                         * m2p_CGParametricImages[cb][v];
                    
          parametric_image_norm *= mp_ID->GetNbTimeFrames()
                                 * mp_ID->GetNbCardGates();
                                 
          for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
          {
            // Initialization of  corrections for time basis functions 
            for (int th=0 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++)
              mp_corrBasisFunctions[th]=0;
              
            for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
              for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                for (int fb=0 ; fb<m_nbTimeBF ; fb++)
                  for (int cb=0 ; cb<m_nbCgateBF ; cb++)
                  {
                    int v;
                    #pragma omp parallel for private(v) schedule(static, 1)
                    for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
                    {
                      int th = 0;
                      #ifdef CASTOR_OMP
                      th = omp_get_thread_num();
                      #endif
                      mp_corrBasisFunctions[th] += ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v]
                                                 * m2p_parametricImages[fb][v] 
                                                 * m2p_RGParametricImages[rb][v]
                                                 * m2p_CGParametricImages[cb][v];
                    }
                  }
            
            // Reduce
            for (int th=1 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++)
              mp_corrBasisFunctions[0] += mp_corrBasisFunctions[th];
              
            // Apply corrections and normalization to the temporal basis functions values 
            if (mp_corrBasisFunctions[0] > 0.)
              m2p_RGModelTACs[rb][rg] *= mp_corrBasisFunctions[0]/parametric_image_norm; 
            
          }
        }
      }
          
          
      if( m_nbCgateBF>1 )
      {
        if(m_verbose >=3) Cout("iLinearModel::NestedEM() -> Estimate Basis Functions - Cardiac gate basis functions estimation step" <<endl);
          
        for (int cb=0 ; cb<m_nbCgateBF ; cb++) 
        {
          // Compute normalization related to the vowelwise time basis functions coefficients.
          parametric_image_norm = 0;
          
          for (int fb=0 ; fb<m_nbTimeBF ; fb++)
            for (int rb=0 ; rb<m_nbRgateBF ; rb++) 
                for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
                  parametric_image_norm += m2p_parametricImages[fb][v]
                                         * m2p_RGParametricImages[rb][v]
                                         * m2p_CGParametricImages[cb][v];
                                         
          parametric_image_norm *= mp_ID->GetNbTimeFrames()
                                 * mp_ID->GetNbRespGates();
                    
          for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
          {
            // Initialization of  corrections for time basis functions 
            for (int th=0 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++)
              mp_corrBasisFunctions[th]=0;
              
            for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
              for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
                for (int fb=0 ; fb<m_nbTimeBF ; fb++)
                  for (int rb=0 ; rb<m_nbRgateBF ; rb++) 
                  {
                    int v;
                    #pragma omp parallel for private(v) schedule(static, 1)
                    for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
                    {
                      int th = 0;
                      #ifdef CASTOR_OMP
                      th = omp_get_thread_num();
                      #endif
                      mp_corrBasisFunctions[th] += ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v]
                                                 * m2p_parametricImages[fb][v] 
                                                 * m2p_RGParametricImages[rb][v]
                                                 * m2p_CGParametricImages[cb][v];
                    }
                  }
            
            // Reduce
            for (int th=1 ; th<mp_ID->GetNbThreadsForImageComputation() ; th++)
              mp_corrBasisFunctions[0] += mp_corrBasisFunctions[th];
              
            // Apply corrections and normalization to the temporal basis functions values 
            if (mp_corrBasisFunctions[0] > 0.)
              m2p_CGModelTACs[cb][cg] *= mp_corrBasisFunctions[0]/parametric_image_norm; 
            
          }
        }
      }
      
      m_basisFunctionsUpdIdx++;
  
      // Some feedback :
      if(m_verbose >=3)
      {
        for (int fb=0 ; fb<m_nbTimeBF ; fb++) 
          for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
            Cout( "iLinearModel::NestedEM() -> Basis function ["<<fb<<"] coefficients for frame ["<<fr<<"] :" << m2p_modelTACs[fb][fr] << endl);
            
        for (int rb=0 ; rb<m_nbRgateBF ; rb++)
          for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
            Cout( "iLinearModel::NestedEM() -> Basis function ["<<rb<<"] coefficients for resp gate ["<<rg<<"] :" << m2p_RGModelTACs[rb][rg] << endl);
      
        for (int cb=0 ; cb<m_nbCgateBF ; cb++)
          for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
            Cout( "iLinearModel::NestedEM() -> Basis function ["<<cb<<"] coefficients for card gate ["<<cg<<"] :" << m2p_CGModelTACs[cb][cg] << endl);
      }
    } // end of if loop (basis functions update)



    // Step 2b : Basis functions estimation
    else 
    {
      // Normalization general factor
      HPFLTNB basis_functions_norm = 0.;
  
      // Regularisation according to frame basis functions
      if( m_nbModelParam>1 )
      {
        if(m_verbose >=3) Cout("iLinearModel::NestedEM() -> Estimate Coefficients - Frame coeffs estimation step" <<endl);
          
        for (int fb=0 ; fb<m_nbTimeBF ; fb++)
        {
          // Initialization of voxelwise corrections coefficients for time basis functions coefficients
          for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
            mp_corrBasisCoeffs[v] = 0;
          
          basis_functions_norm = 0.;
          
          
          // Compute normalization related to the temporal basis functions.
          for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
            for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
              for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                for (int rb=0 ; rb<m_nbRgateBF ; rb++)
                  for (int cb=0 ; cb<m_nbCgateBF ; cb++)
                    basis_functions_norm += m2p_modelTACs[fb][fr]
                                          * m2p_RGModelTACs[rb][rg]
                                          * m2p_CGModelTACs[cb][cg];
        


          int v;
          #pragma omp parallel for private(v) schedule(static, 1)
          for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
          {
            for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
              for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
                for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                  for (int rb=0 ; rb<m_nbRgateBF ; rb++)
                    for (int cb=0 ; cb<m_nbCgateBF ; cb++)
                    {
                      mp_corrBasisCoeffs[v] += ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v] 
                                             * m2p_modelTACs[fb][fr] 
                                             * m2p_RGModelTACs[rb][rg]
                                             * m2p_CGModelTACs[cb][cg];
                    }
                    
            // Apply corrections and normalization to the frame model parametric image.
            if (mp_corrBasisCoeffs[v] > 0.)
              m2p_parametricImages[fb][v] *= mp_corrBasisCoeffs[v]/basis_functions_norm; 
          }
        }

      }
         
      // Regularisation according to respiratory basis functions
      
      if( m_nbRGModelParam>1 )
      {
        if(m_verbose >=3) Cout("iLinearModel::NestedEM() -> Estimate Coefficients - Respiratory Gate coeffs estimation step" <<endl);
          
        for (int rb=0 ; rb<m_nbRgateBF ; rb++)
        {
          // Initialization of voxelwise corrections for time basis functions coefficients
          for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
            mp_corrBasisCoeffs[v] = 0;
          
          basis_functions_norm = 0.; 
          
          for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
            for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++) 
              for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                for (int fb=0 ; fb<m_nbTimeBF ; fb++)
                  for (int cb=0 ; cb<m_nbCgateBF ; cb++)
                    basis_functions_norm += m2p_modelTACs[fb][fr]
                                          * m2p_RGModelTACs[rb][rg]
                                          * m2p_CGModelTACs[cb][cg];
                
          // Compute corrections for the time vowelwise basis functions coefficients  
          int v;
          #pragma omp parallel for private(v) schedule(static, 1)
          for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
          {
            for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
              for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
                for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                  for (int fb=0 ; fb<m_nbTimeBF ; fb++)
                    for (int cb=0 ; cb<m_nbCgateBF ; cb++)
                    {
                      mp_corrBasisCoeffs[v] += ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v] 
                                             * m2p_modelTACs[fb][fr] 
                                             * m2p_RGModelTACs[rb][rg]
                                             * m2p_CGModelTACs[cb][cg];
                    }
                    
            // Apply corrections and normalization to the respiratory model parametric image.
            if (mp_corrBasisCoeffs[v] > 0.)
              m2p_RGParametricImages[rb][v] *= mp_corrBasisCoeffs[v]/basis_functions_norm;
          }
        }
      }
  
  
      // Regularisation according to cardiac basis functions
      if( m_nbCGModelParam>1 )
      {
        if(m_verbose >=3) Cout("iLinearModel::NestedEM() -> Estimate Coefficients - Cardiac Gate coeffs estimation step" <<endl);
          
        for (int cb=0 ; cb<m_nbCgateBF ; cb++)
        {
          // Initialization of voxelwise corrections for time basis functions coefficients
          for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
            mp_corrBasisCoeffs[v] = 0;
          
          basis_functions_norm = 0.; 
          
          for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
            for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++) 
              for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                for (int fb=0 ; fb<m_nbTimeBF ; fb++)
                  for (int rb=0 ; rb<m_nbRgateBF ; rb++)
                    basis_functions_norm += m2p_modelTACs[fb][fr]
                                          * m2p_RGModelTACs[rb][rg]
                                          * m2p_CGModelTACs[cb][cg];
                
          // Compute corrections for the time vowelwise basis functions coefficients  
          int v;
          #pragma omp parallel for private(v) schedule(static, 1)
          for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
          {
            for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
              for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
                for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
                  for (int fb=0 ; fb<m_nbTimeBF ; fb++)
                    for (int rb=0 ; rb<m_nbRgateBF ; rb++)
                    {     
                      mp_corrBasisCoeffs[v] += ap_ImageS->m6p_backwardImage[0][0][fr][rg][cg][v] 
                                             * m2p_modelTACs[fb][fr] 
                                             * m2p_RGModelTACs[rb][rg]
                                             * m2p_CGModelTACs[cb][cg];
                                                 
                    }
                    
            // Apply corrections and normalization to the cardiac model parametric image.
            if (mp_corrBasisCoeffs[v] > 0.)
              m2p_CGParametricImages[cb][v] *= mp_corrBasisCoeffs[v]/basis_functions_norm;
          }
        }
      }
      
      m_basisFunctionsUpdIdx++;
      
      // Some feedback :
      if(m_verbose >=3)
      {
        if( m_nbModelParam>1 )
          for (int fb=0 ; fb<m_nbTimeBF ; fb++) 
          { 
            HPFLTNB avg = 0;
            
            for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
              avg += m2p_parametricImages[fb][v];
            
            avg /= mp_ID->GetNbVoxXYZ();
            Cout( "iLinearModel::NestedEM() -> Frame parametric image["<<fb<<"] avg value:" << avg << endl);
          }
  
        if( m_nbRGModelParam>1 )
          for (int rb=0 ; rb<m_nbRgateBF ; rb++) 
          { 
            HPFLTNB avg = 0;
            
            for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
              avg += m2p_RGParametricImages[rb][v];
            
            avg /= mp_ID->GetNbVoxXYZ();
            Cout( "iLinearModel::NestedEM() -> Resp gate parametric image["<<rb<<"] avg value:" << avg << endl);
          }
        
        if( m_nbCGModelParam>1 )
          for (int cb=0 ; cb<m_nbCgateBF ; cb++) 
          { 
            HPFLTNB avg = 0;
            
            for (int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
              avg += m2p_CGParametricImages[cb][v];
            
            avg /= mp_ID->GetNbVoxXYZ();
            Cout( "iLinearModel::NestedEM() -> Card gate parametric image["<<cb<<"] avg value:" << avg << endl);
          }
      }
      
    } // end of else loop (parametric images update)
    
  } // end of loop on cycles (linear model iterations)

  return 0;
}



// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn EstimateImageWithModel
  \param ap_ImageS : pointer to the ImageSpace
  \param a_ite : index of the actual iteration (not used)
  \param a_sset : index of the actual subset (not used)
  \brief Re-estimate image using the linear parametric images and basis functions
  \return 0 if success, other value otherwise.
*/
int iLinearModel::EstimateImageWithModel(oImageSpace* ap_ImageS, int a_ite, int a_sset) 
{
  if(m_verbose >= 3) Cout("iLinearModel::EstimateImageWithModel ... " <<endl);

  #ifdef CASTOR_DEBUG
  if (!m_initialized)
  {
    Cerr("***** iLinearModel::EstimateImageWithModel() -> Called while not initialized !" << endl);
    Exit(EXIT_DEBUG);
  }
  #endif

  int v;
  #pragma omp parallel for private(v) schedule(static, 1)
  for (int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
    for (int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
      for (int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
        for (v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
        {
          // Reset current estimated value
          ap_ImageS->m4p_image[fr][rg][cg][v] = 0.;

          for (int fb=0 ; fb<m_nbTimeBF ; fb++)
            for (int rb=0 ; rb<m_nbRgateBF ; rb++)
              for (int cb=0 ; cb<m_nbCgateBF ; cb++)
              {
                // Retrieve current estimation of image according to coeffs/basis functions
                ap_ImageS->m4p_image[fr][rg][cg][v] += m2p_parametricImages[fb][v] 
                                                     * m2p_modelTACs[fb][fr] 
                                                     * m2p_RGParametricImages[rb][v] 
                                                     * m2p_RGModelTACs[rb][rg]
                                                     * m2p_CGParametricImages[cb][v] 
                                                     * m2p_CGModelTACs[cb][cg] ;
              }
        }
  
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      SaveParametricImages
  \param   a_iteration : current iteration index
  \param   a_subset : current number of subsets (or -1 by default)
  \brief   This function is virtual it can be overloaded by children 
           if required
  \return  0 if success, positive value otherwise
*/
int iLinearModel::SaveParametricImages(int a_iteration, int a_subset)
{
  if(m_verbose >=3) Cout("iLinearModel::SaveParametricImages ..." <<endl);
  
          
  if(m_saveParImageFlag)
  {
    // Get the output manager
    sOutputManager* p_output_manager = sOutputManager::GetInstance();
    
    // Recover path to output interfile
    string path_to_image = p_output_manager->GetPathName() + p_output_manager->GetBaseName();

    // Write interfile frame parametric image if required 
    if( m_nbModelParam>1 ) // Frame linear model enabled
    {
      // Add a suffix for iteration
      if (a_iteration >= 0)
      {
        stringstream ss; ss << a_iteration + 1;
        path_to_image.append("par_it").append(ss.str());
      }
  
      // Add a suffix for subset (if not negative by default), this means that we save a 'subset' image
      if (a_subset >= 0)
      {
        stringstream ss; ss << a_subset + 1;
        path_to_image.append("_ss").append(ss.str());
      }

      if(IntfWriteImgDynCoeffFile(path_to_image, 
                                  m2p_outputParImages, 
                                  mp_ID, 
                                  m_nbModelParam, 
                                  m_verbose) )
      {
        Cerr("***** iLinearModel::SaveParametricImages()-> Error writing Interfile of output image !" << endl);  
        return 1;
      }
    }

    // Write interfile respiratory gating parametric image if required 
    if( m_nbRGModelParam>1 ) // Respiratory gate linear model enabled
    {
      // Recover path to output interfile
      path_to_image = p_output_manager->GetPathName() + p_output_manager->GetBaseName();
      
      // Add a suffix for iteration
      if (a_iteration >= 0)
      {
        stringstream ss; ss << a_iteration + 1;
        path_to_image.append("parRGmodel_it").append(ss.str());
      }
  
      // Add a suffix for subset (if not negative by default), this means that we save a 'subset' image
      if (a_subset >= 0)
      {
        stringstream ss; ss << a_subset + 1;
        path_to_image.append("_ss").append(ss.str());
      }
      
      if(IntfWriteImgDynCoeffFile(path_to_image, 
                                  m2p_RGParametricImages, 
                                  mp_ID, 
                                  m_nbRGModelParam, 
                                  m_verbose) )
      {
        Cerr("***** iLinearModel::SaveParametricImages()-> Error writing Interfile of output image !" << endl);  
        return 1;
      }
    }

    // Write interfile cardiac gating parametric image if required 
    if( m_nbCGModelParam>1 ) // Cardiac gate linear model enabled
    {
      // Recover path to output interfile
      path_to_image = p_output_manager->GetPathName() + p_output_manager->GetBaseName();
      
      // Add a suffix for iteration
      if (a_iteration >= 0)
      {
        stringstream ss; ss << a_iteration + 1;
        path_to_image.append("parCGmodel_it").append(ss.str());
      }
  
      // Add a suffix for subset (if not negative by default), this means that we save a 'subset' image
      if (a_subset >= 0)
      {
        stringstream ss; ss << a_subset + 1;
        path_to_image.append("_ss").append(ss.str());
      }
      
      if(IntfWriteImgDynCoeffFile(path_to_image, 
                                  m2p_CGParametricImages, 
                                  mp_ID, 
                                  m_nbCGModelParam, 
                                  m_verbose) )
      {
        Cerr("***** iLinearModel::SaveParametricImages()-> Error writing Interfile of output image !" << endl);  
        return 1;
      }
    }
    
  }

  return 0;
}
