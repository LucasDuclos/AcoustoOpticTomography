
/*!
  \file
  \ingroup  dynamic
  \brief    Implementation of class vDynamicModel
*/

#include "vDynamicModel.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn vDynamicModel
  \brief Constructor of vDynamicModel. Simply set all data members to default values.
*/
vDynamicModel::vDynamicModel() 
{
  mp_ID = NULL;
  m_nbTimeBF = -1;
  m_nbModelParam = -1;
  m2p_parametricImages = NULL;
  m2p_modelTACs = NULL;
  m2p_outputParImages = NULL;
  m_verbose = -1;
  
  m_checked = false;
  m_initialized = false;
  m_saveParImageFlag = true;
  m_noImageUpdateFlag = false;
  m_noParametersUpdateFlag = false;
  m_startIteUpdateFlag = 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*!
  \fn ~vDynamicModel
  \brief Destructor of vDynamicModel.
*/
vDynamicModel::~vDynamicModel() 
{
  if(m_initialized)
  {
    for(int b=0 ; b<m_nbTimeBF ; b++)
    {
      if (m2p_modelTACs  && m2p_modelTACs[b] )        
      {
        delete[] m2p_modelTACs[b];
        m2p_modelTACs[b] = NULL;
      }
    }

    if (m2p_modelTACs) 
    {
      delete[] m2p_modelTACs;
      m2p_modelTACs = NULL;
    }
    
    for(int p=0 ; p<m_nbModelParam ; p++)
    {
      if (m2p_outputParImages != m2p_parametricImages
       && m2p_outputParImages  && m2p_outputParImages[p])
      {
        delete[] m2p_outputParImages[p];
        m2p_outputParImages[p] = NULL;
      }
      if (m2p_parametricImages && m2p_parametricImages[p]) 
      {
        delete[] m2p_parametricImages[p];
        m2p_parametricImages[p] = NULL;
      }
    }
 
    if (m2p_outputParImages
     && m2p_outputParImages != m2p_parametricImages)
    {
      delete[] m2p_outputParImages;
      m2p_outputParImages = NULL;
    }
    if (m2p_parametricImages) 
    {
      delete[] m2p_parametricImages;
      m2p_parametricImages = NULL;
    }
  }
}



// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*!
  \fn CheckParameters
  \brief This function is used to check parameters after the latter
         have been all set using Set functions.
  \return 0 if success, positive value otherwise.
*/
int vDynamicModel::CheckParameters()
{
  
  if(m_verbose>=2) Cout("vDynamicModel::CheckParameters ..."<< endl); 
    
  // Check image dimensions
  if (mp_ID==NULL)
  {
    Cerr("***** vDynamicModel::CheckParameters() -> No image dimensions provided !" << endl);
    return 1;
  }
  
  // Check verbosity
  if (m_verbose<0)
  {
    Cerr("***** vDynamicModel::CheckParameters() -> Wrong verbosity level provided !" << endl);
    return 1;
  }

  // Check number of basis functions
  if (m_nbTimeBF <0)
  {
    Cerr("***** vDynamicModel::CheckParameters() -> Basis functions number has not been initialized !" << endl);
    return 1;
  }

  // Check number of parameters in the model
  if (m_nbModelParam <0)
  {
    Cerr("***** vDynamicModel::CheckParameters() -> Number of model parameter has not been initialized !" << endl);
    return 1;
  }
  
  // Check parameters of the child class (if this function is overloaded)
  if (CheckSpecificParameters())
  {
    Cerr("***** vDynamicModel::CheckParameters() -> An error occurred while checking parameters of the child dynamic class !" << endl);
    return 1;
  }
  
  // Normal end
  m_checked = true;
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      vDynamicModel::Initialize()
  \brief   A public function used to initialize the dynamic model
  \details This function does not take any parameter and is used to initialize everything that
           should be initialized. At the end, it calls the pure virtual InitializeSpecific()
           function implemented by children.
  \return  An integer reflecting the initialization status; 0 if no problem, another value otherwise.
*/
int vDynamicModel::Initialize()
{
  if(m_verbose >=2) Cout("vDynamicModel::Initialize ..."<< endl); 

  // Forbid initialization without check
  if (!m_checked)
  {
    Cerr("***** vDynamicModelr::Initialize() -> Must call CheckParameters functions before Initialize() !" << endl);
    return 1;
  }





  // Call the specific initialization function of the child
  if (InitializeSpecific())
  {
    Cerr("***** vDynamicModel::Initialize() -> A problem occurred while initializing stuff specific to the dynamic model !" << endl);
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
  \fn      vDynamicModel::ComputeOutputParImage
  \brief   Compute output image using the m2p_parametricImages matrix Store the result in the m2p_outputParImages matrix
*/
void vDynamicModel::ComputeOutputParImage()
{
  if(m_verbose >=2) Cout("vDynamicModel::ComputeOutputParImage ..." <<endl);
  
  // If we save parametric image
  if(m_saveParImageFlag)
    for(int p=0 ; p<m_nbModelParam ; p++)
    {
      // If output image matrix is allocated, then copy the current parametric images
      if(m2p_outputParImages    != NULL 
      && m2p_outputParImages[p] != m2p_parametricImages[p])
      {
        for(int v=0 ; v<mp_ID->GetNbVoxXYZ() ; v++)
          m2p_outputParImages[p][v] = m2p_parametricImages[p][v] ;
      }
      // Point to parametric images otherwise
      else
        m2p_outputParImages = m2p_parametricImages;
    }
    
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      vDynamicModel::ApplyOutputFOVMaskingOnParametricImages
  \brief   Mask the outside of the transaxial FOV based on the m_fovOutPercent
  \details Similar to the eponym function in ImageSpace, but on parametric images
*/
int vDynamicModel::ApplyOutputFOVMaskingOnParametricImages()
{
  // If the output FOV percent is under 0 (the default value) and number of axial slices to be removed is 0, we do not mask anything
  if ( mp_ID->GetFOVOutPercent()<=0. &&
       mp_ID->GetNbSliceOutMask()==0 ) return 0;
       
  // Verbose
  if (m_verbose>=2)
  {
    Cout("vDynamicModel::ApplyOutputFOVMaskingOnParametricImages() -> Mask output image" << endl);
    if (mp_ID->GetFOVOutPercent()>0.) Cout("  --> Mask transaxial FOV outside " << mp_ID->GetFOVOutPercent() << " %" << endl);
    if (mp_ID->GetNbSliceOutMask()>0) Cout("  --> Mask " << mp_ID->GetNbSliceOutMask() << " slices from both axial FOV limits" << endl);
  }
  // -----------------------------------------------
  // Transaxial FOV masking
  // -----------------------------------------------
  if (mp_ID->GetFOVOutPercent()>0.)
  {
    // Precast half the number of voxels over X and Y minus 1 (for efficiency)
    FLTNB flt_base_x = 0.5*((FLTNB)(mp_ID->GetNbVoxX()-1));
    FLTNB flt_base_y = 0.5*((FLTNB)(mp_ID->GetNbVoxY()-1));
    
    // Compute FOV elipse radius over X and Y, then squared
    FLTNB squared_radius_x = 0.5 * ((FLTNB)(mp_ID->GetNbVoxX())) * mp_ID->GetVoxSizeX()
                           * mp_ID->GetFOVOutPercent() / 100.;
    squared_radius_x *= squared_radius_x;
    FLTNB squared_radius_y = 0.5 * ((FLTNB)(mp_ID->GetNbVoxY())) * mp_ID->GetVoxSizeY()
                           * mp_ID->GetFOVOutPercent() / 100.;
    squared_radius_y *= squared_radius_y;
    
    // We assume that the computation of the distance from the center for a given
    // voxel and comparing it with the output FOV percentage costs more than performing
    // the loops in an inverse order compared to how the image is stored in memory.
    // Thus we begin the loops over X, then Y, then we test and if test passes, we
    // do the remaining loop over Z and over all dynamic dimensions.
    int x;
    #pragma omp parallel for private(x) schedule(guided)
    for (x=0; x<mp_ID->GetNbVoxX(); x++)
    {
      // Compute X distance from image center, then squared
      FLTNB squared_distance_x = (((FLTNB)x)-flt_base_x) * mp_ID->GetVoxSizeX();
      squared_distance_x *= squared_distance_x;
      // Loop over Y
      for (int y=0; y<mp_ID->GetNbVoxY(); y++)
      {
        // Compute Y distance from image center, then squared
        FLTNB squared_distance_y = (((FLTNB)y)-flt_base_y) * mp_ID->GetVoxSizeY();
        squared_distance_y *= squared_distance_y;
        // Test if the voxel is inside the FOV elipse, then we skip this voxel
        if ( squared_distance_x/squared_radius_x + squared_distance_y/squared_radius_y <= 1. ) continue;
        // Loop over Z
        for (int z=0; z<mp_ID->GetNbVoxZ(); z++)
        {
          // Compute global voxel index
          INTNB index = z*mp_ID->GetNbVoxXY() + y*mp_ID->GetNbVoxX() + x;
          
          for (int b=0 ; b<m_nbTimeBF ; b++)
            m2p_outputParImages[b][index] = 0.;
            
        }
      }
    }
  }
  
  // -----------------------------------------------
  // Axial FOV masking
  // -----------------------------------------------
  if (mp_ID->GetNbSliceOutMask()>0)
  {
    INTNB removed_slices = mp_ID->GetNbSliceOutMask();
        
    // Mask slices
    for (int b=0 ; b<m_nbTimeBF ; b++)
      for (int z=0; z<removed_slices; z++)
      {
        // First slices
        INTNB base_z_first = z*mp_ID->GetNbVoxXY();
        // Loop over Y and X
        for (int i=0; i<mp_ID->GetNbVoxXY(); i++)
        {
          INTNB index = base_z_first + i;
          m2p_outputParImages[b][index] = 0.;
        }
        // Last slices
        INTNB base_z_last = (mp_ID->GetNbVoxZ()-1-z)*mp_ID->GetNbVoxXY();
        // Loop over Y and X
        for (int i=0; i<mp_ID->GetNbVoxXY(); i++)
        {
          INTNB index = base_z_last + i;
          m2p_outputParImages[b][index] = 0.;
        }
      }
  }
  
  // End
  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      vDynamicModel::EstimateModel
  \param   ap_ImageS : pointer to the ImageSpace
  \param   a_ite : index of the actual iteration
  \param   a_sset : index of the actual subset
  \brief   This function checks if the EstimateModelParameters() function (specific to each model)
           must be called at this stage of the reconstruction depending on the m_xxxUpdateflags.
  \return  0 if success, other value otherwise.
*/
int vDynamicModel::EstimateModel(oImageSpace* ap_ImageS, int a_ite, int a_sset) 
{
  if(m_verbose >= 2) Cout("vDynamicModel::EstimateModel ... " <<endl);

  #ifdef CASTOR_DEBUG
  if (!m_initialized)
  {
    Cerr("***** vDynamicModel::EstimateModel() -> Called while not initialized !" << endl);
    Exit(EXIT_DEBUG);
  }
  #endif

  if(!m_noParametersUpdateFlag)
    if( EstimateModelParameters(ap_ImageS, a_ite, a_sset) )
    {
      Cerr("***** vDynamicModel::EstimateModel() -> A problem occurred while estimating dynamic image series with model parameters !" << endl);
      return 1;
    }

  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      vDynamicModel::EstimateImageWithModel
  \param   ap_ImageS : pointer to the ImageSpace
  \param   a_ite : index of the actual iteration
  \param   a_sset : index of the actual subset
  \brief   This function checks if the EstimateImageWithModel() function (specific to each model)
           must be called at this stage of the reconstruction depending on the m_xxxUpdateflags.
  \return  0 if success, other value otherwise.
*/
int vDynamicModel::EstimateImage(oImageSpace* ap_ImageS, int a_ite, int a_sset) 
{
  if(m_verbose >= 2) Cout("vDynamicModel::EstimateImage ... " <<endl);

  #ifdef CASTOR_DEBUG
  if (!m_initialized)
  {
    Cerr("***** vDynamicModel::EstimateImage() -> Called while not initialized !" << endl);
    Exit(EXIT_DEBUG);
  }
  #endif

  if(!m_noImageUpdateFlag && a_ite>=m_startIteUpdateFlag)
    if( EstimateImageWithModel(ap_ImageS, a_ite, a_sset) )
    {
      Cerr("***** vDynamicModel::EstimateImage() -> A problem occurred while applying dynamic model to current estimate images !" << endl);
      return 1;
    }

  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      ReadAndCheckConfigurationFile
  \param   const string& a_configurationFile : ASCII file containing informations about a dynamic model
  \brief   This function is used to read options from a configuration file. \n
           It looks for the parameters implemented by the mother class,
           such as 'No image update', 'No parameters update', or 'Save parametric images'.
           Call 'ReadAndCheckConfigurationFileSpecific()' function of child class
  \return  0 if success, other value otherwise.
*/
int vDynamicModel::ReadAndCheckConfigurationFile(string a_fileOptions)
{
  if(m_verbose >=3) Cout("vDynamicModel::ReadAndCheckConfigurationFile ..."<< endl); 
  
  ifstream in_file(a_fileOptions.c_str(), ios::in);
  
  if(in_file)
  {
    
    string line="", kword_to_search="";
    while(!in_file.eof())
    {
      getline(in_file, line);
  
      //remove comment
      if (line.find("#") != string::npos) line = line.substr(0, line.find_first_of("#")) ;
      
    
      // Disable estimation of images from parametric images ?
      string dtag = "No image update";
      
      if( ReadDataASCIIFile(a_fileOptions,
                            dtag,
                            &m_noImageUpdateFlag,
                            1,
                            KEYWORD_OPTIONAL) == 1)
      {
        Cerr("***** vDynamicModel::ReadAndCheckConfigurationFile() -> Error while trying to read '"<< dtag <<"' flag in " << a_fileOptions << endl);
        return 1;
      }
      
      
      // Disable estimation of parameters from dynamic images ?
      dtag = "No parameters update";
      
      if( ReadDataASCIIFile(a_fileOptions,
                            dtag,
                            &m_noParametersUpdateFlag,
                            1,
                            KEYWORD_OPTIONAL) == 1)
      {
        Cerr("***** vDynamicModel::ReadAndCheckConfigurationFile() -> Error while trying to read '"<< dtag <<"' flag in " << a_fileOptions << endl);
        return 1;
      }
      
      
      // Update images from parametric images after a certain number of iterations ?
      dtag = "Number of iterations before image update";
      
      if( ReadDataASCIIFile(a_fileOptions,
                            dtag,
                            &m_startIteUpdateFlag,
                            1,
                            KEYWORD_OPTIONAL) == 1)
      {
        Cerr("***** vDynamicModel::ReadAndCheckConfigurationFile() -> Error while trying to read '"<< dtag <<"' flag in " << a_fileOptions << endl);
        return 1;
      }
      
      
      // Save parametric images on disk ?
      dtag = "Save parametric images";
      
      if( ReadDataASCIIFile(a_fileOptions, 
                            dtag, 
                            &m_saveParImageFlag, 
                            1, 
                            KEYWORD_OPTIONAL) == 1)
      {
        Cerr("***** vDynamicModel::ReadAndCheckConfigurationFile() -> Error while trying to read '"<< dtag <<"' flag in " << a_fileOptions << endl);
        return 1;
      }
    
      // Recover the file path here
      m_fileOptions = a_fileOptions; 
    }
    
    // Call the specific function of the child dynamic model class
    if( ReadAndCheckConfigurationFileSpecific() )
    {
      Cerr("***** vDynamicModel::ReadAndCheckConfigurationFile() -> Error occured when trying to process the configuration file from a dynamic model child class " << endl);
      return 1;
    }
  }
  else
  {
    Cerr("***** iLinearModel::ReadAndCheckConfigurationFile -> Error while trying to read configuration file at: " << a_fileOptions << endl);
    return 1;
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
int vDynamicModel::SaveParametricImages(int a_iteration, int a_subset)
{
  if(m_verbose >=2) Cout("vDynamicModel::SaveParametricImages ..." <<endl);
  
          
  if(m_saveParImageFlag)
  {
    // Get the output manager
    sOutputManager* p_output_manager = sOutputManager::GetInstance();
    
    // Interfile
    string path_to_image = p_output_manager->GetPathName() + p_output_manager->GetBaseName();
    
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
  
    // Write interfile parametric image
    if(IntfWriteImgDynCoeffFile(path_to_image, 
                                m2p_outputParImages, 
                                mp_ID, 
                                m_nbModelParam, 
                                m_verbose) )
    {
      Cerr("***** vDynamicModel::SaveParametricImages()-> Error writing Interfile of output image !" << endl);  
      return 1;
    }
    
  }
  
  return 0;
}







// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      NNLS
  \param   a : On entry, a[ 0... N ][ 0 ... M ] contains the M by N matrix A.\n
               On exit, a[][] contains the product matrix Q*A, where Q is an m by n \n
               orthogonal matrix generated implicitly by this function.
  \param   m : Matrix dimension m
  \param   n : Matrix dimension n
  \param   b : On entry, b[] must contain the m-vector B. \n
               On exit, b[] contains Q*B
  \param   x : On exit, x[] will contain the solution vector
  \param   rnorm : On exit, rnorm contains the Euclidean norm of the residual vector. \n
                   If NULL is given, no rnorm is calculated
  \param   wp: An n-array of working space, wp[].  \n
               On exit, wp[] will contain the dual solution vector.  \n
               wp[i]=0.0 for all i in set p and wp[i]<=0.0 for all i in set z.  \n
               Can be NULL, which causes this algorithm to allocate memory for it.
  \param   zzp : An m-array of working space, zz[]. \n
                 Can be NULL, which causes this algorithm to allocate memory for it.
  \param   indexp : An n-array of working space, index[]. \n
                    Can be NULL, which causes this algorithm to allocate memory for it. *
  \brief   Implementation of NNLS (non-negative least squares) algorithm
           Derived from Turku PET center libraries (authors: Vesa Oikonen and Kaisa Sederholm)
           This routine is based on the text and fortran code in
           C.L. Lawson and R.J. Hanson, Solving Least Squares Problems,
           Prentice-Hall, Englewood Cliffs, New Jersey, 1974.
  \details Given an m by n matrix A, and an m-vector B, computes an n-vector X,
           that solves the least squares problem
           A * X = B   , subject to X>=0
          
           Instead of pointers for working space, NULL can be given to let this
           function to allocate and free the required memory.

  \return  0 if success, positive value otherwise
*/
int vDynamicModel::NNLS( HPFLTNB **A,
                         int m,
                         int n,
                         HPFLTNB *B,
                         HPFLTNB *X,
                         HPFLTNB *rnorm,
                         HPFLTNB *wp,
                         HPFLTNB *zzp,
                         int *indexp ) 
{
  int pfeas, iz, jz, iz1, iz2, npp1, *index;
  HPFLTNB d1, d2, sm, up, ss, *w, *zz;
  int iter, k, j=0, l, itmax, izmax=0, nsetp, ii, jj=0, ip;
  HPFLTNB temp, wmax, t, alpha, asave, dummy, unorm, ztest, cc;

  // Check the parameters and data */
  if(m<=0 || n<=0 || A==NULL || B==NULL || X==NULL)
  {
    Cerr("***** vDynamicModel::NNLS()-> Incorrect input parameters !" << endl);  
    return 1;
  }
  
  // Allocate memory for working space, if required
  if(wp!=NULL)   w=wp; else w=(HPFLTNB*)calloc(n, sizeof(HPFLTNB));
  if(zzp!=NULL) zz=zzp; else zz=(HPFLTNB*)calloc(m, sizeof(HPFLTNB));
  if(indexp!=NULL) index=indexp; else index=(int*)calloc(n, sizeof(int));
  
  if(w==NULL || zz==NULL || index==NULL)
  {
    Cerr("***** vDynamicModel::NNLS()-> Incorrect memory allocation on working space !" << endl);  
    return 1;
  }
  
  // Initialize the arrays INDEX[] and X[]
  for(k=0; k<n; k++) 
  {
    X[k] = 0.;
    index[k] = k;
  }
  
  iz2=n-1; iz1=0; nsetp=0; npp1=0;

  // Main loop; quit if all coeffs are already in the solution or
  // if M cols of A have been triangularized
  iter=0; 
  
  if(n<3) 
    itmax=n*3; 
  else 
    itmax=n*n;
    
  while(iz1<=iz2 && nsetp<m) 
  {
    // Compute components of the dual (negative gradient) vector W[]
    for(iz=iz1; iz<=iz2; iz++) 
    {
      j=index[iz]; sm=0.; 
      for(l=npp1; l<m; l++) 
        sm+=A[j][l]*B[l];
        
      w[j]=sm;
    }

    while(1) 
    {
      // Find largest positive W[j] */
      for(wmax=0., iz=iz1; iz<=iz2; iz++) 
      {
        j=index[iz]; 
        if(w[j]>wmax) 
        {
          wmax=w[j]; 
          izmax=iz;
        }
      }

      // Terminate if wmax<=0.;
      // it indicates satisfaction of the Kuhn-Tucker conditions
      if(wmax<=0.0) break;
      
      iz=izmax; j=index[iz];

      // The sign of W[j] is ok for j to be moved to set P.
      // Begin the transformation and check new diagonal element to avoid
      // near linear dependence.
      asave=A[j][npp1]; up=0.0;
      
      NNLS_LSS_H12(1, npp1, npp1+1, m, &A[j][0], 1, &up, &dummy, 1, 1, 0);
      
      unorm=0.;
      if(nsetp!=0) 
        for(l=0; l<nsetp; l++) 
        {
          d1 = A[j][l];
          unorm+=d1*d1;
        }
      unorm=sqrt(unorm);
      d2 = unorm + (d1=A[j][npp1], fabs(d1)) * 0.01;
      if( (d2-unorm)>0. ) 
      {
        // Col j is sufficiently independent. Copy B into ZZ, update ZZ 
        // and solve for ztest ( = proposed new value for X[j] )
        for(l=0; l<m; l++) 
          zz[l]=B[l];
          
        NNLS_LSS_H12(2, npp1, npp1+1, m, &A[j][0], 1, &up, zz, 1, 1, 1);
        
        ztest = zz[npp1]/A[j][npp1];
        // See if ztest is positive */
        if(ztest>0.) break;
      }

      // Reject j as a candidate to be moved from set Z to set P. 
      // Restore A[npp1,j], set W[j]=0., and loop back to test dual coeffs again 
      A[j][npp1]=asave;
      w[j]=0.;
    } // while(1) */
    
    if(wmax<=0.0) break;

    // Index j=INDEX[iz] has been selected to be moved from set Z to set P.
    // Update B and indices, apply householder transformations to cols in
    // new set Z, zero subdiagonal elts in col j, set W[j]=0.
    for(l=0; l<m; ++l) 
      B[l]=zz[l];
      
    index[iz]=index[iz1]; index[iz1]=j; iz1++; nsetp=npp1+1; npp1++;
    
    if(iz1<=iz2) for(jz=iz1; jz<=iz2; jz++) 
    {
      jj=index[jz];
      NNLS_LSS_H12(2, nsetp-1, npp1, m, &A[j][0], 1, &up, &A[jj][0], 1, m, 1);
    }
    
    if(nsetp!=m) for(l=npp1; l<m; l++) A[j][l]=0.;
    w[j]=0.;
    
    // Solve the triangular system; store the solution temporarily in Z[]
    for(l=0; l<nsetp; l++) 
    {
      ip=nsetp-(l+1);
      if(l!=0) 
        for(ii=0; ii<=ip; ii++) 
          zz[ii] -= A[jj][ii] * zz[ip+1];
      jj = index[ip]; 
      zz[ip] /= A[jj][ip];
    }

    // Secondary loop begins here
    while( ++iter<itmax ) 
    {
      // See if all new constrained coeffs are feasible; if not, compute alpha
      for(alpha=2.0, ip=0; ip<nsetp; ip++) 
      {
        l=index[ip];
        if(zz[ip]<=0.) 
        {
          t = -X[l] / (zz[ip]-X[l]);
          if(alpha>t) 
          {
            alpha=t; 
            jj=ip-1;
          }
        }
      }

      // If all new constrained coeffs are feasible then still alpha==2. 
      // If so, then exit from the secondary loop to main loop 
      if(alpha==2.0) break;

      // Use alpha (0.<alpha<1.) to interpolate between old X and new ZZ
      for(ip=0; ip<nsetp; ip++) 
      {
        l = index[ip]; 
        X[l] += alpha*(zz[ip]-X[l]);
      }

      // Modify A and B and the INDEX arrays to move coefficient i 
      // from set P to set Z. 
      k=index[jj+1]; pfeas=1;
      do 
      {
        X[k]=0.;
        if(jj!=(nsetp-1)) 
        {
          jj++;
          for(j=jj+1; j<nsetp; j++) 
          {
            ii=index[j]; index[j-1]=ii;
            
            NNLS_LSS_G1(A[ii][j-1], A[ii][j], &cc, &ss, &A[ii][j-1]);
            
            for(A[ii][j]=0., l=0; l<n; l++) if(l!=ii) 
            {
              // Apply procedure G2 (CC,SS,A(J-1,L),A(J,L)) */
              temp = A[l][j-1];
              A[l][j-1] = cc*temp+ss*A[l][j];
              A[l][j] = -ss*temp+cc*A[l][j];
            }
            // Apply procedure G2 (CC,SS,B(J-1),B(J))
            temp=B[j-1]; B[j-1]=cc*temp+ss*B[j]; B[j]=-ss*temp+cc*B[j];
          }
        }
        npp1=nsetp-1; nsetp--; iz1--; index[iz1]=k;

        // See if the remaining coeffs in set P are feasible; 
        // they should be because of the way alpha was determined. 
        // If any are infeasible it is due to round-off error. 
        // Any that are nonpositive will be set to zero and moved from set P to set Z 
        for(jj=0, pfeas=1; jj<nsetp; jj++) 
        {
          k=index[jj]; 
          if(X[k]<=0.) 
          {
            pfeas=0; 
            break;
          }
        }
      } while(pfeas==0);

      // Copy B[] into zz[], then solve again and loop back 
      for(k=0; k<m; k++) 
        zz[k]=B[k];
        
      for(l=0; l<nsetp; l++) 
      {
        ip = nsetp-(l+1);
        if(l!=0) 
          for(ii=0; ii<=ip; ii++) 
            zz[ii] -= A[jj][ii]*zz[ip+1];
        jj = index[ip]; 
        zz[ip] /= A[jj][ip];
      }
    } // end of secondary loop 

    // Normal return if we reach max number of iteration.
    // Maybe throw warning here
    if(iter>=itmax)
      return 0;
      
    for(ip=0; ip<nsetp; ip++) 
    {
      k=index[ip]; 
      X[k]=zz[ip];
    }
    
  } // end of main loop 
  
  // Compute the norm of the final residual vector
  sm=0.;
  
  if(rnorm != NULL) 
  {
    if(npp1<m) 
    {
      for(k=npp1; k<m; k++) 
        sm+=(B[k]*B[k]);
    }
    else 
    {
      for(j=0; j<n; j++) 
        w[j]=0.;
    }
        
    *rnorm=sqrt(sm);
  }	
 
  // Free working space, if it was allocated here
  if(wp==NULL) free(w);
  if(zzp==NULL) free(zz);
  if(indexp==NULL) free(index);
  
  return 0;
}








// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      NNLS_LSS_G1
  \param   mode : mode=1 to construct and apply a Householder transformation, or \n
                  mode=2 to apply a previously constructed transformation
  \param   lpivot: Index of the pivot element, on pivot vector
  \param   l1: Transformation is constructed to zero elements indexed from l1 to M
  \param   m: Transformation is constructed to zero elements indexed from l1 to M
  \param   u:  With mode=1: On entry, u[] must contain the pivot vector.
                            On exit, u[] and up contain quantities defining 
                            the vector u[] of the Householder transformation.
               With mode=2: On entry, u[] and up should contain quantities previously
                            computed with mode=1. These will not be modified
  \param   u_dim1: u_dim1 is the storage increment between elements
  \param   up: with mode=1, here is stored an element defining housholder vector scalar,
                 on mode=2 it's only used, and is not modified
  \param   cm: On entry, cm[] must contain the matrix (set of vectors) to which the
               Householder transformation is to be applied. 
               On exit, cm[] will contain the set of transformed vectors
  \param   ice: Storage increment between elements of vectors in cm[] 
  \param   icv: Storage increment between vectors in cm[]
  \param   nvc: Nr of vectors in cm[] to be transformed;
                if ncv<=0, then no operations will be done on cm[]
  \brief   Construction and/or application of a single Householder transformation:
           Q = I + U*(U**T)/B
           Derived from Turku PET center libraries (authors: Vesa Oikonen and Kaisa Sederholm)
           This routine is based on the text and fortran code in
           C.L. Lawson and R.J. Hanson, Solving Least Squares Problems,
           Prentice-Hall, Englewood Cliffs, New Jersey, 1974.
  \return  0 if success, positive value otherwise (erroneous parameters)
*/
int vDynamicModel::NNLS_LSS_H12(int mode,
                                int lpivot,
                                int l1,
                                int m,
                                HPFLTNB *u,
                                int u_dim1,
                                HPFLTNB *up,
                                HPFLTNB *cm,
                                int ice,
                                int icv,
                                int ncv
) 
{
  HPFLTNB d1, b, clinv, cl, sm;
  int k, j;
  
  /* Check parameters */
  if(mode!=1 && mode!=2) return(1);
  if(m<1 || u==NULL || u_dim1<1 || cm==NULL) return(1);
  if(lpivot<0 || lpivot>=l1 || l1>m) return(1);

  /* Function Body */
  cl = fabs(u[lpivot*u_dim1]);
  // cl= (d1 = u[lpivot*u_dim1], fabs(d1));

  if(mode==2) { /* Apply transformation I+U*(U**T)/B to cm[] */
    if(cl<=0.) return(0);
  } else {   /* Construct the transformation */
  
    /* trying to compensate overflow */
    for(j=l1; j<m; j++) {  // Computing MAX 
      cl = fmax(fabs(u[j*u_dim1]), cl);
    }
    // zero vector?   
    if(cl<=0.) return(0);

    clinv=1.0/cl;
       
    // Computing 2nd power 
    d1=u[lpivot*u_dim1]*clinv; 
    sm=d1*d1;
    for(j=l1; j<m; j++) {
      d1=u[j*u_dim1]*clinv;
      sm+=d1*d1;
    }
    cl*=sqrt(sm);
    // cl = sqrt( (u[pivot]*clinv)^2 + sigma(i=l1..m)( (u[i]*clinv)^2 ) )
    if(u[lpivot*u_dim1] > 0.) cl=-cl;
    *up = u[lpivot*u_dim1] - cl; 
    u[lpivot*u_dim1]=cl;
  }

  // no vectors where to apply? only change pivot vector!	
  b=(*up)*u[lpivot*u_dim1];
  
  /* b must be nonpositive here; if b>=0., then return */
  if(b>=0.0) return(0); // was if(b==0) before 2013-06-22
  
  // ok, for all vectors we want to apply
  for(j=0; j<ncv; j++) {
    // take s = c[p,j]*h + sigma(i=l..m){ c[i,j] *v [ i ] }
    sm = cm[ lpivot*ice + j*icv ] * (*up);
    for(k=l1; k<m; k++) sm += cm[ k * ice + j*icv ] * u[ k*u_dim1 ]; 
    if(sm!=0.0) {
      sm *= (1.0/b); // was (1/b) before 2013-06-22
      // cm[lpivot, j] = ..
      cm[ lpivot * ice + j*icv] += sm*(*up);
      // for i = l1...m , set c[i,j] = c[i,j] + s*v[i]
      for(k=l1; k<m; k++) cm[ k*ice + j*icv] += u[k * u_dim1]*sm;
    }
  }
   
  return(0);
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn      NNLS_LSS_G1
  \param   a
  \param   b
  \param   cterm
  \param   sterm
  \param   sig: sig = sqrt(A**2+B**2)
  \brief   Compute orthogonal rotation matrix:
           (C, S) so that (C, S)(A) = (sqrt(A**2+B**2))
           (-S,C)         (-S,C)(B)   (   0          )
            sig is computed last to allow for the possibility that sig may be in
            the same location as A or B.
           Derived from Turku PET center libraries (authors: Vesa Oikonen and Kaisa Sederholm)
           This routine is based on the text and fortran code in
           C.L. Lawson and R.J. Hanson, Solving Least Squares Problems,
           Prentice-Hall, Englewood Cliffs, New Jersey, 1974.
*/
void vDynamicModel::NNLS_LSS_G1(HPFLTNB a, HPFLTNB b, HPFLTNB *cterm, HPFLTNB *sterm, HPFLTNB *sig)
{
  HPFLTNB d1, xr, yr;

  if(fabs(a)>fabs(b)) 
  {
    xr=b/a; d1=xr; yr=sqrt(d1*d1 + 1.); d1=1./yr;
    *cterm=(a>=0.0 ? fabs(d1) : -fabs(d1));
    *sterm=(*cterm)*xr; *sig=fabs(a)*yr;
  } 
  else if(b!=0.) 
  {
    xr=a/b; d1=xr; yr=sqrt(d1*d1 + 1.); d1=1./yr;
    *sterm=(b>=0.0 ? fabs(d1) : -fabs(d1));
    *cterm=(*sterm)*xr; *sig=fabs(b)*yr;
  } 
  else 
  {
    *sig=0.; *cterm=0.; *sterm=1.;
  }
}
