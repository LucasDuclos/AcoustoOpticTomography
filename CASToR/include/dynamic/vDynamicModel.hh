
/*!
  \file
  \ingroup  dynamic
  \brief    Declaration of class vDynamicModel
*/

#ifndef VDYNAMICMODEL_HH
#define VDYNAMICMODEL_HH 1

#include "gVariables.hh"
#include "oImageDimensionsAndQuantification.hh"
#include "oOptimizerManager.hh"

class oImageSpace;

/*!
  \class   vDynamicModel
  \brief   This is the mother class of dynamic model classes
  \details This class is a virtual one, in the sense that it cannot be used on its own \n
           because several pure virtual functions belong to it. 
           Its children are implementations of actual dynamic models. \n
           Everywhere in the code, this parent class should be used instead of any of its children. \n
           It can be used during the reconstruction process by the oDynamicModelManager through the
           use of the EstimateModelParameters() and EstimateImageWithModel() functions
           
           All children must implement the following pure virtual functions: \n
            - ReadAndCheckConfigurationFile(): read specific options from a configuration file \n
            - ReadAndCheckOptionsList(): read specific options from a string \n
            - ShowHelp(): print helps about the projector specifications \n
            - Initialize(): initialize specific data of the projector (if required) \n
            - CheckParameters(): Check the initialization of the parameters (if required) \n
            
            - EstimateModelParameters() : (virtual only)  \n
                                          Estimate any temporal functions or coefficients  \n
                                          related to the dynamic model (if required) \n
            - EstimateImageWithModel(): Fit the dynamic model to the series of dynamic images
*/
class vDynamicModel
{
  // -----------------------------------------------------------------------------------------
  // Constructor & Destructor
  public:
    /*!
      \fn      vDynamicModel::vDynamicModel
      \brief   Constructor of vDynamicModel. Simply set all data members to default values.
    */
    vDynamicModel();
    /*!
      \fn      vDynamicModel::~vDynamicModel
      \brief   Destructor of vDynamicModel.
    */
    virtual ~vDynamicModel();


  // -----------------------------------------------------------------------------------------
  // Public member functions related to the initialization of the model
  public:
    /*!
      \fn      vDynamicModel::SetImageDimensionsAndQuantification
      \param   ap_ImageDimensionsAndQuantification
      \brief   Set the image dimensions in use
    */
    inline void SetImageDimensionsAndQuantification(oImageDimensionsAndQuantification* ap_ImageDimensionsAndQuantification)
                           {mp_ID = ap_ImageDimensionsAndQuantification;}
    /*!
      \fn      vDynamicModel::SetVerbose
      \param   a_verboseLevel
      \brief   Set the verbose level
    */
    inline void SetVerbose(int a_verbose) 
               {m_verbose = a_verbose;}
    /*!
      \fn      vDynamicModel::CheckParameters
      \brief   This function is used to check parameters after the latter
               have been all set using Set functions.
      \return  0 if success, positive value otherwise.
    */
    virtual int CheckParameters();
    /*!
      \fn      vDynamicModel::CheckSpecificParameters
      \brief   This function is used to check the parameters of the child functions before initialization if required.
      \details It could be overloaded by the child if needed. Default implementation is empty and return 0.
      \return  0 if success, other value otherwise.
    */
    virtual int CheckSpecificParameters() = 0;
    /*!
      \fn      vDynamicModel::ReadAndCheckConfigurationFile
      \param   const string& a_configurationFile : ASCII file containing informations about a dynamic model
      \brief   This function is used to read options from a configuration file. \n
               It looks for the parameters implemented by the mother class,
               such as 'No image update', 'No parameters update', or 'Save parametric images'
      \return  0 if success, other value otherwise.
    */
    virtual int ReadAndCheckConfigurationFile(string a_fileOptions);
    /*!
      \fn      vDynamicModel::ReadAndCheckConfigurationFileSpecific
      \brief   This function is used to read options from a configuration file. \n
               It is pure virtual so must be implemented by children.
      \return  0 if success, other value otherwise.
    */
    virtual int ReadAndCheckConfigurationFileSpecific() = 0;
    /*!
      \fn      vDynamicModel::ReadAndCheckOptionsList
      \param   const string& a_optionsList : a list of parameters separated by commas
      \brief   This function is used to read parameters from a string. \n
               It is pure virtual so must be implemented by children.
      \return  0 if success, other value otherwise.
    */
    virtual int ReadAndCheckOptionsList(string a_listOptions) = 0;
    /*!
      \fn      vDynamicModel::Initialize
      \brief   A public function used to initialize the dynamic model
      \details This function does not take any parameter and is used to initialize everything that
               should be initialized. At the end, it calls the pure virtual InitializeSpecific()
               function implemented by children.
      \return  0 if success, other value otherwise.
    */
    virtual int Initialize();
    /*!
      \fn      vDynamicModel::InitializeSpecific() = 0
      \brief   A private function used to initialize everything specific to the child model
      \details This function is used to initialize everything specific to the model that should be
               initialized. It is called by the Initialize() function. It is pure virtual so is
               implemented by children.
      \return  An integer reflecting the initialization status; 0 if no problem, another value
               otherwise.
    */
    virtual int InitializeSpecific() = 0;
    /*!
      \fn      vDynamicModel::ShowHelp
      \brief   This function is used to print out specific help about the deformation and its options. It is
               pure virtual so must be implemented by children.
    */
    virtual void ShowHelp() = 0;


  // -----------------------------------------------------------------------------------------
  // Public member functions called by the main iterative algorithm class
  public:
    /*!
      \fn      vDynamicModel::EstimateModel
      \param   ap_ImageS : pointer to the ImageSpace
      \param   a_ite : index of the actual iteration
      \param   a_sset : index of the actual subset
      \brief   This function checks if the EstimateModelParameters() function (specific to each model)
               must be called at this stage of the reconstruction depending on the m_xxxUpdateflags.
      \return  0 if success, other value otherwise.
    */
    virtual int EstimateModel(oImageSpace* ap_Image, int a_ite, int a_sset);
    /*!
      \fn      vDynamicModel::EstimateModelParameters
      \param   ap_ImageS : pointer to the ImageSpace
      \param   a_ite : index of the actual iteration
      \param   a_sset : index of the actual subset
      \brief   This function is pure virtual so must be implemented by children. \n
               It can be used to estimate any temporal functions or coefficients 
               related to the dynamic model, if required
      \return  0 if success, other value otherwise.
    */
    virtual int EstimateModelParameters(oImageSpace* ap_Image, int a_ite, int a_sset) = 0;
    /*!
      \fn      vDynamicModel::EstimateImageWithModel
      \param   ap_ImageS : pointer to the ImageSpace
      \param   a_ite : index of the actual iteration
      \param   a_sset : index of the actual subset
      \brief   This function checks if the EstimateImageWithModel() function (specific to each model)
               must be called at this stage of the reconstruction depending on the m_xxxUpdateflags.
      \return  0 if success, other value otherwise.
    */
    virtual int EstimateImage(oImageSpace* ap_Image, int a_ite, int a_sset);
    /*!
      \fn      vDynamicModel::EstimateImageWithModel
      \param   ap_ImageS : pointer to the ImageSpace
      \param   a_ite : index of the actual iteration
      \param   a_sset : index of the actual subset
      \brief   This function is pure virtual so must be implemented by children. \n
               It is used to fit the dynamic model to the series of dynamic images
      \return  0 if success, other value otherwise.
    */
    virtual int EstimateImageWithModel(oImageSpace* ap_Image, int a_ite, int a_sset) = 0;
    /*!
      \fn      SaveParametricImages
      \param   a_iteration : current iteration index
      \param   a_subset : current number of subsets (or -1 by default)
      \brief   This function is virtual it can be overloaded by children 
               if required
      \return  0 if success, positive value otherwise
    */
    virtual int SaveParametricImages(int a_iteration, int a_subset = -1);
    /*!
      \fn      vDynamicModel::ApplyOutputFOVMaskingOnParametricImages
      \brief   Mask the outside of the transaxial FOV based on the m_fovOutPercent
      \details Similar to the eponym function in ImageSpace, but on parametric images
    */
    virtual int ApplyOutputFOVMaskingOnParametricImages();
    /*!
      \fn      vDynamicModel::ComputeOutputParImage
      \brief   Compute output image using the m2p_parametricImages matrix Store the result in the m2p_outputParImages matrix
    */
    virtual void ComputeOutputParImage();

    /*!
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
    int NNLS( HPFLTNB **A,
              int m,
              int n,
              HPFLTNB *B,
              HPFLTNB *X,
              HPFLTNB *rnorm,
              HPFLTNB *wp,
              HPFLTNB *zzp,
              int *indexp );


  private:
  
    /*!
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
      \brief   This function is used by the NNLS function()
               Construction and/or application of a single Householder transformation:
               Q = I + U*(U**T)/B
               Derived from Turku PET center libraries (authors: Vesa Oikonen and Kaisa Sederholm)
               This routine is based on the text and fortran code in
               C.L. Lawson and R.J. Hanson, Solving Least Squares Problems,
               Prentice-Hall, Englewood Cliffs, New Jersey, 1974.
      \return  0 if success, positive value otherwise (erroneous parameters)
    */
    int NNLS_LSS_H12( int mode,
                      int lpivot,
                      int l1,
                      int m,
                      HPFLTNB *u,
                      int u_dim1,
                      HPFLTNB *up,
                      HPFLTNB *cm,
                      int ice,
                      int icv,
                      int ncv );

    /*!
      \fn      NNLS_LSS_G1
      \param   a
      \param   b
      \param   cterm
      \param   sterm
      \param   sig: sig = sqrt(A**2+B**2)
      \brief   This function is used by the NNLS function()
               Compute orthogonal rotation matrix:
               (C, S) so that (C, S)(A) = (sqrt(A**2+B**2))
               (-S,C)         (-S,C)(B)   (   0          )
                sig is computed last to allow for the possibility that sig may be in
                the same location as A or B.
               Derived from Turku PET center libraries (authors: Vesa Oikonen and Kaisa Sederholm)
               This routine is based on the text and fortran code in
               C.L. Lawson and R.J. Hanson, Solving Least Squares Problems,
               Prentice-Hall, Englewood Cliffs, New Jersey, 1974.
    */
    void NNLS_LSS_G1(HPFLTNB a, HPFLTNB b, HPFLTNB *cterm, HPFLTNB *sterm, HPFLTNB *sig);


  // -----------------------------------------------------------------------------------------
  // Data members
  protected:
    oImageDimensionsAndQuantification* mp_ID; /*!< Pointer to the oImageDimensionsAndQuantification object in use */
    int m_verbose;                            /*!< The verbose level */
    int m_nbTimeBF;                           /*!< Number of time basis functions in the model */
    int m_nbModelParam;                       /*!< Number of model parameters */

    FLTNB** m2p_parametricImages; /*!< Image matrix containing the parametric images \n
                                       2 pointers:  \n
                                       1: Parametric image related to the dynamic model basis functions. \n
                                       2: 3D voxels */
                                       
    HPFLTNB** m2p_modelTACs;       /*!< Vector containing the Model temporal basis functions \n
                                       2 pointers:  \n
                                       1: index of the temporal function \n
                                       2: coefficient of the functions for each time points of a dynamic acquisition */

    FLTNB** m2p_outputParImages; /*!< Image matrix to gather the parametric image before writing on disk \n
                                      By default it will point directly to the parametric m2p_parametricImages. \n
                                      They are allocated if post-processing are enabled before writing the image (i.e FOV masking) \n
                                       2 pointers:  \n
                                       1: Parametric image related to the dynamic model basis functions. \n
                                       2: 3D voxels */

    string      m_fileOptions;            /*!<Path to a configuration file */
    string      m_listOptions;            /*!<String containing a list of options */
    
    bool        m_checked;                /*!< Boolean indicating whether the parameters were checked or not */
    bool        m_initialized;            /*!< Boolean indicating whether the manager was initialized or not */
    bool        m_saveParImageFlag;       /*!< Flag indicating if parametric images should be written on disk (default=true) */
    
    bool        m_noImageUpdateFlag;      /*!< If true, the reconstructed images are not estimated from the parametric images, the EstimateImageWithModel() functions is not called
                                               so the class only estimate parametric images from each current estimation of the images (default=false) */
                                           
               
    bool        m_noParametersUpdateFlag; /*!< If true, the parameters are not estimated from the serie of dynamic images, the EstimateModelParameters() functions is not called (default=false) */
                                           
    int         m_startIteUpdateFlag;     /*!< Number of iterations after which the reconstructed images are estimated from the parametric images. (default or negative value =0) */
                                  
    // NNLS function
    HPFLTNB*    mp_w;             /*!<Vector containing the weights for NNLS estimation */
    HPFLTNB***  m3p_nnlsA;        /*!<2D coefficient matrix for NNLS estimation (multithreaded)*/
    HPFLTNB**   m2p_nnlsB;        /*!<1D vector for NNLS estimation, containing the solution (multithreaded)*/
    HPFLTNB**   m2p_nnlsMat;      /*!<1D working vector for NNLS estimation (multithreaded)*/
    uint16_t    m_nnlsN=0;        /*!<Number of parameters in NNLS estimation */
};

// ----------------------------------------------------------------------
// Part of code that manages the auto declaration of children classes
// ----------------------------------------------------------------------

// Macro for the function that creates the object
#define FUNCTION_DYNAMICMODEL(CLASS) \
  static vDynamicModel *make_dynamic_model() { return new CLASS(); };

// Macro for the class that links the appropriate function to the map of objects
#define CLASS_DYNAMICMODEL(NAME,CLASS)                                                               \
  class NAME##DynamicModelCreator                                                                    \
  {                                                                                                  \
    public:                                                                                          \
      NAME##DynamicModelCreator()                                                                    \
        { sAddonManager::GetInstance()->mp_listOfDynamicModels[#NAME] = CLASS::make_dynamic_model; } \
  };                                                                                                 \
  static NAME##DynamicModelCreator DynamicModelCreator##NAME;

#endif
