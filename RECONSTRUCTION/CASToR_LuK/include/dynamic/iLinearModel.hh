
/*!
  \file
  \ingroup  dynamic
  \brief    Declaration of class iLinearModel
*/

#ifndef ILINEARMODEL_HH
#define ILINEARMODEL_HH 1



#include "vDynamicModel.hh"
#include "sAddonManager.hh"

/*!
  \class   iLinearModel
  \brief   This class implements a general linear dynamic model applied between the images of a dynamic acquisition \n
           The model is applied on a voxel-by-voxel basis between the images of the frames and/or respiratory/cardiac gates
*/
class iLinearModel : public vDynamicModel
{
  // -----------------------------------------------------------------------------------------
  // Constructor & Destructor
  public:
    /*!
      \fn      iLinearModel::iLinearModel
      \brief   Constructor of iLinearModel. Simply set all data members to default values.
    */
    iLinearModel();
    /*!
      \fn      iLinearModel::~iLinearModel
      \brief   Destructor of iLinearModel
    */
    ~iLinearModel();


  // -----------------------------------------------------------------------------------------
  // Public member functions related to the initialization of the model
  public:
    // Function for automatic insertion (put the class name as the parameters and do not add semi-colon at the end of the line)
    FUNCTION_DYNAMICMODEL(iLinearModel)
    /*!
      \fn      iLinearModel::CheckSpecificParameters
      \brief   This function is used to check whether all member variables
               have been correctly initialized or not.
      \return  0 if success, positive value otherwise.
    */
    int CheckSpecificParameters();
    /*!
      \fn      iLinearModel::ReadAndCheckConfigurationFileSpecific
      \brief   This function is used to read options from a configuration file.
      \return  0 if success, other value otherwise.
    */
    int ReadAndCheckConfigurationFileSpecific();
    /*!
      \fn      iLinearModel::ReadAndCheckOptionsList
      \param   const string& a_optionsList : a list of parameters separated by commas
      \brief   This function is used to read parameters from a string.
      \return  0 if success, other value otherwise.
    */
    int ReadAndCheckOptionsList(string a_listOptions);
    /*!
      \fn      iLinearModel::InitializeSpecific
      \brief   This function is used to initialize the parametric images and basis functions
      \todo    Read Interfile for parametric images initialization
      \return  0 if success, other value otherwise.
    */
    int InitializeSpecific();
    /*!
      \fn      iLinearModel::ShowHelp
      \brief Print out specific help about the implementation 
             of this model and its initialization
    */
    void ShowHelp();


  // -----------------------------------------------------------------------------------------
  // Public member functions called by the main iterative algorithm class
    /*!
      \fn      iLinearModel::EstimateModelParameters
      \param   ap_ImageS : pointer to the ImageSpace
      \param   a_ite : index of the actual iteration (not used)
      \param   a_sset : index of the actual subset (not used)
      \brief Estimate model parameters (parametric images and basis functions)
      \return  0 if success, other value otherwise.
    */
    int EstimateModelParameters(oImageSpace* ap_Image, int a_ite, int a_sset);
    /*!
      \fn      iLinearModel::EstimateImageWithModel
      \param   ap_ImageS : pointer to the ImageSpace
      \param   a_ite : index of the actual iteration (not used)
      \param   a_sset : index of the actual subset (not used)
      \brief   Re-estimate image using the linear parametric images and basis functions
      \return  0 if success, other value otherwise.
    */
    int EstimateImageWithModel(oImageSpace* ap_Image, int a_ite, int a_sset);
    /*
      \fn NestedEM
      \param ap_ImageS : pointer to the ImageSpace
      \param a_ite : index of the actual iteration (not used)
      \brief Estimate parametric images and basis functions (if enabled) using the nested EM method
      \return 0 if success, other value otherwise.
    */
    int NestedEM(oImageSpace* ap_ImageS, int a_ite);
    /*!
      \fn      SaveParametricImages
      \param   a_iteration : current iteration index
      \param   a_subset : current number of subsets (or -1 by default)
      \brief   Save the parametric images of all dynamic models
               This function is virtual it can be overloaded by children 
               if required
      \return  0 if success, positive value otherwise
    */
    virtual int SaveParametricImages(int a_iteration, int a_subset = -1);

  // -----------------------------------------------------------------------------------------
  // Data members
  protected:

    int m_nbRgateBF;                /*!< Number of time basis functions applied to respiratory gates in the model */
    int m_nbCgateBF;                /*!< Number of time basis functions applied to cardiac gates in the model */
    int m_nbRGModelParam;           /*!< Number of model parameters applied to respiratory gates */
    int m_nbCGModelParam;           /*!< Number of model parameters applied to cardiac gates */

    FLTNB** m2p_RGParametricImages; /*!< Image matrix containing the parametric images of the respiratory gating model\n
                                       2 pointers:  \n
                                       1: Parametric image related to the respiratory gating model basis functions. \n
                                       2: 3D voxels */
    FLTNB** m2p_CGParametricImages; /*!< Image matrix containing the parametric images of the cardiac gating model\n
                                       2 pointers:  \n
                                       1: Parametric image related to the cardiac gating model basis functions. \n
                                       2: 3D voxels */
                                       
    HPFLTNB** m2p_RGModelTACs;       /*!< Vector containing the respiratory gating model temporal basis functions \n
                                       2 pointers:  \n
                                       1: index of the temporal function \n
                                       2: coefficient of the functions for each respiratory gate */
    HPFLTNB** m2p_CGModelTACs;       /*!< Vector containing the cardiac gating model temporal basis functions \n
                                       2 pointers:  \n
                                       1: index of the temporal function \n
                                       2: coefficient of the functions for each cardiac gate */

    FLTNB*   mp_corrBasisCoeffs;      /*!< Image matrix containing correction factors for the parametric images updates */
    HPFLTNB* mp_corrBasisFunctions;   /*!< Vector containing correction factors for the temporal basis functions updates */
    uint32_t m_nbLinearModelCycles;   /*!< Number of iteration of the model (one cycle consists in several updates of either the parametric images or the basis function) */
    int m_basisFunctionsUpdStartIte;  /*!< Starting iteration for the update of basis functions. \n
                                           If negative, no update of the basis functions is performed (only parametric images are updated) */
    int m_basisFunctionsUpdRatio;     /*!< Ratio for the parametric images/basis functions updates cycle. \n
                                           Cycles consist in 'm_basisFunctionsUpdRate' iterations of the parametric images, following by 'm_basisFunctionsUpdRate' iterations of the basis functions */
    int m_basisFunctionsUpdIdx;       /*!< Index to compute the ratio for the parametric images/basis functions updates cycle */

};

// Class for automatic insertion (set here the visible dynamic model's name, put the class name as the parameters and do not add semi-colon at the end of the line)
CLASS_DYNAMICMODEL(LinearModel,iLinearModel)

#endif
