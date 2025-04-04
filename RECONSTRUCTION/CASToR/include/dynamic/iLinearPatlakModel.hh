
/*!
  \file
  \ingroup  dynamic
  \brief    Declaration of class iLinearPatlakModel
*/

#ifndef ILINEARPATLAKMODEL_HH
#define ILINEARPATLAKMODEL_HH 1


// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/**
 * @defgroup PATLAK_METHOD Patlak parameters estimation method
 *
 *    \brief Keywords corresponding to the method enabled to estimate Patlak parameters \n
 *           Defined in iPatlakModel.hh
 * @{
 */

/** Constant corresponding to the least-square method (=0) */
#define PATLAK_METHOD_LS 0
/** Constant corresponding to the non-negative least-square method (=1) */
#define PATLAK_METHOD_NNLS 1
/** Constant corresponding to the NESTED EM method (=2) */
#define PATLAK_METHOD_NESTED_EM 2
/** @} */




#include "vDynamicModel.hh"
#include "iLinearModel.hh"
#include "sAddonManager.hh"

/*!
  \class   iLinearPatlakModel
  \brief   This class implements the Patlak model, to model kinetics of irreversible radiotracers
*/
class iLinearPatlakModel : public iLinearModel
{
  // -----------------------------------------------------------------------------------------
  // Constructor & Destructor
  public:
    /*!
      \fn      iLinearPatlakModel::iLinearPatlakModel
      \brief   Constructor of iLinearPatlakModel. Simply set all data members to default values.
    */
    iLinearPatlakModel();
    /*!
      \fn      iLinearPatlakModel::~iLinearPatlakModel
      \brief   Destructor of iLinearPatlakModel
    */
    ~iLinearPatlakModel();


  // -----------------------------------------------------------------------------------------
  // Public member functions related to the initialization of the model
  public:
    // Function for automatic insertion (put the class name as the parameters and do not add semi-colon at the end of the line)
    FUNCTION_DYNAMICMODEL(iLinearPatlakModel)
    /*!
      \fn      iLinearPatlakModel::CheckSpecificParameters
      \brief   This function is used to check whether all member variables
               have been correctly initialized or not.
      \return  0 if success, positive value otherwise.
    */
    int CheckSpecificParameters();
    /*!
      \fn      iLinearPatlakModel::ReadAndCheckConfigurationFileSpecific
      \brief   This function is used to read options from a configuration file.
      \return  0 if success, other value otherwise.
    */
    int ReadAndCheckConfigurationFileSpecific();
    /*!
      \fn      iLinearPatlakModel::ReadAndCheckOptionsList
      \param   const string& a_optionsList : a list of parameters separated by commas
      \brief   This function is used to read parameters from a string.
      \return  0 if success, other value otherwise.
    */
    int ReadAndCheckOptionsList(string a_listOptions);
    /*!
      \fn      iLinearPatlakModel::InitializeSpecific
      \brief   This function is used to initialize Patlak parametric images and basis functions
      \todo    Read Interfile for parametric images initialization
      \return  0 if success, other value otherwise.
    */
    int InitializeSpecific();
    /*!
      \fn      iLinearPatlakModel::ShowHelp
      \brief   Print out specific help about the implementation of the Patlak
               model and its initialization
    */
    void ShowHelp();


  // -----------------------------------------------------------------------------------------
  // Public member functions called by the main iterative algorithm class
    /*!
      \fn      iLinearPatlakModel::EstimateModelParameters
      \param   ap_ImageS : pointer to the ImageSpace
      \param   a_ite : index of the actual iteration (not used)
      \param   a_sset : index of the actual subset (not used)
      \brief   Estimate Patlak parametric images
      \return  0 if success, other value otherwise.
    */
    int EstimateModelParameters(oImageSpace* ap_Image, int a_ite, int a_sset);

    /*
      \fn Patlak_NNLS
      \param ap_ImageS : pointer to the ImageSpace
      \param a_ite : index of the actual iteration (not used)
      \brief Estimate Patlak parametric images using the NNLS method
      \return 0 if success, other value otherwise.
    */
    int Patlak_NNLS(oImageSpace* ap_ImageS, int a_ite);

  
    /*!
      \fn Patlak_LS
      \param ap_ImageS : pointer to the ImageSpace
      \param a_ite : index of the actual iteration (not used)
      \brief Estimate Patlak parametric images using the voxelwise least square method
      \return 0 if success, other value otherwise.
    */
    int Patlak_LS(oImageSpace* ap_ImageS, int a_ite);

  // -----------------------------------------------------------------------------------------
  // Data members
  protected:

    int  m_PatlakMethodFlag;      /*!<Flag indicating the method to estimate Patlak parameters. \n
                                      0 = Least Square
                                      1 = Non negative least-square
                                      2 = POSEM method */
                                      
                                      

};

// Class for automatic insertion (set here the visible dynamic model's name, put the class name as the parameters and do not add semi-colon at the end of the line)
CLASS_DYNAMICMODEL(Patlak,iLinearPatlakModel)

#endif
