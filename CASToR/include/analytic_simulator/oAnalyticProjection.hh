
/*!
  \file
  \ingroup  analytic_simulator
  \brief    Declaration of class oAnalyticProjection
*/

#ifndef OANALYTICPROJECTION_HH
#define OANALYTICPROJECTION_HH 1

#include "oImageDimensionsAndQuantification.hh"
#include "oProjectorManager.hh"
#include "oImageConvolverManager.hh"
#include "oImageSpace.hh"
#include "oComputeProjection.hh"
#include "vScanner.hh"


/*!
  \class oAnalyticProjection
  \brief   This class manages the analytic projection of an image and the computation of the associated datafile. \n
  \details It uses the following components : \n
           - vDataFile \n
           - oProjector \n
           - a specific class for the data update step (ComputeProjection) \n
           - oImageSpace.
*/
class oAnalyticProjection
{
  // Constructor & Destructor
  public:

  /*!
    \brief oAnalyticProjection constructor.
           Initialize the member variables to their default values.
  */
    oAnalyticProjection();

  /*!
    \brief oAnalyticProjection destructor. 
  */
    ~oAnalyticProjection();


  // -------------------------------------------------------------------
  // Public member functions
  public:
    /*!
      \fn Launch
      \brief Just call either the LaunchCPU or the LaunchGPU function as asked for
      \return 0 if success, positive value otherwise
    */
    int Launch();
    int LaunchCPU();
    
    #ifdef CASTOR_GPU
    int LaunchGPU();
    #endif

    void InitOptimizer(oImageDimensionsAndQuantification* ap_ImageDimensionsAndQuantification)
                       {mp_ComputeProjection = new oComputeProjection(ap_ImageDimensionsAndQuantification);};

    int InitNoiseModel(string aNoiseModel)
                       {return mp_ComputeProjection->InitNoiseModel(aNoiseModel);};
                       
    /*!
      \fn SetImageDimensionsAndQuantification
      \param ap_ImageDimensionsAndQuantification
      \brief Set the Image Dimensions and Quantification Object 
    */
    inline void SetImageDimensionsAndQuantification(oImageDimensionsAndQuantification* ap_ImageDimensionsAndQuantification) {mp_ID = ap_ImageDimensionsAndQuantification;};

    /*!
      \fn SetImageSpace
      \param ap_ImageSpace
      \brief Set the Image Space Object 
    */
    inline void SetImageSpace(oImageSpace* ap_ImageSpace) {mp_ImageSpace = ap_ImageSpace;};
    
    /*!
      \fn SetProjectorManager
      \param ap_ProjectorManager
      \brief Set the Projector Manager Object 
    */
    inline void SetProjectorManager(oProjectorManager* ap_ProjectorManager) {mp_ProjectorManager = ap_ProjectorManager;};

    /*!
      \fn SetImageConvolverManager
      \param ap_ImageConvolverManager
      \brief Set the Image Convolver Manager Object 
    */
    inline void SetImageConvolverManager(oImageConvolverManager* ap_ImageConvolverManager) {mp_ImageConvolverManager = ap_ImageConvolverManager;}
    
    /*!
      \fn SetDataFile
      \param a2p_DataFile
      \brief Set the list of DataFile
    */
    inline void SetDataFile(vDataFile** a2p_DataFile) {m2p_DataFile = a2p_DataFile;};
    
    /*!
      \fn SetGPUflag
      \param a_flagGPU
      \brief Set the GPU flag
    */
    inline void SetGPUflag(bool a_flagGPU) {m_flagGPU = a_flagGPU;};
    
    /*!
      \fn SetVerbose
      \param a_verboseLevel
      \brief Set Verbosity
    */
    inline void SetVerbose(int a_verboseLevel) {m_verbose = a_verboseLevel;};
    
    /*!
      \fn SetNbBeds
      \param a_nbBeds
      \brief Set number of beds (bed positions)
    */
    inline void SetNbBeds(int a_nbBeds) {m_nbBeds = a_nbBeds;};
    
    /*!
      \fn SetPathInitImage
      \param a_pathToInitialImage
      \brief Set path to an initial image
    */
    inline void SetPathInitImage(string a_pathToInitialImage) {m_pathToInitialImg = a_pathToInitialImage;};

    /*!
      \fn SetPathAtnImage
      \param a_pathToAtnImage
      \brief Set path to an attenuation image
    */
    inline void SetPathAtnImage(string a_pathToAtnImg) {m_pathToAtnImg = a_pathToAtnImg;};
    
    /*!
      \fn SetScanner
      \param  ap_Scanner
      \brief Set the scanner in use
    */
    inline void SetScanner(vScanner* ap_Scanner) {mp_Scanner = ap_Scanner;}

    /*!
      \fn SetNoZeroEvent
      \param  a_flag : true / false
      \brief Set the flag to keep/discard zero events
    */
    inline void SetNoZeroEvent(bool a_flag) {m_discardZeroEvent = a_flag;}


  // -------------------------------------------------------------------
  // Private member functions
  private:

  // Data members
  private:
    int m_verbose;                                    /*!< Verbosity */  
    bool m_flagGPU;                                   /*!< Do we use GPU or not (default=false) */
    bool m_discardZeroEvent;                          /*!< Do not save zero event (default=false) */  
    oImageDimensionsAndQuantification* mp_ID;         /*!< Pointer to the oImageDimensionsAndQuantification object */
    vDataFile** m2p_DataFile;                         /*!< Pointer to the array of vDataFile object */
    oProjectorManager* mp_ProjectorManager;           /*!< Pointer to the Projector Manager object */
    oImageConvolverManager* mp_ImageConvolverManager; /*!< Pointer to the Image Convolver object */
    oImageSpace* mp_ImageSpace;                       /*!< Pointer to the Image Space object */
    oComputeProjection* mp_ComputeProjection;         /*!< Pointer to the object which manages the data update step for analytic projection*/
    int m_nbBeds;                                     /*!< number of bed FOVs (1 datafile by bed) */  
    string m_pathToInitialImg;                        /*!< String containing the path to an initialization image */ 
    string m_pathToAtnImg;                            /*!< String containing the path to an attenuation image */ 
    vScanner* mp_Scanner;                             /*!< Pointer to the Scanner object */
};

#endif













