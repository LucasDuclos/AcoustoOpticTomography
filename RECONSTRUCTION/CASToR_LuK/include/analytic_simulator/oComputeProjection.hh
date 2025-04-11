
/*!
  \file
  \ingroup  analytic_simulator
  \brief    Declaration of class oComputeProjection
*/

#ifndef OCOMPUTEPROJECTION_HH
#define OCOMPUTEPROJECTION_HH 1

#include "oProjectionLine.hh"
#include "oImageSpace.hh"
#include "vEvent.hh"


/*!
  \class   oComputeProjection
  \brief   Class that manages the data update step of analytic projection.
  \details The DataUpdateStep computes the data update step and manages Poisson RNG \n
           For SPECT, the image of the projection is written in ImageSpace->m2p_projectionImage \n
           This implementation will be subjected to major modifications with the next update
           of the analytic simulator tools.
*/
class oComputeProjection
{
  // Constructor & Destructor
  public:
  
  /*!
    \brief oComputeProjection constructor. 
    \param ap_ImageDimensionsAndQuantification
           Initialize the member variables to their default values.
  */
  oComputeProjection(oImageDimensionsAndQuantification* ap_ImageDimensionsAndQuantification);

  /*!
    \brief oComputeProjection destructor. 
  */
  ~oComputeProjection();


  // -------------------------------------------------------------------
  // Public member functions
  public:

  /*!
    \fn DataUpdateStep
    \param ap_DataFile : pointer to the datafile which manage output data writing
    \param a2p_Line : pointer to the structure containing system matrix components
    \param ap_Image : pointer to the image space to access forward/attenuation/projection images
    \param ap_Event : pointer to the structure containing the event geometric indices
    \param a_fr : frame index
    \param a_rg : respiratory gate index
    \param a_cg : cardiac gate index
    \param a_timestamp : event timestamp
    \param a_discardZeroEvent : if true, discard zero event
    \brief Perform the forward-projection, and send the result to the datafile for writing. \n
           Implements random Poisson noise. \n
           Write the result in the projection image matrix (only implemented for SPECT).
    \return 0 if success, positive value otherwise
  */
  int DataUpdateStep(vDataFile* ap_DataFile, 
               oProjectionLine* a2p_Line, 
                   oImageSpace* ap_Image, 
                        vEvent* ap_Event,
                            int a_fr, 
                            int a_rg, 
                            int a_cg, 
                            int th, 
                       uint32_t a_timestamp,
                           bool a_discardZeroEvent );

  
  /*!
    \fn InitNoiseModel
    \param aNoiseModel : string corresponding to a noise model
    \brief This is a premature implementation of noise model initialization
           for analytic simulator. \n
           Currently, only the Poisson noise can be selected.
    \return 0 if success, positive value otherwise
  */
  int InitNoiseModel(string aNoiseModel);
  
  /*!
    \fn GetPoissonNoise
    \param a_lambda : event rate
    \brief Generate a Poisson random variable from a provided lambda
    \details Depending on lambda, use either : \n
             - Poisson noise if a_lambda <= 60 \n
             - Normal approximation otherwise
    \return a random value
    \todo Implementation not checked
    \todo Use C++11 Poisson noise distribution ?
  */
  uint32_t GetPoissonNoise(FLTNB a_lambda);

  /*!
    \fn SetVerbose
    \param a_verboseLevel
    \brief set verbosity
  */
  void SetVerbose(int a_verboseLevel) {m_verbose = a_verboseLevel;};


  // -------------------------------------------------------------------
  // Private member functions
  private:


  // -------------------------------------------------------------------
  // Data members
  private:
    oImageDimensionsAndQuantification* mp_ID; /*!< Pointer to the oImageDimensionsAndQuantification object */
    int m_verbose;                            /*!< Verbosity (default=-1)*/
    bool m_noiseModelEnabled;                 /*!< Poisson noise enabled or not (default=false) */
};

#endif

