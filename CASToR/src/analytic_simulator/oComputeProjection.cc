
/*!
  \file
  \ingroup analytic_simulator

  \brief Implementation of class oComputeProjection
*/

#include "oComputeProjection.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \brief oComputeProjection constructor. 
  \param ap_ImageDimensionsAndQuantification
         Initialize the member variables to their default values.
*/
oComputeProjection::oComputeProjection(oImageDimensionsAndQuantification* ap_ImageDimensionsAndQuantification)
{
  mp_ID = ap_ImageDimensionsAndQuantification;
  m_noiseModelEnabled = false;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \brief oComputeProjection destructor. 
*/
oComputeProjection::~oComputeProjection()
{

}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
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
  \brief Perform the forward-projection, and send the result to the datafile for writing.
         Implements random Poisson noise.
         Write the result in the projection image matrix (only implemented for SPECT).
  \return 0 if success, positive value otherwise
*/
int oComputeProjection::DataUpdateStep(vDataFile* ap_DataFile, 
                                 oProjectionLine* a2p_Line,
                                     oImageSpace* ap_Image, 
                                          vEvent* ap_Event,
                                              int a_fr, 
                                              int a_rg, 
                                              int a_cg, 
                                              int th, 
                                         uint32_t a_timestamp,
                                             bool a_discardZeroEvent )
{

  #ifdef CASTOR_VERBOSE
  if(m_verbose >=4) Cout("oComputeProjection::DataUpdateStep ... " << endl);
  #endif
  
  FLTNB fp = 0., fp_atn = 0.;


  // Projection with attenuation for PET
  //TODO : add fonction in oProjectionLine for PET attenuation ?
  if(ap_DataFile->GetDataType() == TYPE_PET) 
  {
    // Forward project input image
    fp = a2p_Line->ForwardProject(ap_Image->m4p_forwardImage[a_fr][a_rg][a_cg]);
  
    // Forward project attenuation if atn map has been provided
    if(ap_Image->m4p_attenuation != NULL && ap_Image->m4p_attenuation[a_fr][a_rg][a_cg] != NULL)
      fp_atn = a2p_Line->ForwardProject(ap_Image->m4p_attenuation[a_fr][a_rg][a_cg]) * 0.1; // 0.1 -> convert in mm-1
    
    // Apply attenuation factor
    fp *= std::exp(-fp_atn);

    // Store Atn correction factor in Event if PET (not recorded in datafile if no atn map provided by the user)
    (dynamic_cast<iEventPET*>(ap_Event))->SetAttenuationCorrectionFactor(1/exp(-fp_atn));
  }
  // Project with attenuation for SPECT
  else if (ap_DataFile->GetDataType() == TYPE_SPECT)
  {
    if(ap_Image->m4p_attenuation != NULL) // Attenuation has been provided
      fp = a2p_Line->ForwardProjectWithSPECTAttenuation(ap_Image->m4p_attenuation[a_fr][a_rg][a_cg] ,
                                                        ap_Image->m4p_forwardImage[a_fr][a_rg][a_cg] );
    else // No attenuation
      fp = a2p_Line->ForwardProjectWithSPECTAttenuation(NULL ,
                                                        ap_Image->m4p_forwardImage[a_fr][a_rg][a_cg] );
  }
  
  #ifdef CASTOR_VERBOSE
  if(m_verbose >=4) Cout("oComputeProjection::DataUpdateStep : Projeted value = "<< fp << endl);
  #endif
  
  // Poisson noise
  if(m_noiseModelEnabled)
  {
    fp = GetPoissonNoise(fp);

    #ifdef CASTOR_VERBOSE
    if(m_verbose >=4) Cout("oComputeProjection::DataUpdateStep : Projeted value with Poisson noise = "<< fp << endl);
    #endif
  }
  
  // Write event in datafile
  ap_Event->SetEventValue(0, fp);
  ap_Event->SetTimeInMs(a_timestamp);
  
  // If we chose to discard zero event, write it only if fp > 0
  if(!a_discardZeroEvent 
  || fp>0.)
  {
    if(ap_DataFile->WriteEvent(ap_Event, th) )
    {
      Cerr("*****oComputeProjection::DataUpdateStep()-> Error while trying to write a projeted event" << endl);
      return 1;
    }
  }
  
  // Add the result to the projection image (only implemented for SPECT)
  if(ap_DataFile->GetDataType() == TYPE_SPECT) 
    ap_Image->m2p_projectionImage[ap_Event->GetID1(0)][ap_Event->GetID2(0)] += fp;

  return 0;
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn InitNoiseModel
  \param aNoiseModel : string corresponding to a noise model
  \brief This is a premature implementation of noise model initialization
         for analytic simulator.
         Currently, only the Poisson noise can be selected.
         It will be replaced with a more efficient system when a more
         advanced analytical simulator will be implemented
  \return 0 if success, positive value otherwise
*/
int oComputeProjection::InitNoiseModel(string aNoiseModel)
{
  if(m_verbose >=3) Cout("oComputeProjection::InitNoiseModel ... " << endl);
  
  // Get the noise model name in the options and isolate the real model's options

  // First check emptyness of the options
  if (aNoiseModel=="")
  {
    m_noiseModelEnabled = false;
    return 0;
  }
  else
  {
    string name_noise_model = "";
    string list_options_noise_model = "";
    string file_options_noise_model = "";
    
    // Search for a colon ":", this indicates that a configuration file is provided after the optimizer name
    size_t colon = aNoiseModel.find_first_of(":");
    size_t comma = aNoiseModel.find_first_of(",");
  
    // Case 1: we have a colon
    if (colon!=string::npos)
    {
      // Get the optimizer name before the colon
      name_noise_model = aNoiseModel.substr(0,colon);
      // Get the configuration file after the colon
      file_options_noise_model = aNoiseModel.substr(colon+1);
      // List of options is empty
      list_options_noise_model = "";
    }
    // Case 2: we have a comma
    else if (comma!=string::npos)
    {
      // Get the optimizer name before the first comma
      name_noise_model = aNoiseModel.substr(0,comma);
      // Get the list of options after the first comma
      list_options_noise_model = aNoiseModel.substr(comma+1);
      // Configuration file is empty
      file_options_noise_model = "";
    }
    // Case 3: no colon and no comma (a single model name)
    else
    {
      // Get the optimizer name
      name_noise_model = aNoiseModel;
      // Configuration file is empty
      file_options_noise_model = "";
      // List of options is empty
      list_options_noise_model = "";
    }
  
    // Check if the noise model is known
    if (aNoiseModel == "poisson")
    {
      m_noiseModelEnabled = true;
      return 0;
    }
    else
    {
      Cerr("*****oComputeProjection::InitNoiseModel()-> Error while trying to initialize noise model. Model '"<<aNoiseModel<<"' is unknown"<<endl;);
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
  \fn GetPoissonNoise
  \param a_lambda : event rate
  \brief Generate a Poisson random variable from a provided lambda
  \details Depending on lambda, use either :
           - Poisson noise if a_lambda <= 60
           - Normal approximation otherwise
  \return a random value
  \todo Check Implementation
  \todo Use C++11 Poisson noise distribution ?
*/
//uint32_t oComputeProjection::GetPoissonNoise(int32_t a_lambda)
uint32_t oComputeProjection::GetPoissonNoise(FLTNB a_lambda)
{
  #ifdef CASTOR_VERBOSE
  if(m_verbose >=4) Cout("oComputeProjection::GetPoissonNoise ... " << endl);
  #endif
  
  // Get instance of Random Number Generator
  sRandomNumberGenerator* p_RNG = sRandomNumberGenerator::GetInstance(); 
  
  HPFLTNB return_value = 0.;

  // Normal approximation for projection with high value
  if (a_lambda > 60.) 
  {     
    HPFLTNB rdm_1, rdm_2, w;
    
    do
    {
      rdm_1 = 2.0 * p_RNG->GenerateRdmNber() - 1.0;
      rdm_2 = 2.0 * p_RNG->GenerateRdmNber() - 1.0;
      w = rdm_1*rdm_1 + rdm_2*rdm_2;
    }  while (w > 1.0);

    w = sqrt( (-2.0 * log(w) ) / w );

    return_value = a_lambda + sqrt(a_lambda)*rdm_1*w;

    return_value = (return_value<0) ? 0 : round(return_value);
  }
  // Poisson noise
  else
  {
    HPFLTNB L = exp(-(a_lambda));
    int n = 0;
    HPFLTNB p = 1;
    
    do
    {
      n++;
      // Generate uniform random number u in [0,1] and let p ← p × u.
      p *= p_RNG->GenerateRdmNber();
      //cout << a_lambda << " ; " <<  L << " ; " << p <<  endl;
    } while (p > L);
    
    return_value = n-1;
  }

  return (uint32_t) return_value;
}
