
/*!
  \file
  \ingroup analytic_simulator

  \brief Implementation of class oAnalyticProjection
*/

#include "gVariables.hh"
#include "oAnalyticProjection.hh"

/*
  \brief oAnalyticProjection constructor.
         Initialize the member variables to their default values.
*/
oAnalyticProjection::oAnalyticProjection()
{
  m_verbose = -1;
  m_flagGPU = false;
  m_discardZeroEvent = false;
  mp_ID = NULL;
  m2p_DataFile = NULL;
  mp_ProjectorManager = NULL;
  mp_ImageConvolverManager = NULL;
  mp_ImageSpace = NULL;
  mp_ComputeProjection = NULL;
  m_nbBeds = -1;
  m_pathToInitialImg = "";
  m_pathToAtnImg = "";
  mp_Scanner = NULL;
}



/*
  \brief oAnalyticProjection destructor. 
*/
oAnalyticProjection::~oAnalyticProjection()
{}



/*
  \fn Launch
  \brief Just call either the LaunchCPU or the LaunchGPU function as asked for
  \return 0 if success, positive value otherwise
*/
int oAnalyticProjection::Launch()
{
  #ifdef CASTOR_GPU
  if (m_flagGPU) return LaunchGPU();
  else return LaunchCPU();
  #else
  return LaunchCPU();
  #endif
}




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn LaunchGPU
  \brief Perform the projection by executing each possible LOR of the scanner,
         call the different object for optimization, projection, update, etc.
         Function designed to be executed on the GPU only.
         Returns error by default as it is not implemented
  \return 0 if success, positive value otherwise
*/
#ifdef CASTOR_GPU
int oAnalyticProjection::LaunchGPU()
{
  Cerr("***** oAnalyticProjection::LaunchGPU() -> The GPU specific function is not implemented !" << endl);
  return 1;
}
#endif




// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
/*
  \fn LaunchCPU
  \brief Perform the projection by executing each possible LOR of the scanner,
         call the different object for optimization, projection, update, etc.
         Function designed to be executed on the GPU only.
         Returns error by default as it is not implemented
  \return 0 if success, positive value otherwise
*/
int oAnalyticProjection::LaunchCPU()
{
  // Verbose
  if (m_verbose>=1) Cout("oAnalyticProjection::LaunchCPU() -> Start analytic projection" << endl);
    
  // Spread verbosity
  mp_ComputeProjection->SetVerbose(m_verbose);

  // Image Space allocations
  mp_ImageSpace->LMS_InstantiateImage();
  mp_ImageSpace->LMS_InstantiateForwardImage();

  // Instanciate and initialize projection image for SPECT
  if(mp_Scanner->GetScannerType() == SCANNER_SPECT_CONVERGENT) 
    mp_ImageSpace->PROJ_InstantiateProjectionImage(mp_Scanner->PROJ_GetSPECTNbProjections() , mp_Scanner->PROJ_GetSPECTNbPixels());
  
  // Initialize the image to project
  if(mp_ImageSpace->PROJ_InitImage(m_pathToInitialImg) )
  {
    Cerr("***** oAnalyticProjection::LaunchCPU()-> Error during image initialization !" << endl);  
    return 1;
  }

  if(mp_ImageSpace->InitAttenuationImage(m_pathToAtnImg) )
  {
    Cerr("***** oAnalyticProjection::LaunchCPU()-> Error during attenuation image initialization !" << endl);  
    return 1;
  }

  // Copy current image in forward-image buffer
  mp_ImageSpace->LMS_PrepareForwardImage();

  if (mp_ImageConvolverManager->ConvolveForward(mp_ImageSpace))
  {
    Cerr("***** oAnalyticProjection::LaunchCPU() -> A problem occurred while applying image convolver to forward image !" << endl);
    return 1;
  }
  
  
  // Initial clock
  clock_t clock_start = clock();
  time_t time_start = time(NULL);
  
  sScannerManager* p_scannerManager; 
  p_scannerManager = sScannerManager::GetInstance();

  for(int bed=0 ; bed<m_nbBeds ; bed++)
  {
    // Initialize main loop start and stop values
    unsigned int main_loop_start_index = 0 ;
    unsigned int main_loop_stop_index = 0;
    
    // Check beforehand any issue with the loop start/stop values 
    // (not possible to do this check in the inner multithreaded loop)
    if(p_scannerManager->PROJ_GetModalityStopValueMainLoop() > 0)
      main_loop_stop_index = p_scannerManager->PROJ_GetModalityStopValueMainLoop();
    else
    {
      Cerr("***** oAnalyticProjection::LaunchCPU()-> An error occurred when trying to initialize main loop stop index !" << endl);
      return 1;
    }

    if(p_scannerManager->PROJ_GetModalityStartValueInnerLoop(0) < 0)
    {
      Cerr("***** oAnalyticProjection::LaunchCPU()-> An error occurred when trying to initialize inner loop start index !" << endl);
      return 1;
    }

    // Prepare pre-computed sums of events to avoid the exponential evolution of the percentage (for PET)
    // TODO Perhaps replace this with a call to scannerManager for potential error management
    int32_t nb_total_elts = mp_Scanner->GetSystemNbElts();
    int64_t* elements_sum = (int64_t*)malloc(nb_total_elts*sizeof(int64_t));
    elements_sum[0] = 0;
    
    for (int64_t idx_elt=1 ; idx_elt<nb_total_elts ; idx_elt++) 
      elements_sum[idx_elt] = elements_sum[idx_elt-1] + (uint64_t)(nb_total_elts-idx_elt);

    uint64_t* total_events = (uint64_t*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(uint64_t));
    HPFLTNB* total_prompts = (HPFLTNB*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(HPFLTNB));

    // Index for progression printing
    uint64_t printing_index = 0;
    uint64_t progression_index_total = p_scannerManager->PROJ_GetProgressionFinalValue()*
                                                           mp_ID->GetNbTimeFrames()*
                                                           mp_ID->GetNbRespGates()*
                                                           mp_ID->GetNbCardGates();
    
    // look on the dynamic frames
    for(int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
    {
      // Time reference to initialize the timestamp of the events
      uint32_t timestamp = mp_ID->GetFrameTimeStartInMs(bed,fr);
      uint64_t* total_events_in_frame = (uint64_t*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(uint64_t));
      HPFLTNB* total_prompts_in_frame = (HPFLTNB*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(HPFLTNB));
    
      for(int rg=0 ; rg<mp_ID->GetNbRespGates() ; rg++)
      {
        uint64_t* total_events_in_rgate = (uint64_t*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(uint64_t));
        HPFLTNB* total_prompts_in_rgate = (HPFLTNB*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(HPFLTNB));
        
        for(int cg=0 ; cg<mp_ID->GetNbCardGates() ; cg++)
        {
          uint64_t* total_events_in_cgate = (uint64_t*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(uint64_t));
          HPFLTNB* total_prompts_in_cgate = (HPFLTNB*)calloc(mp_ID->GetNbThreadsForProjection(),sizeof(HPFLTNB));

          if(mp_ID->GetNbTimeFrames() > 1)
            if(m_verbose>=0) Cout(endl << "Processing frame #" << fr+1 << " out of " << mp_ID->GetNbTimeFrames() << endl);
          if(mp_ID->GetNbRespGates() > 1)
            if(m_verbose>=0) Cout(endl << "Processing resp gate #" << rg+1 << " out of " << mp_ID->GetNbRespGates() << endl);
          if(mp_ID->GetNbCardGates() > 1)
            if(m_verbose>=0) Cout(endl << "Processing card gate #" << cg+1 << " out of " << mp_ID->GetNbCardGates() << endl);

          // Used to catch errors inside the parallel loop
          bool problem = false;

          // Multi-threaded loop on the scanner elements
          // Use (static,1) as dynamic mode actually different output histogram with multithreading.
          int64_t idx_elt1; //openMP throws error with unsigned
          #pragma omp parallel for private(idx_elt1) schedule(static, 1)
          for (idx_elt1=main_loop_start_index ; idx_elt1<(int)main_loop_stop_index ; idx_elt1++)
          {
            // Get the thread index
            int th = 0;
            #ifdef CASTOR_OMP
            th = omp_get_thread_num();
            #endif

            // Initialize inner loop start and stop values
            int64_t inner_loop_start_index = p_scannerManager->PROJ_GetModalityStartValueInnerLoop(idx_elt1) ;
            int64_t inner_loop_stop_index = mp_Scanner->GetSystemNbElts();
                    
            // Inner loop on scanner elements (idx_elt2) which are used to form LOR using the first scanner element (idx_elt1)
            for (int64_t idx_elt2=inner_loop_start_index ; idx_elt2<inner_loop_stop_index ; idx_elt2++)
            {
              // Print progression (do not log out with Cout here)
              if (th==0)
              {
                if (printing_index%10000==0)
                {
                  int64_t progression_index_current = p_scannerManager->PROJ_GetCurrentProgression(idx_elt1, idx_elt2, elements_sum, 
                                                                                                         mp_ID->GetNbRespGates(), 
                                                                                                         mp_ID->GetNbCardGates(), 
                                                                                                                                  fr, rg, cg);

                  FLTNB percent = ( (FLTNB)progression_index_current  /((FLTNB)progression_index_total) ) * ((FLTNB)100);
                  cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b      "
                     << percent << " %                    ";

                }
                printing_index++;
              }
            
              // Allocate an event using the iDataFile
              vEvent* event = m2p_DataFile[bed]->PROJ_GenerateEvent(idx_elt1, idx_elt2, th);
      
              // TODO : Perhaps replace this with a call to scannerManager for potential error management
              if (!mp_Scanner->IsAvailableLOR(event->GetID1(0), event->GetID2(0))) continue;

              // Generate the projection event and compute the projection line 
              oProjectionLine* line = mp_ProjectorManager->ComputeProjectionLine(event, th);
              if (line==NULL)
              {
                Cerr("***** oAnalyticProjection::LaunchCPU() -> A problem occurred while computing the projection line !" << endl);
                // Specify that there was a problem
                problem = true;
                // We must continue here because we are inside an OpenMP loop
                continue;
              }

              // Optimize in the data space (forward-proj, update, backward-proj) if the line has been performed
              if (line->NotEmptyLine())
              {
                if(mp_ComputeProjection->DataUpdateStep(m2p_DataFile[bed], line, mp_ImageSpace, event, fr, rg, cg, th, timestamp, m_discardZeroEvent) )
                {
                  Cerr("***** oAnalyticProjection::LaunchCPU() -> An error occurred during the data update step !" << endl);
                  //return 1; TODO : return in open MP structured block (stop all threads)
                }
                
                // If we chose to discard zero event, write it only if event value > 0
                if( !m_discardZeroEvent 
                ||  event->GetEventValue(0)>0. )
                {
                  // Increment number of events
                  total_events[th]++;
                  total_events_in_cgate[th]++;
                  total_events_in_rgate[th]++;
                  total_events_in_frame[th]++;
                  HPFLTNB event_value = event->GetEventValue(0);
                  total_prompts[th] += event_value;
                  total_prompts_in_cgate[th] += event_value;
                  total_prompts_in_rgate[th] += event_value;
                  total_prompts_in_frame[th] += event_value;
                }
              }

            }

          } // End thread loop

          // If a problem was encountered, then report it here
          if (problem)
          {
            Cerr("***** oAnalyticProjection::LaunchCPU() -> A problem occurred inside the parallel loop over events !" << endl);
            return 1;
          }

          // Write data for this frame
          if (m2p_DataFile[bed]->PROJ_WriteData() )
          {
            Cerr("***** oAnalyticProjection::LaunchCPU()-> An error occurred during the data writing step" << endl);
            return 1;
          }

          // Merge counters
          for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_events_in_cgate[0] += total_events_in_cgate[th];
          for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_prompts_in_cgate[0] += total_prompts_in_cgate[th];
          
          if(m_verbose>=2 && mp_ID->GetNbCardGates() > 1)
          {
            Cout(endl << "Total events in cgate #" << cg+1 << ": " << total_events_in_cgate[0] << endl);
            Cout("Total prompts in cgate #" << cg+1 << ": " << total_prompts_in_cgate[0] << endl);
          }
          
          free(total_prompts_in_cgate);
          free(total_events_in_cgate);
        }
        
        // Merge counters
        for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_events_in_rgate[0] += total_events_in_rgate[th];
        for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_prompts_in_rgate[0] += total_prompts_in_rgate[th];
        
        if(m_verbose>=2 && mp_ID->GetNbRespGates() > 1)
        {
          Cout(endl << "Total events in rgate #" << rg+1 << ": " << total_events_in_rgate[0] << endl);
          Cout("Total prompts in rgate #" << rg+1 << ": " << total_prompts_in_rgate[0] << endl);
        }
        free(total_prompts_in_rgate);
        free(total_events_in_rgate);
      }
      
      // Get time reference for this frame in order to initialize the timestamp (in ms) of the events
      timestamp += mp_ID->GetFrameDurationInMs(bed,fr);

      // Merge counters
      for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_events_in_frame[0] += total_events_in_frame[th];
      for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_prompts_in_frame[0] += total_prompts_in_frame[th];

      if(m_verbose>=2 && mp_ID->GetNbTimeFrames() > 1)
      {
        Cout(endl << "Total events in frame #" << fr+1 << ": " << total_events_in_frame[0] << endl);
        Cout("Total prompts in frame #" << fr+1 << ": " << total_prompts_in_frame[0] << endl);
      }
        
      free(total_prompts_in_frame);
      free(total_events_in_frame);
    }
    
    // Compute simulated acquisition duration for 4D projection
    int acquisition_duration = 0;
    for(int fr=0 ; fr<mp_ID->GetNbTimeFrames() ; fr++)
      acquisition_duration += mp_ID->GetFrameDurationInMs(bed,fr);
     
    // If 3D, set the acquisition duration to 1 by default
    if(acquisition_duration <= 0 ) acquisition_duration = 1;  // Set default value for acquisition duration
    
    // End of progression printing (do not log out with Cout here)
    cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b      100 %                    " << endl;

    // Save a projection image for output visualization (currently only for SPECT)
    mp_ImageSpace->PROJ_SaveProjectionImage();
    
    // Merge counters
    if (m_verbose>=2)
    {
      Cout("Total number of projected events:" << endl);
      for (int th=0; th<mp_ID->GetNbThreadsForProjection(); th++) Cout("  --> Thread " << th << " | " << total_events[th] << " events" << endl);
    }
    for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_events[0] += total_events[th];
    if (m_verbose>=1) Cout("Final total number of events projected: " << total_events[0] << endl);
    for (int th=1; th<mp_ID->GetNbThreadsForProjection(); th++) total_prompts[0] += total_prompts[th];
    if (m_verbose>=1) Cout("Final total number of prompts: " << total_prompts[0] << endl);
    
    // Write header (Cdh)
    // TODO : acquisition duration must depend on frames, and we should have one datafile by frame
    m2p_DataFile[bed]->SetStartTime( mp_ID->GetFrameTimeStartInSec(bed,0) );
    m2p_DataFile[bed]->SetDuration( mp_ID->GetFrameDurationInSec(bed,0) );
    m2p_DataFile[bed]->SetNbEvents(total_events[0]);
    m2p_DataFile[bed]->WriteHeader();
    m2p_DataFile[bed]->PROJ_DeleteTmpDataFile();

    free(total_prompts);
    free(total_events);
    free(elements_sum);
  }

  // Free memory
  if(mp_Scanner->GetScannerType() == SCANNER_SPECT_CONVERGENT) mp_ImageSpace->PROJ_DeallocateProjectionImage(mp_Scanner->PROJ_GetSPECTNbProjections());  
  mp_ImageSpace->LMS_DeallocateAttenuationImage();
  mp_ImageSpace->LMS_DeallocateForwardImage();
  mp_ImageSpace->LMS_DeallocateImage();
  
 // Clock total
  clock_t clock_stop = clock();
  time_t time_stop = time(NULL);
  Cout (endl << endl << "  Total Time spent - Analytic Projection | User: " << time_stop-time_start << " sec | CPU: " << (clock_stop-clock_start)/((FLTNB)CLOCKS_PER_SEC) << " sec" << endl);
  
  return 0;
}
