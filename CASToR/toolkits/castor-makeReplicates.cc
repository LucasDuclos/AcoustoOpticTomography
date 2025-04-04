
/*!
  \file
  \ingroup utils
  \brief  This program is used to build replicates of less statistics than the provided input list-mode file.
*/

#include "gVariables.hh"
#include "gOptions.hh"
#include "oImageDimensionsAndQuantification.hh"
#include "vDataFile.hh"
#include "iDataFilePET.hh"
#include "iDataFileAOT.hh"
#include "sOutputManager.hh"
#include "sRandomNumberGenerator.hh"

// =============================================================================================================================================
// =============================================================================================================================================
// =============================================================================================================================================
//                                                        H E L P     F U N C T I O N S
// =============================================================================================================================================
// =============================================================================================================================================
// =============================================================================================================================================


/*!
  \fn      ShowHelp()
  \brief   Display main command line options for castor-mergeBedPositions
*/
void ShowHelp()
{
  // Show help
  cout << endl;
  cout << "Usage:  castor-makeReplicates  -df datafile.cdh  -(f/d)out output  (-rep replicates || -down factor)  [settings]" << endl;
  cout << endl;
  cout << "This program can be used to build replicates of less statistics than the provided input CASToR datafile by the use" << endl;
  cout << "of the '-rep' options and choosing the number of replicates. It can also be used to create only one datafile by" << endl;
  cout << "down-scaling the statistics by the use of the '-down' option. The input datafile is assumed to be a list-mode." << endl;
  cout << endl;
  cout << "If the '-rep' option is used, based on the provided number of replicates, the latter are built based on the time" << endl;
  cout << "flag of each event: a pseudo gating strategy is used where each gate will correspond to a replicate. To avoid" << endl;
  cout << "seeing a difference between replicates when a high number is required, the pseudo gate duration has to be as small" << endl;
  cout << "as possible. The default value is thus 2 ms." << endl;
  cout << endl;
  cout << "If the '-down' option is used, based on the provided down-sampling factor ]0.;1.[, for each event, a random number" << endl;
  cout << "is shot to decide whether the event is kept or not." << endl;
  cout << "All additive corrections as well as the quantification factor are modified accordingly to keep the same image quantification." << endl;
  cout << endl;
  cout << "[Mandatory parameters]:" << endl;
  cout << "  -df datafile.cdh     : Give a CASToR list-mode datafile." << endl;
  cout << "  -fout name           : Give the root name for all output files (no default, alternative to -dout)" << endl;
  cout << "  -dout name           : Give the name of the output directory where all output files will be written (no default, alternative to -fout)" << endl;
  cout << "  One of the two following options has to be chosen: " << endl;
  cout << "  -rep value           : Give the number of replicates to be built using a pseudo-gating strategy." << endl;
  cout << "  -down value          : Give the down-sampling factor to build only one file with less statistics ]0.;1.[." << endl;
  cout << endl;
  cout << "[Options]:" << endl;
  cout << "  -rng                 : Give a seed for the random number generator (should be >=0)" << endl;
  cout << "  -gate value          : Give the pseudo gate duration in milliseconds (default: 2)" << endl;
  cout << "  -vb value            : Give the verbosity level, from 0 (no verbose) to 2 (default: 1)" << endl;
  cout << "  --help,-h,-help      : Print out this help page." << endl; // managed by main
  cout << endl;
  #ifdef BUILD_DATE
  cout << "  Build date: " << BUILD_DATE << endl;
  cout << endl;
  #endif
  #ifdef CASTOR_VERSION
  cout << "  This program is part of the CASToR release version " << CASTOR_VERSION << "." << endl;
  cout << endl;
  #endif
}

// =============================================================================================================================================
// =============================================================================================================================================
// =============================================================================================================================================
//                                                        M A I N     P R O G R A M
// =============================================================================================================================================
// =============================================================================================================================================
// =============================================================================================================================================

int main(int argc, char** argv)
{
  // ============================================================================================================
  // MPI stuff (we make all instances except the first one returning 0 directly)
  // ============================================================================================================
  int mpi_rank = 0;
  #ifdef CASTOR_MPI
  int mpi_size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_rank!=0) return 0;
  #endif

  // No argument, then show help
  if (argc==1)
  {
    ShowHelp();
    Exit(EXIT_SUCCESS);
  }

  // ============================================================================================================
  // Parameterized variables with their default values
  // ============================================================================================================

  // --------------------------------------------------------------------------------
  // Input settings
  // --------------------------------------------------------------------------------

  // Input datafile
  string datafile = "";

  // --------------------------------------------------------------------------------
  // Output settings
  // --------------------------------------------------------------------------------

  // Output directory name.
  string path_dout = "";
  // Or root name
  string path_fout = "";

  // --------------------------------------------------------------------------------
  // Miscellaneous settings
  // --------------------------------------------------------------------------------

  // Number of replicates to build
  int nb_replicates = -1;
  // Pseudo gate duration in ms (default: 2)
  uint32_t gate_duration = 2;
  // Down sampling factor
  HPFLTNB down_sampling = -1.;
  // Verbose level
  int verbose = 1;
  // Initial seed for random number generator
  int64_t random_generator_seed = -1;

  // ============================================================================================================
  // Read command-line parameters
  // ============================================================================================================

  // Must manually increment the option index when an argument is needed after an option
  for (int i=1; i<argc; i++)
  {
    // Get the option as a string
    string option = (string)argv[i];

    // --------------------------------------------------------------------------------
    // Miscellaneous settings
    // --------------------------------------------------------------------------------

    // Show help
    if (option=="-h" || option=="--help" || option=="-help")
    {
      ShowHelp();
      Exit(EXIT_SUCCESS);
    }
    // RNG seed
    else if (option=="-rng")
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      if (ConvertFromString(argv[i+1], &random_generator_seed))
      {
        Cerr("***** castor-makeReplicates() -> Exception when trying to read provided number '" << random_generator_seed << " for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      i++;
    }
    // Number of replicates to build
    else if (option=="-rep")
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      if (ConvertFromString(argv[i+1], &nb_replicates))
      {
        Cerr("***** castor-makeReplicates() -> Exception when trying to read provided number of replicates '" << nb_replicates << " for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      i++;
    }
    // Pseudo gate duration in milliseconds
    else if (option=="-gate")
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      if (ConvertFromString(argv[i+1], &gate_duration))
      {
        Cerr("***** castor-makeReplicates() -> Exception when trying to read provided pseudo gate duration '" << gate_duration << " for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      i++;
    }
    // Down sampling factor
    else if (option=="-down")
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      if (ConvertFromString(argv[i+1], &down_sampling))
      {
        Cerr("***** castor-makeReplicates() -> Exception when trying to read provided down-sampling factor '" << down_sampling << " for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      i++;
    }
    // General verbosity level
    else if (option=="-vb")
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      if (ConvertFromString(argv[i+1], &verbose))
      {
        Cerr("***** castor-makeReplicates() -> Exception when trying to read provided verbosity level '" << verbose << " for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      i++;
    }

    // --------------------------------------------------------------------------------
    // Input settings
    // --------------------------------------------------------------------------------

    // Images
    else if (option=="-df") // This is a mandatory option
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      datafile = (string)argv[i+1];
      i++;
    }

    // --------------------------------------------------------------------------------
    // Output settings
    // --------------------------------------------------------------------------------

    // Name of the output directory
    else if (option=="-dout") // This is a mandatory option
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      path_dout = argv[i+1];
      i++;
    }
    // Base name of the output files
    else if (option=="-fout") // This is a mandatory option
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-makeReplicates() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      path_fout = argv[i+1];
      i++;
    }

    // --------------------------------------------------------------------------------
    // Unknown option!
    // --------------------------------------------------------------------------------

    else
    {
      Cerr("***** castor-makeReplicates() -> Unknown option '" << option << "' !" << endl);
      Exit(EXIT_FAILURE);
    }
  }

  // ============================================================================================================
  // Some checks
  // ============================================================================================================

  // Data file
  if (datafile=="")
  {
    Cerr("***** castor-makeReplicates() -> Please provide an input datafile !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Output files
  if (path_fout.empty() && path_dout.empty())
  {
    Cerr("***** castor-makeReplicates() -> Please provide an output option for output files (-fout or -dout) !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Check that only one option has been provided
  if (!path_fout.empty() && !path_dout.empty())
  {
    Cerr("***** castor-makeReplicates() -> Please provide either output option -fout or -dout but not both !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Check that at least a number of replicates or a down sampling factor has been provided
  if (nb_replicates==-1 && down_sampling==-1.)
  {
    Cerr("***** castor-makeReplicates() -> Please provide one of the two following options '-rep' or '-down' !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Check number of replicates when no down-sampling factor is provided
  if (down_sampling==-1. && nb_replicates<1)
  {
    Cerr("***** castor-makeReplicates() -> Please provide a correct number of replicates (at least 2) !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Check pseudo gate duration when no down-sampling factor is provided
  if (down_sampling==-1. && gate_duration<1)
  {
    Cerr("***** castor-makeReplicates() -> Please provide a correct pseudo gate duration in milliseconds (at least 1) !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Check dow sampling factor when no number of replicates is provided
  if (nb_replicates==-1 && (down_sampling<=0. || down_sampling>=1.))
  {
    Cerr("***** castor-makeReplicates() -> Please provide a correct down-sampling factor ]0.;1.[ !" << endl);
    Exit(EXIT_FAILURE);
  }

  // ============================================================================================================
  // Create sOutputManager
  // ============================================================================================================
  sOutputManager* p_OutputManager = sOutputManager::GetInstance();  
  // Set verbose level
  p_OutputManager->SetVerbose(verbose);
  // Set MPI rank
  p_OutputManager->SetMPIRank(mpi_rank);
  // Initialize output directory and base name
  if (p_OutputManager->InitOutputDirectory(path_fout, path_dout))
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while initializing output directory !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Log command line
  if (p_OutputManager->LogCommandLine(argc,argv))
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while logging command line arguments !" << endl);
    Exit(EXIT_FAILURE);
  }

  // ============================================================================================================
  // Random number generator
  // ============================================================================================================

  sRandomNumberGenerator* p_RandomNumberGenerator = sRandomNumberGenerator::GetInstance(); 
  p_RandomNumberGenerator->SetVerbose(verbose);
  // Use a user-provided seed to initialize the RNG if one has been provided. Use random number otherwise.
  int no_thread = 0;
  int one_generator = 1;
  if (random_generator_seed>=0) p_RandomNumberGenerator->Initialize(random_generator_seed, no_thread, one_generator);
  else p_RandomNumberGenerator->Initialize(no_thread, one_generator);

  // ============================================================================================================
  // Create sScannerManager (in order to get the datafile type)
  // ============================================================================================================
  sScannerManager* p_ScannerManager = sScannerManager::GetInstance();  
  p_ScannerManager->SetVerbose(verbose);
  // Get system name from the dataFile
  string scanner_name = "";
  if (ReadDataASCIIFile(datafile, "Scanner name", &scanner_name, 1, KEYWORD_MANDATORY))
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while trying to find the system name in the datafile header !" << endl);
    Exit(EXIT_FAILURE);
  } 
  if (p_ScannerManager->FindScannerSystem(scanner_name) )
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while searching for scanner system !" << endl);
    Exit(EXIT_FAILURE);
  } 
  if (p_ScannerManager->BuildScannerObject() )
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred during scanner object construction ! !" << endl);
    Exit(EXIT_FAILURE);
  }
  if (p_ScannerManager->InstantiateScanner() )
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while creating Scanner object !" << endl);
    Exit(EXIT_FAILURE);
  }
  if (p_ScannerManager->GetGeometricInfoFromDataFile(datafile))
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while retrieving scanner fields from the datafile header !" << endl);
    Exit(EXIT_FAILURE);
  }
  if (p_ScannerManager->BuildLUT() )
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while generating/reading the LUT !" << endl);
    Exit(EXIT_FAILURE);
  } 

  // ============================================================================================================
  // Create the input datafile
  // ============================================================================================================

  // Create a default image dimensions and quantification object
  oImageDimensionsAndQuantification* p_ID = new oImageDimensionsAndQuantification();
  p_ID->SetDefault();
  p_ID->SetVerbose(verbose);
  // Create the datafile based on the data type
  vDataFile* p_DataFile = NULL;
  if (p_ScannerManager->GetScannerType() == SCANNER_PET) p_DataFile = new iDataFilePET();
  else
  {
    Cerr("***** castor-makeReplicates() -> Unknown scanner type (" << p_ScannerManager->GetScannerType() << ") for datafile construction ! Abort." << endl);
    Exit(EXIT_FAILURE);
  }
  p_DataFile->SetImageDimensionsAndQuantification(p_ID);
  p_DataFile->SetHeaderDataFileName(datafile);
  p_DataFile->SetVerbose(verbose);
  p_DataFile->SetBedIndex(0);
  bool do_not_affect_quantification = false;
  if (p_DataFile->ReadInfoInHeader(do_not_affect_quantification))
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred during datafile header reading ! Abort." << endl);
    Exit(EXIT_FAILURE);
  }
  if (p_DataFile->CheckParameters())
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred while checking datafile parameters ! Abort." << endl);
    Exit(EXIT_FAILURE);
  }
  if (p_DataFile->ComputeSizeEvent())
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred in datafile initialization ! Abort." << endl);
    Exit(EXIT_FAILURE);
  }
  if (p_DataFile->InitializeMappedFile())
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred in datafile initialization ! Abort." << endl);
    Exit(EXIT_FAILURE);
  }
  if (p_DataFile->PrepareDataFile())
  {
    Cerr("***** castor-makeReplicates() -> A problem occurred in datafile preparation ! Abort." << endl);
    Exit(EXIT_FAILURE);
  }
  // Check if datafile is a listmode, otherwise, throw an error
  if (p_DataFile->GetDataMode()!=MODE_LIST)
  {
    Cerr("***** castor-makeReplicates() -> The input datafile is not a list-mode, this program is only suitable to list-mode files !" << endl);
    Exit(EXIT_FAILURE);
  }

  // ============================================================================================================
  // Pseudo-gating mode
  // ============================================================================================================

  if (down_sampling==-1.)
  {
    // --------------------------------------------------
    // Initialization part
    // --------------------------------------------------
    // Create output datafiles
    vDataFile** pp_OutputDataFile = (vDataFile**)malloc(nb_replicates*sizeof(vDataFile*));
    for (int rep=0; rep<nb_replicates; rep++)
    {
      if (p_ScannerManager->GetScannerType() == SCANNER_PET) pp_OutputDataFile[rep] = new iDataFilePET();
      // Build output data file from the input one
      if (pp_OutputDataFile[rep]->SetParametersFrom(p_DataFile))
      {
        Cerr("***** castor-makeReplicates() -> An error occurred while setting parameters of output file from input file (replicate " << rep+1 << ") !" << endl);
        Exit(EXIT_FAILURE);
      }
      // Build a suffix for the filename based on the replicate number
      char tmp[10]; sprintf(tmp,"%d",rep+1); string rep_suffix = "_" + ((string)tmp);
      // Open output file
      if (pp_OutputDataFile[rep]->OpenFileForWriting(rep_suffix))
      {
        Cerr("***** castor-makeReplicates() -> An error occurred while opening file for writing (replicate " << rep+1 << ") !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
    // --------------------------------------------------
    // Processing part
    // --------------------------------------------------
    // Verbose
    if (verbose>=1) Cout("castor-makeReplicates() -> Start pseudo-replicating input datafile with " << nb_replicates << " replicates ..." << endl);
    if (verbose>=2) Cout("  --> Use a pseudo-gate duration of " << gate_duration << " ms" << endl);
    // Counters
    int64_t* nb_events_per_rep = (int64_t*)calloc(nb_replicates,sizeof(int64_t));
    int64_t total_events = 0;
    // Pseudo gating management
    int current_replicate = 0;
    uint32_t current_replicate_time = 0;
    // Get index start and stop
    int64_t index_start = 0; int64_t index_stop = 0;
    p_DataFile->GetEventIndexStartAndStop(&index_start, &index_stop);
    // Launch the loop on all events
    int64_t printing_index = 0;
    for (int64_t index = index_start  ;  index < index_stop  ;  index++)
    {
      // Verbose
      if (verbose>=2)
      {
        if (printing_index%10000==0)
        {
          int percent = ((int)( ((FLTNB)(index-index_start)) * 100. / ((FLTNB)(index_stop-index_start)) ));
          cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b      "
               << percent << " %                    " << flush;
        }
        printing_index++;
      }
      // Get the current event
      vEvent* event = p_DataFile->GetEvent(index);
      if (event==NULL)
      {
        Cerr("***** castor-makeReplicates() -> An error occurred while getting the event from index " << index << " !" << endl);
        for (int rep=0; rep<nb_replicates; rep++)
          if (pp_OutputDataFile[rep]->CloseFile())
            Cerr("***** castor-makeReplicates() -> An error occurred while closing file during writing (replicate " << rep+1 << ") !" << endl);
        Exit(EXIT_FAILURE);
      }
      // Increment the total number of events
      total_events++;
      // Get time for this event
      uint32_t this_time = event->GetTimeInMs();
      // Check if the time for this event has passed the current replicate time of more than the gate duration
      if (this_time >= current_replicate_time + gate_duration)
      {
        // Compute the time difference
        uint32_t time_difference = this_time - current_replicate_time;
        // Compute the number of gates/replicates passed
        uint32_t nb_replicates_passed = time_difference / gate_duration;
        // Update the reference time
        current_replicate_time += nb_replicates_passed * gate_duration;
        // Update the current gate/replicate
        current_replicate = (current_replicate + nb_replicates_passed) % nb_replicates;
      }
      // Increment the current gate/replicate count
      nb_events_per_rep[current_replicate]++;
      // Apply down sampling factor to additive corrections
      event->MultiplyAdditiveCorrections(1./((FLTNB)nb_replicates));
      // Write the event to the current gate/replicate
      pp_OutputDataFile[current_replicate]->WriteEvent(event);
    }
    if (verbose>=2) cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
                         << "  --> 100 %                        " << endl;
    // --------------------------------------------------
    // Conclusion part
    // --------------------------------------------------
    // Loop on replicates
    for (int rep=0; rep<nb_replicates; rep++)
    {
      // Close file
      if (pp_OutputDataFile[rep]->CloseFile())
      {
        Cerr("***** castor-makeReplicates() -> An error occurred while closing file after writing (replicate " << rep+1 << ") !" << endl);
        Exit(EXIT_FAILURE);
      }
      // Set number of events
      pp_OutputDataFile[rep]->SetNbEvents(nb_events_per_rep[rep]);
      // Apply dow sampling factor onto the calibration factor
      pp_OutputDataFile[rep]->SetCalibrationFactor(p_DataFile->GetCalibrationFactor()*((FLTNB)nb_replicates));
    }
    // Verbose
    if (verbose>=2)
    {
      Cout("castor-makeReplicates() -> Here are some events statistics:" << endl);
      Cout("  --> Total number of events: " << total_events << endl);
      for (int rep=0; rep<nb_replicates; rep++) Cout("  --> Number of events for replicate " << rep+1 << ": " << nb_events_per_rep[rep] << endl);
    }
    // Loop on replicates
    for (int rep=0; rep<nb_replicates; rep++)
    {
      // Write header
      if (pp_OutputDataFile[rep]->WriteHeader())
      {
        Cerr("***** castor-makeReplicates() -> An error occurred while writing output header file (replicate " << rep+1 << ") !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
  }

  // ============================================================================================================
  // Down sampling mode
  // ============================================================================================================

  if (nb_replicates==-1)
  {
    // --------------------------------------------------
    // Initialization part
    // --------------------------------------------------
    // Create output datafile
    vDataFile* p_OutputDataFile = NULL;
    if (p_ScannerManager->GetScannerType() == SCANNER_PET) p_OutputDataFile = new iDataFilePET();
    // Build output data file from the input one
    if (p_OutputDataFile->SetParametersFrom(p_DataFile))
    {
      Cerr("***** castor-makeReplicates() -> An error occurred while setting parameters of output file from input file !" << endl);
      Exit(EXIT_FAILURE);
    }
    // Open output file
    if (p_OutputDataFile->OpenFileForWriting())
    {
      Cerr("***** castor-makeReplicates() -> An error occurred while opening file for writing !" << endl);
      Exit(EXIT_FAILURE);
    }
    // --------------------------------------------------
    // Processing part
    // --------------------------------------------------
    // Verbose
    if (verbose>=1) Cout("castor-makeReplicates() -> Start down-sampling input datafile with factor " << down_sampling << " ..." << endl);
    // Counters
    int64_t kept_events = 0, rejected_events = 0;
    // Get index start and stop
    int64_t index_start = 0; int64_t index_stop = 0;
    p_DataFile->GetEventIndexStartAndStop(&index_start, &index_stop);
    // Launch the loop on all events
    int64_t printing_index = 0;
    for (int64_t index = index_start  ;  index < index_stop  ;  index++)
    {
      // Verbose
      if (verbose>=2)
      {
        if (printing_index%10000==0)
        {
          int percent = ((int)( ((FLTNB)(index-index_start)) * 100. / ((FLTNB)(index_stop-index_start)) ));
          cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b      "
               << percent << " %                    " << flush;
        }
        printing_index++;
      }
      // Get the current event
      vEvent* event = p_DataFile->GetEvent(index);
      if (event==NULL)
      {
        Cerr("***** castor-makeReplicates() -> An error occurred while getting the event from index " << index << " !" << endl);
        if (p_OutputDataFile->CloseFile())
          Cerr("***** castor-makeReplicates() -> An error occurred while closing file during writing !" << endl);
        Exit(EXIT_FAILURE);
      }
      // Shoot a random number
      HPFLTNB chance = p_RandomNumberGenerator->GenerateExtraRdmNber(0);
      // Continue to next event or not ?
      if (chance>down_sampling)
      {
        rejected_events++;
        continue;
      }
      // Apply down sampling factor to additive corrections
      event->MultiplyAdditiveCorrections(down_sampling);
      // Write the event
      p_OutputDataFile->WriteEvent(event);
      kept_events++;
    }
    if (verbose>=2) cout << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
                         << "  --> 100 %                        " << endl;
    // --------------------------------------------------
    // Conclusion part
    // --------------------------------------------------
    // CLose file
    if (p_OutputDataFile->CloseFile())
    {
      Cerr("***** castor-makeReplicates() -> An error occurred while closing file after writing !" << endl);
      Exit(EXIT_FAILURE);
    }
    // Set number of events
    p_OutputDataFile->SetNbEvents(kept_events);
    // Apply dow sampling factor onto the calibration factor
    p_OutputDataFile->SetCalibrationFactor(p_DataFile->GetCalibrationFactor()/down_sampling);
    // Verbose
    if (verbose>=2)
    {
      int64_t total_events = kept_events + rejected_events;
      Cout("castor-makeReplicates() -> Here are some events statistics:" << endl);
      Cout("  --> Total number of events: " << total_events << endl);
      Cout("  --> Kept events: " << kept_events << endl);
      Cout("  --> Rejected events: " << rejected_events << endl);
      Cout("  --> Effective down-sampling factor: " << ((double)kept_events) / ((double)total_events) << endl);
    }
    // Write header
    if (p_OutputDataFile->WriteHeader())
    {
      Cerr("***** castor-makeReplicates() -> An error occurred while writing output header file !" << endl);
      Exit(EXIT_FAILURE);
    }
  }

  // ============================================================================================================
  // Free memory properly
  // ============================================================================================================

  // Ending
  #ifdef CASTOR_MPI
  MPI_Finalize();
  #endif
  return EXIT_SUCCESS;
}
