
/*!
  \file
  \ingroup utils
  \brief  This program is used to merge a group of images corresponding to multiple bed positions reconstructed separately.
*/

#include "gVariables.hh"
#include "gOptions.hh"
#include "oIterativeAlgorithm.hh"
#include "oSensitivityGenerator.hh"
#include "oImageDimensionsAndQuantification.hh"
#include "iDataFilePET.hh"
#include "iDataFileAOT.hh"
#include "sOutputManager.hh"
#include "sScannerManager.hh"
#include "sRandomNumberGenerator.hh"
#include "iScannerPET.hh"
#include "sAddonManager.hh"

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
  cout << "Usage:  castor-sumImages  -img image1.hdr  -img image2.hdr  -(f/d)out output  [settings]" << endl;
  cout << endl;
  cout << "This program can be used to perform operations on a bunch of images of the same dimensions." << endl;
  cout << "The sum or average image can be computed." << endl;
  cout << "The variance or standard-deviation image can be computed (the biased one, i.e. divided by the number of images and not minus one)." << endl;
  cout << "If a reference image is provided, the bias as well as the root mean square error images can also be computed." << endl;
  cout << "All operations are performed independently voxel by voxel." << endl;
  cout << "Important note: this program only works with 3D images (without time dimension)." << endl;
  cout << endl;
  cout << "[Mandatory parameters]:" << endl;
  cout << "  -img imageX.hdr      : Give an image (must give at least two images)" << endl;
  cout << "  -fout name           : Give the root name for all output files (no default, alternative to -dout)" << endl;
  cout << "  -dout name           : Give the name of the output directory where all output files will be written (no default, alternative to -fout)" << endl;
  cout << endl;
  cout << "[Options]:" << endl;
  cout << "  -sum                 : Compute the sum image" << endl;
  cout << "  -avg                 : Compute the average image" << endl;
  cout << "  -var                 : Compute the variance image" << endl;
  cout << "  -stdv                : Compute the standard-deviation image" << endl;
  cout << "  -ref imageRef.hdr    : Provide a reference image (needed for bias and rmse options)" << endl;
  cout << "  -bias                : Compute the bias image with respect to the provided reference image" << endl;
  cout << "  -mse                 : Compute the mean square error image with respect to the provided reference image" << endl;
  cout << "  -rmse                : Compute the root mean square error image with respect to the provided reference image" << endl;
  cout << "  -norm                : Compute normalized values:" << endl;
  cout << "                          - with respect to the average for the standard-deviation" << endl;
  cout << "                          - with respect to the reference for the bias and root mean square error" << endl;
  cout << "                          - other computations are left unchanged" << endl;
  cout << "  -vb value            : Give the verbosity level, from 0 (no verbose) to 2 (default: 1)" << endl;
  cout << "  --help,-h,-help      : Print out this help page." << endl; // managed by main
  cout << endl;
  #ifdef CASTOR_OMP
  cout << "  Compiled with OpenMP (will use as many cores as available)" << endl;
  #endif
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
  // MPI stuff (we make all instances but the first one returning 0 directly)
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

  // Number of images
  uint32_t nb_images = 0;
  // Vector containing string pointing to the images filenames
  vector<string> path_to_image_filename;
  // String to the reference image filename
  string path_to_reference_image_filename = "";

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

  // Compute the sum
  bool flag_sum = false;
  // Compute the average
  bool flag_mean = false;
  // Compute the standard-deviation
  bool flag_stdv = false;
  // Compute the variance
  bool flag_var = false;
  // Compute the bias
  bool flag_bias = false;
  // Compute the mean square error
  bool flag_mse = false;
  // Compute the root mean square error
  bool flag_rmse = false;
  // Compute normalized values
  bool flag_norm = false;
  // Verbose level
  int verbose = 1;

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
    // General verbosity level
    else if (option=="-vb")
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-sumImages() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      if (ConvertFromString(argv[i+1], &verbose))
      {
        Cerr("***** castor-sumImages() -> Exception when trying to read provided verbosity level '" << verbose << " for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      i++;
    }
    // Sum option
    else if (option=="-sum") flag_sum = true;
    // Average option
    else if (option=="-avg") flag_mean = true;
    // Variance option
    else if (option=="-var") flag_var = true;
    // Standard deviation option
    else if (option=="-stdv") flag_stdv = true;
    // Bias option
    else if (option=="-bias") flag_bias = true;
    // Mean square error option
    else if (option=="-mse") flag_mse = true;
    // Root mean square error option
    else if (option=="-rmse") flag_rmse = true;
    // Normalized values option
    else if (option=="-norm") flag_norm = true;

    // --------------------------------------------------------------------------------
    // Input settings
    // --------------------------------------------------------------------------------

    // Images
    else if (option=="-img") // This is a mandatory option
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-sumImages() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      string file_name = (string)argv[i+1];
      path_to_image_filename.push_back(file_name);
      nb_images++;
      i++;
    }
    // Reference image
    else if (option=="-ref")
    {
      if (i>=argc-1)
      {
        Cerr("***** castor-sumImages() -> Argument missing for option: " << option << endl);
        Exit(EXIT_FAILURE);
      }
      path_to_reference_image_filename = (string)argv[i+1];
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
        Cerr("***** castor-sumImages() -> Argument missing for option: " << option << endl);
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
        Cerr("***** castor-sumImages() -> Argument missing for option: " << option << endl);
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
      Cerr("***** castor-sumImages() -> Unknown option '" << option << "' !" << endl);
      Exit(EXIT_FAILURE);
    }
  }

  // ============================================================================================================
  // Some checks
  // ============================================================================================================

  // Number of images
  if (nb_images < 2)
  {
    Cerr("***** castor-sumImages() -> Please provide at least two image filenames !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Output files
  if (path_fout.empty() && path_dout.empty())
  {
    Cerr("***** castor-sumImages() -> Please provide an output option for output files (-fout or -dout) !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Check that only one option has been provided
  if (!path_fout.empty() && !path_dout.empty())
  {
    Cerr("***** castor-sumImages() -> Please provide either output option -fout or -dout but not both !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Check that a reference image is provided if bias, mse or rmse is required
  if (path_to_reference_image_filename=="" && (flag_bias || flag_rmse || flag_mse))
  {
    Cerr("***** castor-sumImages() -> Please provide a reference image for bias or (root) mean square error computation !" << endl);
    Exit(EXIT_FAILURE);
  }

  // ============================================================================================================
  // Set maximum number of threads if compiled with OpenMP
  // ============================================================================================================

  #ifdef CASTOR_OMP
  omp_set_num_threads(omp_get_max_threads());
  #endif

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
    Cerr("***** castor-sumImages() -> A problem occurred while initializing output directory !" << endl);
    Exit(EXIT_FAILURE);
  }
  // Log command line
  if (p_OutputManager->LogCommandLine(argc,argv))
  {
    Cerr("***** castor-sumImages() -> A problem occurred while logging command line arguments !" << endl);
    Exit(EXIT_FAILURE);
  }

  // ============================================================================================================
  // Read all input images
  // ============================================================================================================

  // Verbose
  if (verbose>=VERBOSE_LIGHT) Cout("castor-sumImages() -> Load input images" << endl);

  // Get user endianness (interfile I/O)
  GetUserEndianness();

  // Create interfile image fields and image pointers
  Intf_fields** p_image_fields = (Intf_fields**)malloc(nb_images*sizeof(Intf_fields*));
  FLTNB** p_images = (FLTNB**)malloc(nb_images*sizeof(FLTNB*));

  // Read interfile images
  for (uint32_t img=0; img<nb_images; img++)
  {
    // Create the image fields
    p_image_fields[img] = new Intf_fields;
    // Read the image
    p_images[img] = IntfLoadImageFromScratch(path_to_image_filename[img],p_image_fields[img],verbose);
    // Check reading
    if (p_images[img]==NULL)
    {
      Cerr("***** castor-sumImages() -> A problem occurred while reading image number " << img+1 << " !" << endl);
      return 1;
    }
  }

  // Check dimensions consistency between all image fields
  for (uint32_t img=1; img<nb_images; img++)
  {
    if (IntfCheckDimensionsConsistency(*p_image_fields[0],*p_image_fields[img]))
    {
      Cerr("***** castor-sumImages() -> Inconsistency between image dimensions !" << endl);
      Exit(EXIT_FAILURE);
    }
  }

  // Verbose
  if (verbose>=VERBOSE_NORMAL)
  {
    Cout("  --> Image dimensions: [ " << p_image_fields[0]->mtx_size[0] << " : "
                                      << p_image_fields[0]->mtx_size[1] << " : "
                                      << p_image_fields[0]->mtx_size[2] << " ]" << endl);
    Cout("  --> Voxel sizes: [ " << p_image_fields[0]->vox_size[0] << " : "
                                 << p_image_fields[0]->vox_size[1] << " : "
                                 << p_image_fields[0]->vox_size[2] << " ]" << endl);
  }

  // Get dimensions locally
  uint32_t dim_x = p_image_fields[0]->mtx_size[0];
  uint32_t dim_y = p_image_fields[0]->mtx_size[1];
  uint32_t dim_z = p_image_fields[0]->mtx_size[2];
  uint32_t dim_xyz = dim_x * dim_y * dim_z;

  // ============================================================================================================
  // If a reference image is provided, then load it
  // ============================================================================================================

  // Create the reference image matrix
  FLTNB* p_reference = NULL;

  // If we have a reference
  if (path_to_reference_image_filename!="")
  {
    // Verbose
    if (verbose>=1) Cout("castor-sumImages() -> Load reference image" << endl);
    // Create interfile image fields and image pointers for sensitivity
    Intf_fields* p_reference_fields = new Intf_fields;
    // Read the image
    p_reference = IntfLoadImageFromScratch(path_to_reference_image_filename,p_reference_fields,verbose);
    // Check reading
    if (p_reference==NULL)
    {
      Cerr("***** castor-sumImages() -> A problem occurred while reading reference image !" << endl);
      return 1;
    }
    // Check dimensions consistency between input images and reference image
    if (IntfCheckDimensionsConsistency(*p_image_fields[0],*p_reference_fields))
    {
      Cerr("***** castor-sumImages() -> Inconsistency between input images and reference image !" << endl);
      Exit(EXIT_FAILURE);
    }
  }

  // ============================================================================================================
  // Images allocations
  // ============================================================================================================

  // Usefull images
  FLTNB* p_mean = NULL;
  FLTNB* p_misc = NULL;

  // The mean image is mandatory in all cases expect for (r)mse
  if (flag_sum || flag_mean || flag_bias || flag_stdv || flag_var)
    p_mean = (FLTNB*)malloc(dim_xyz*sizeof(FLTNB));

  // The miscellaneous image is used for all other cases than sum/mean
  if (flag_bias || flag_mse || flag_rmse || flag_stdv || flag_var)
    p_misc = (FLTNB*)malloc(dim_xyz*sizeof(FLTNB));

  // ============================================================================================================
  // Compute the sum/mean image
  // ============================================================================================================

  // The mean image is computed in all cases expect for rmse
  if (flag_sum || flag_mean || flag_bias || flag_stdv || flag_var)
  {
    // Verbose
    if (verbose>=1) Cout("castor-sumImages() -> Compute sum/average image" << endl);
    // Reset the image
    for (uint32_t v=0; v<dim_xyz; v++) p_mean[v] = 0.;
    // Loop on all voxels
    uint32_t v;
    #pragma omp parallel for private(v) schedule(guided)
    for (v=0; v<dim_xyz; v++)
    {
      // High precision computation
      HPFLTNB sum = 0.;
      // Loop on all images
      for (uint32_t img=0; img<nb_images; img++)
      {
        // Add the contribution of this image to the sum image
        sum += ((HPFLTNB)(p_images[img][v]));
      }
      // Affect the value
      p_mean[v] = sum;
    }
    // Save sum image if required
    if (flag_sum)
    {
      // Verbose
      if (verbose>=1) Cout("castor-sumImages() -> Write sum image" << endl);
      // File name
      string sum_image_filename = p_OutputManager->GetPathName() + p_OutputManager->GetBaseName() + "_sum";
      // Write the image
      if (IntfWriteImageFromIntfFields(sum_image_filename, p_mean, *p_image_fields[0], verbose))
      {
        Cerr("***** castor-sumImages() -> A problem occurred while writing sum image !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
    // Transform the sum image into the mean image
    #pragma omp parallel for private(v) schedule(guided)
    for (v=0; v<dim_xyz; v++)
    {
      p_mean[v] /= ((FLTNB)nb_images);
    }
    // Save mean image if required
    if (flag_mean)
    {
      // Verbose
      if (verbose>=1) Cout("castor-sumImages() -> Write average image" << endl);
      // File name
      string mean_image_filename = p_OutputManager->GetPathName() + p_OutputManager->GetBaseName() + "_avg";
      // Write the image
      if (IntfWriteImageFromIntfFields(mean_image_filename, p_mean, *p_image_fields[0], verbose))
      {
        Cerr("***** castor-sumImages() -> A problem occurred while writing average image !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
  }

  // ============================================================================================================
  // Compute the variance/standard-deviation image
  // ============================================================================================================

  if (flag_var || flag_stdv)
  {
    // Verbose
    if (verbose>=1) Cout("castor-sumImages() -> Compute variance image" << endl);
    // Reset the image
    for (uint32_t v=0; v<dim_xyz; v++) p_misc[v] = 0.;
    // Loop on all voxels
    uint32_t v;
    #pragma omp parallel for private(v) schedule(guided)
    for (v=0; v<dim_xyz; v++)
    {
      // High precision computation
      HPFLTNB sum_square = 0.;
      // Loop on all images
      for (uint32_t img=0; img<nb_images; img++)
      {
        // Add the contribution of this image to the variance image
        HPFLTNB diff = ((HPFLTNB)(p_images[img][v])) - ((HPFLTNB)(p_mean[v]));
        sum_square += diff * diff;
      }
      // Divide by the number of images
      p_misc[v] = (sum_square / ((HPFLTNB)nb_images));
    }
    // Save variance image if required
    if (flag_var)
    {
      // Verbose
      if (verbose>=1) Cout("castor-sumImages() -> Write variance image" << endl);
      // File name
      string var_image_filename = p_OutputManager->GetPathName() + p_OutputManager->GetBaseName() + "_var";
      // Write the image
      if (IntfWriteImageFromIntfFields(var_image_filename, p_misc, *p_image_fields[0], verbose))
      {
        Cerr("***** castor-sumImages() -> A problem occurred while writing variance image !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
    // Compute standard deviation if required
    if (flag_stdv)
    {
      // Verbose
      if (verbose>=1)
      {
        if (flag_norm) Cout("castor-sumImages() -> Compute normalized standard-deviation image" << endl);
        else Cout("castor-sumImages() -> Compute standard-deviation image" << endl);
      }
      // Loop on all voxels
      #pragma omp parallel for private(v) schedule(guided)
      for (v=0; v<dim_xyz; v++)
      {
        // Apply square root
        p_misc[v] = sqrt(p_misc[v]);
        // Normalize if required
        if (flag_norm) p_misc[v] /= p_mean[v];
      }
      // Verbose
      if (verbose>=1) Cout("castor-sumImages() -> Write standard deviation image" << endl);
      // File name
      string stdv_image_filename = p_OutputManager->GetPathName() + p_OutputManager->GetBaseName() + "_stdv";
      // Write the image
      if (IntfWriteImageFromIntfFields(stdv_image_filename, p_misc, *p_image_fields[0], verbose))
      {
        Cerr("***** castor-sumImages() -> A problem occurred while writing standard deviation image !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
  }

  // ============================================================================================================
  // Compute the bias image
  // ============================================================================================================

  if (flag_bias)
  {
    // Verbose
    if (verbose>=1)
    {
      if (flag_norm) Cout("castor-sumImages() -> Compute normalized bias image" << endl);
      else Cout("castor-sumImages() -> Compute bias image" << endl);
    }
    // Reset the image
    for (uint32_t v=0; v<dim_xyz; v++) p_misc[v] = 0.;
    // Loop on all voxels
    uint32_t v;
    #pragma omp parallel for private(v) schedule(guided)
    for (v=0; v<dim_xyz; v++)
    {
      // Simply the mean minus the reference
      p_misc[v] = p_mean[v] - p_reference[v];
      // Normalize if required
      if (flag_norm) p_misc[v] /= p_reference[v];
    }
    // Verbose
    if (verbose>=1) Cout("castor-sumImages() -> Write bias image" << endl);
    // File name
    string bias_image_filename = p_OutputManager->GetPathName() + p_OutputManager->GetBaseName() + "_bias";
    // Write the image
    if (IntfWriteImageFromIntfFields(bias_image_filename, p_misc, *p_image_fields[0], verbose))
    {
      Cerr("***** castor-sumImages() -> A problem occurred while writing bias image !" << endl);
      Exit(EXIT_FAILURE);
    }
  }

  // ============================================================================================================
  // Compute the (R)MSE image
  // ============================================================================================================

  if (flag_rmse || flag_mse)
  {
    // Verbose
    if (verbose>=1) Cout("castor-sumImages() -> Compute mean square error image" << endl);
    // Reset the image
    for (uint32_t v=0; v<dim_xyz; v++) p_misc[v] = 0.;
    // Loop on all voxels
    uint32_t v;
    #pragma omp parallel for private(v) schedule(guided)
    for (v=0; v<dim_xyz; v++)
    {
      // High precision computation
      HPFLTNB sum_square = 0.;
      // Loop on all images
      for (uint32_t img=0; img<nb_images; img++)
      {
        // Add the contribution of this image to the rmse image
        HPFLTNB diff = ((HPFLTNB)(p_images[img][v])) - ((HPFLTNB)(p_reference[v]));
        sum_square = diff * diff;
      }
      // Divide by the number of images
      p_misc[v] = (sum_square / ((HPFLTNB)nb_images));
    }
    // Write MSE if required
    if (flag_mse)
    {
      // Verbose
      if (verbose>=1) Cout("castor-sumImages() -> Write mean square error image" << endl);
      // File name
      string mse_image_filename = p_OutputManager->GetPathName() + p_OutputManager->GetBaseName() + "_mse";
      // Write the image
      if (IntfWriteImageFromIntfFields(mse_image_filename, p_misc, *p_image_fields[0], verbose))
      {
        Cerr("***** castor-sumImages() -> A problem occurred while writing mean square error image !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
    // Compute RMSE if required
    if (flag_rmse)
    {
      // Verbose
      if (verbose>=1)
      {
        if (flag_norm) Cout("castor-sumImages() -> Compute normalized root mean square error image" << endl);
        else Cout("castor-sumImages() -> Compute root mean square error image" << endl);
      }
      // Loop on all voxels
      #pragma omp parallel for private(v) schedule(guided)
      for (v=0; v<dim_xyz; v++)
      {
        // Apply square root
        p_misc[v] = sqrt(p_misc[v]);
        // Normalize if required
        if (flag_norm) p_misc[v] /= p_reference[v];
      }
      // Verbose
      if (verbose>=1) Cout("castor-sumImages() -> Write root mean square error image" << endl);
      // File name
      string rmse_image_filename = p_OutputManager->GetPathName() + p_OutputManager->GetBaseName() + "_rmse";
      // Write the image
      if (IntfWriteImageFromIntfFields(rmse_image_filename, p_misc, *p_image_fields[0], verbose))
      {
        Cerr("***** castor-sumImages() -> A problem occurred while writing root mean square error image !" << endl);
        Exit(EXIT_FAILURE);
      }
    }
  }

  // ============================================================================================================
  // Free memory properly
  // ============================================================================================================

  // Input images
  if (p_images)
  {
    for (uint32_t img=0; img<nb_images; img++) if (p_images[img]) free(p_images[img]);
    free(p_images);
  }
  // Reference image
  if (p_reference) free(p_reference);
  // Mean image
  if (p_mean) free(p_mean);
  // Miscellaneous image
  if (p_misc) free(p_misc);

  // Ending
  #ifdef CASTOR_MPI
  MPI_Finalize();
  #endif
  return EXIT_SUCCESS;
}
