
/*!
  \file
  \ingroup  image
  \brief    Implementation of class iImageConvolverTemplate
*/

#include "vImageConvolver.hh"
#include "oImageDimensionsAndQuantification.hh"
#include "oImageSpace.hh"
#include "iImageConvolverTemplate.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iImageConvolverTemplate::iImageConvolverTemplate() : vImageConvolver()
{
  // Affect default values to the parameters specific to this convolver
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iImageConvolverTemplate::~iImageConvolverTemplate()
{
  // Delete or free all structures specific to this convolver that were allocated by this convolver
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iImageConvolverTemplate::ReadConfigurationFile(const string& a_configurationFile)
{
  // Implement here the reading of any options specific to this image convolver, through a configuration file

  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iImageConvolverTemplate::ReadOptionsList(const string& a_optionsList)
{
  // Implement here the reading of any options specific to this image convolver, through a list of options separated by commas

  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

void iImageConvolverTemplate::ShowHelp()
{
  // Here, display some help and guidance to how to use this image convolver module and what it does.
  // Also describes the form of the configuration file or the list of options that parameterize your module.
  cout << "This image convolver is a template class dedicated to add your own custom image convolver." << endl;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iImageConvolverTemplate::CheckSpecificParameters()
{
  // Implement here all mandatory checks specific to this image convolver needed for a nice use

  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iImageConvolverTemplate::BuildConvolutionKernel()
{
  // Here you must build the convolution kernels. This means first specifying the number of kernels: m_nbKernels.
  // If it is stationary (i.e. m_nbKernels=1), then the m_stationary boolean should be set to true, as the Convolve
  // function implemented by the mother vImageConvolver will be used. Otherwise, this function Convolve(), as well
  // as ConvolveTranspose() must be overloaded by this class in order to properly use the convolution kernels.
  // Second, the dimensions of the kernels mp_dimKernelX, mp_dimKernelY and mp_dimKernelZ must first be allocated
  // with respect to the number of kernels, and then be specified. Finally, the kernels values m2p_kernel must also be
  // allocated with respect to the number of kernels and their dimensions, and then be specified. It is assumed that
  // the normalization of the kernels is made in this function.

  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
