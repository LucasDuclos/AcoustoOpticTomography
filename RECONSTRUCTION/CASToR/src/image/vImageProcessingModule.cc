
/*!
  \file
  \ingroup  image
  \brief    Implementation of class vImageProcessingModule
*/

#include "vImageProcessingModule.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

vImageProcessingModule::vImageProcessingModule()
{
  // Affect default values
  mp_ImageDimensionsAndQuantification = NULL;
  m_verbose = 0;
  m_affectTimeDimensionFlag = false;
  m_affectRespDimensionFlag = false;
  m_affectCardDimensionFlag = false;
  m_checked = false;
  m_initialized = false;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

vImageProcessingModule::~vImageProcessingModule()
{
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int vImageProcessingModule::CheckParameters()
{
  // Check image dimensions
  if (mp_ImageDimensionsAndQuantification==NULL)
  {
    Cerr("***** vImageProcessingModule::CheckParameters() -> oImageDimensionsAndQuantification is null !" << endl);
    return 1;
  }
  // Check verbose level
  if (m_verbose<0)
  {
    Cerr("***** vImageProcessingModule::CheckParameters() -> Verbose level is negative !" << endl);
    return 1;
  }
  // Call the CheckSpecificParameters function of the child
  if (CheckSpecificParameters())
  {
    Cerr("***** vImageProcessingModule::CheckParameters() -> A problem occurred while checking parameters specific to the image processing module !" << endl);
    return 1;
  }
  // All set
  m_checked = true;
  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int vImageProcessingModule::Initialize()
{
  // Call the specific initialization function of the child
  if (InitializeSpecific())
  {
    Cerr("***** vImageProcessingModule::Initialize() -> A problem occurred while initializing stuff specific to the image processing module !" << endl);
    return 1;
  }
  // All set
  m_initialized = true;
  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
