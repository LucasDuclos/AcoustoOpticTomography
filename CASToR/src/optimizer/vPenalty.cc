
/*!
  \file
  \ingroup  optimizer
  \brief    Implementation of class vPenalty
*/

#include "vPenalty.hh"
#include "sOutputManager.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

vPenalty::vPenalty()
{
  // Affect default values
  m_penaltyID = "";
  mp_ImageDimensionsAndQuantification = NULL;
  mp_ImageSpace = NULL;
  m_verbose = 0;
  m_penaltyDerivativesOrder = 0;
  m_penaltyStrength = -1.;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

vPenalty::~vPenalty()
{
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

void vPenalty::ShowHelp()
{
  // Call the specific help function from the children
  ShowHelpSpecific();
}


// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int vPenalty::CheckParameters()
{
  // Check image dimensions
  if (mp_ImageDimensionsAndQuantification==NULL)
  {
    Cerr("***** vPenalty::CheckParameters() -> oImageDimensionsAndQuantification is null !" << endl);
    return 1;
  }
  // Check image space
  if (mp_ImageSpace==NULL)
  {
    Cerr("***** vPenalty::CheckParameters() -> oImageSpace is null !" << endl);
    return 1;
  }
  // Check verbose level
  if (m_verbose<0)
  {
    Cerr("***** vPenalty::CheckParameters() -> Verbose level is negative !" << endl);
    return 1;
  }
  // Check penalty strength
  if (m_penaltyStrength<0.)
  {
    Cerr("***** vPenalty::CheckParameters() -> Penalty strength is negative or unset !" << endl);
    return 1;
  }
  // Call the CheckSpecificParameters function of the child
  if (CheckSpecificParameters())
  {
    Cerr("***** vPenalty::CheckParameters() -> A problem occured while checking parameters specific to the penalty module !" << endl);
    return 1;
  }
  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int vPenalty::Initialize()
{
  // Verbose
  if (m_verbose>=VERBOSE_NORMAL)
  {
    Cout("vPenalty::Initialize() -> Initialize the penalty with strength (beta): " << m_penaltyStrength << endl);
  }
  // Call the specific initialization function of the child
  if (InitializeSpecific())
  {
    Cerr("***** vPenalty::Initialize() -> A problem occured while initializing stuff specific to the optimizer module !" << endl);
    return 1;
  }
  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int vPenalty::GlobalPreProcessingStep()
{
  // This function from the mother class intentionnaly does nothing so that it can be overloaded
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int vPenalty::LocalPreProcessingStep(int a_tbf, int a_rbf, int a_cbf, INTNB a_voxel, int a_th)
{
  // This function from the mother class intentionnaly does nothing so that it can be overloaded
  return 0;
}
