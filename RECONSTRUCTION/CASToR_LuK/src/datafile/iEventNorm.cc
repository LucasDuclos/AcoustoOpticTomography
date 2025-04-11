
/*!
  \file
  \ingroup datafile

  \brief Implementation of class iEventNorm
*/

#include "iEventNorm.hh"
#include "vDataFile.hh"
#include "sOutputManager.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iEventNorm::iEventNorm() : vEvent() 
{
  m_dataMode = MODE_NORMALIZATION;
  m_normalizationFactor = 1.;
  m_attenuationCorrectionFactor = 1.;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iEventNorm::~iEventNorm() {}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

void iEventNorm::Describe()
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  Cout("iEventNorm::Describe() -> Display contents" << endl);
  Cout("Number of lines: " << m_nbLines << endl);
  for (uint16_t l=0; l<m_nbLines; l++) Cout("  --> ID1: " << mp_ID1[l] << " | ID2: " << mp_ID2[l] << endl);
  Cout("Attenuation correction factor: " << m_attenuationCorrectionFactor << endl);
  Cout("Normalization factor: " << m_normalizationFactor << endl);
  Cout(flush);
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

FLTNB iEventNorm::GetEventValue(int a_bin)
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  // This function is an implementation of inheritance, but has no sense in this context
  Cerr("***** iEventNorm::GetEventValue() -> This function should not be used ! Alternatives are:" << endl);
  Cerr("                                     GetNormalizationFactor() and GetAttenuationCorrectionFactor" << endl);
  Exit(EXIT_FAILURE);
  return -1.;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

FLTNB iEventNorm::GetAdditiveCorrections(int a_bin)
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  // This function is an implementation of inheritance, but has no sense in this context
  Cerr("***** iEventNorm::GetAdditiveCorrections() -> This function should not be used ! Alternatives are:" << endl);
  Cerr("                                              GetNormalizationFactor() and GetAttenuationCorrectionFactor" << endl);
  Exit(EXIT_FAILURE);
  return -1.;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

void iEventNorm::SetEventValue(int a_bin, FLTNBDATA a_value)
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  // This function is an implementation of inheritance, but has no sense in this context
  Cerr("***** iEventNorm::GetAdditiveCorrections() -> This function should not be used ! Alternatives are:" << endl);
  Cerr("                                              GetNormalizationFactor() and GetAttenuationCorrectionFactor" << endl);
  Exit(EXIT_FAILURE);
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

INTNB iEventNorm::GetNbValueBins()
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  // This function is an implementation of inheritance, but has no sense in this context
  Cerr("***** iEventNorm::GetNbValueBins() -> This function should not be used ! Alternatives are:" << endl);
  Cerr("                                              GetNormalizationFactor() and GetAttenuationCorrectionFactor" << endl);
  Exit(EXIT_FAILURE);
  return -1;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

void iEventNorm::MultiplyAdditiveCorrections(FLTNB a_factor)
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  // This function is an implementation of inheritance, but has no sense in this context
  Cerr("***** iEventNorm::MultiplyAdditiveCorrections() -> This function should not be used !" << endl);
  Exit(EXIT_FAILURE);
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
