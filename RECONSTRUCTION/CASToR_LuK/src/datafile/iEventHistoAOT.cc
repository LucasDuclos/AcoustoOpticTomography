
/*!
  \file
  \ingroup datafile

  \brief Implementation of class iEventHistoAOT
*/

#include "vEvent.hh"
#include "iEventHistoAOT.hh"
#include "vDataFile.hh"
#include "sOutputManager.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iEventHistoAOT::iEventHistoAOT() : vEvent()
{
  m_dataType = TYPE_AOT;
  m_dataSpec = SPEC_EMISSION;
  m_dataMode = MODE_HISTOGRAM;
  mp_eventValue = NULL;
  m_eventNbTOFBins = 0;
  m_nbLines = 1;
  mp_USTransducersActivation = NULL;
  m_nbUSTransducers = 0;
  m_nbGroupsUSTransducers = 0;
  m_waveAngle = 0;
  m_additiveCorrections = 0.;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iEventHistoAOT::~iEventHistoAOT() 
{
  if (mp_eventValue != NULL) free(mp_eventValue);
  if (mp_USTransducersActivation != NULL) free(mp_USTransducersActivation);
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iEventHistoAOT::AllocateSpecificData()
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)

  // Check that the number of acquisitions is correct
  if (m_eventNbTOFBins<1)
  {
    Cerr("***** iEventHistoAOT::AllocateSpecificData() -> Number of 'TOF bins' has not been initialized (<1) !");
    return 1;
  }
  // Allocate values depending on the number of acquisitions
  mp_eventValue = (FLTNB*)malloc(m_eventNbTOFBins*sizeof(FLTNB));
  // Default initialization
  for(uint16_t tofbn=0 ; tofbn<m_eventNbTOFBins ; tofbn++)
  {
    mp_eventValue[tofbn] = 0.;
  }
  
  // Check that the number of US transducers is correct
  if (m_nbUSTransducers<1)
  {
    Cerr("***** iEventHistoAOT::AllocateSpecificData() -> Number of US transducers has not been initialized (<1) !");
    return 1;
  }
  // Allocate values depending on the number of acquisitions
  mp_USTransducersActivation = (uint8_t*)malloc(m_nbGroupsUSTransducers*sizeof(uint8_t));
  // Default initialization
  for(int tgn=0 ; tgn<m_nbGroupsUSTransducers ; tgn++)
  {
    mp_USTransducersActivation[tgn] = 0x00;
  }
  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

void iEventHistoAOT::Describe()
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  Cout("iEventHistoAOT::Describe() -> Display contents" << endl);
  Cout("Number of acquisitions: " << m_eventNbTOFBins << endl);
  for (uint16_t tofbn=0 ; tofbn<m_eventNbTOFBins ; tofbn++) Cout("  --> event number: " << tofbn << " | value: " << mp_eventValue[tofbn] << endl);
  Cout("Number of US transducers: " << m_nbUSTransducers << endl);
  for (int tgn=0; tgn<m_nbGroupsUSTransducers; tgn++) Cout("  --> transducer group number: " << tgn << " | value: " << setfill('0') << setw(sizeof(uint8_t)*2) << hex << static_cast<int>(mp_USTransducersActivation[tgn]) << dec << endl);//static_cast<int>(mp_USTransducersActivation[tgn])
  Cout("Wave angle: " << static_cast<int>(m_waveAngle) << endl);
  Cout("Additive correction: " << m_additiveCorrections << endl);
  Cout(flush);
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iEventHistoAOT::GetEventIdentifier(string& a_eventName)
{
	a_eventName = "";
	stringstream ss;
	for (int tgn=0; tgn<m_nbGroupsUSTransducers; tgn++) 
	{
		ss << setfill('0') << setw(sizeof(uint8_t)*2) << hex << static_cast<int>(mp_USTransducersActivation[tgn]) << dec;
	}
	ss << "_" << (m_waveAngle < 0);
	ss << setfill('0') << setw(sizeof(int8_t)*2) << abs(m_waveAngle);
	a_eventName.append(ss.str());
	return 0;
}
