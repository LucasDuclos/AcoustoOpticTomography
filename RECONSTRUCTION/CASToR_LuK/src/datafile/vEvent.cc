
/*!
  \file
  \ingroup datafile

  \brief Implementation of class vEvent
*/

#include "vEvent.hh"
#include "vDataFile.hh"
#include "sOutputManager.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

vEvent::vEvent()
{
  m_timeInMs = 0;
  m_nbLines = 0;
  mp_ID1 = NULL;
  mp_ID2 = NULL;
  m_dataType = TYPE_UNKNOWN;
  m_dataMode = MODE_UNKNOWN;
  m_verbose = -1;
  m_eventValue = 0.;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

vEvent::~vEvent() 
{
  if (mp_ID1 != NULL) free(mp_ID1);
  if (mp_ID1 != NULL) free(mp_ID2);
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int vEvent::AllocateID() 
{
  // Verbose
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  if (m_verbose>=VERBOSE_DETAIL) Cout("vEvent::AllocateID() -> Allocate buffers for indices" << endl);
  // Check number of lines
  if (m_nbLines<1)
  {
    Cerr("***** vEvent::AllocateID() -> Error, number of lines has not been initialized (<1) !" << endl);
    return 1;
  }
  else
  {
    // Allocate buffer indices
    mp_ID1 = (uint32_t*)malloc(m_nbLines*sizeof(uint32_t));
    mp_ID2 = (uint32_t*)malloc(m_nbLines*sizeof(uint32_t));
  }
  // Call the pure virtual function implemented in child classes for the allocation of data specific to child classes
  if (AllocateSpecificData())
  {
    Cerr("***** vEvent::AllocateID() -> Error when trying to allocated specific data for the Event !" << endl);
    return 1;
  }
  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
