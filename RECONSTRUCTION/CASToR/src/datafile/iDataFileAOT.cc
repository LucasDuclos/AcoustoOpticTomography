
/*!
  \file
  \ingroup datafile

  \brief Implementation of class iDataFileAOT
*/

#include "iDataFileAOT.hh"

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iDataFileAOT::iDataFileAOT() : vDataFile() 
{
  // Set all members to default values
  m_dataType = TYPE_AOT;
  m_dataSpec = SPEC_EMISSION;
  m_dataMode = MODE_HISTOGRAM;
  m_acquisitionFrequencyInHz = 0.;
  m_nbTOFBins = 0;
  m_nbUSTransducers = 0;
  m_nbGroupsUSTransducers = 0;
  m_hasAdditiveCorrections = false;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

iDataFileAOT::~iDataFileAOT() {;}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::ReadSpecificInfoInHeader(bool a_affectQuantificationFlag)
{
  // Verbose
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  if (m_verbose>=VERBOSE_DETAIL) Cout("iDataFileAOT::ReadSpecificInfoInHeader() -> Read information specific to AOT" << endl);
  // Read fields in the header, check if errors (issue during data reading/conversion (==1) )
  if (ReadDataASCIIFile(m_headerFileName, "Number of acquisitions per event", &m_nbTOFBins, 1, KEYWORD_MANDATORY)==1 ||
      ReadDataASCIIFile(m_headerFileName, "Acquisition frequency (Hz)", &m_acquisitionFrequencyInHz, 1, KEYWORD_MANDATORY)==1 ||
      ReadDataASCIIFile(m_headerFileName, "Number of US transducers", &m_nbUSTransducers, 1, KEYWORD_MANDATORY)==1 ||
      ReadDataASCIIFile(m_headerFileName, "Additive corrections", &m_hasAdditiveCorrections, 1, KEYWORD_OPTIONAL)==1)
  {
    Cerr("***** iDataFileAOT::ReadSpecificInfoInHeader() -> Error while reading optional fields in the header data file !" << endl);
    return 1;
  }

  // Set the number of groups of 8 US transducers
  m_nbGroupsUSTransducers = (m_nbUSTransducers+7)/8;

  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::SetSpecificParametersFrom(vDataFile* ap_DataFile)
{
  iDataFileAOT* p_DataFileAOT = (dynamic_cast<iDataFileAOT*>(ap_DataFile));
  m_acquisitionFrequencyInHz = p_DataFileAOT->GetAcquisitionFrequencyInHz();		
  m_nbTOFBins = p_DataFileAOT->GetNbTOFBins();
  m_nbUSTransducers = p_DataFileAOT->GetNbUSTransducers();
  m_nbGroupsUSTransducers = p_DataFileAOT->GetNbGroupsUSTransducers();
  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::ComputeSizeEvent()
{
  // Verbose
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  if (m_verbose>=VERBOSE_DETAIL) Cout("iDataFileAOT::ComputeSizeEvent() -> In bytes" << endl);

  // Size of the acquisitions of the event
  m_sizeEvent = m_nbTOFBins*sizeof(FLTNB);
  // Size of the US transducers activation list, which may be zero-padded
  m_sizeEvent += m_nbGroupsUSTransducers*sizeof(uint8_t);
  // Size of wave angle
  m_sizeEvent += sizeof(int8_t);

  if (m_hasAdditiveCorrections)
  {
	  m_sizeEvent += sizeof(FLTNB);
  }

  // Check
  if (m_sizeEvent<=0) 
  {
    Cerr("***** iDataFileAOT::ComputeSizeEvent() -> Error, the Event size in bytes should be >= 0 !" << endl;);
    return 1;
  }

  // Verbose
  if (m_verbose>=VERBOSE_DETAIL) Cout("  --> Event size = " << m_sizeEvent << " bytes" << endl);
  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::PrepareDataFile()
{
  // Verbose
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)

  // ==============================================================================
  // Allocate event buffers (one for each thread)
  // ==============================================================================
  if (m_verbose>=VERBOSE_DETAIL) Cout("  --> Allocating an event buffer for each thread" << endl);
  // Instanciation of the event buffer according to the data type
  m2p_BufferEvent = new vEvent*[mp_ID->GetNbThreadsForProjection()];
  
  // Allocate the events per each thread
  for (int th=0 ; th<mp_ID->GetNbThreadsForProjection() ; th++)
  {
	m2p_BufferEvent[th] = new iEventHistoAOT();
	((iEventHistoAOT*)m2p_BufferEvent[th])->SetEventNbTOFBins(m_nbTOFBins);
  }

  // Normal end
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

vEvent* iDataFileAOT::GetEventSpecific(char* ap_buffer, int a_th)
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_EVENT)

  // Work on a copy of the input pointer
  char* file_position = ap_buffer;
  
  // Cast the event pointer
  iEventHistoAOT* event = (dynamic_cast<iEventHistoAOT*>(m2p_BufferEvent[a_th]));
  
  // Number of US transducers and number of acquisitions
  event->SetNbUSTransducers(m_nbUSTransducers);
  event->SetNbGroupsUSTransducers(m_nbGroupsUSTransducers);
  event->SetEventNbTOFBins((uint16_t) m_nbTOFBins);

  event->AllocateSpecificData();
  
  // Activation list of US transducers
  for (int tgn=0 ; tgn<m_nbGroupsUSTransducers ; tgn++)
  {
    event->SetUSTransducersActivation(tgn, *reinterpret_cast<uint8_t*>(file_position));
    file_position += sizeof(uint8_t);
  }
  
  // Angle of the wave
  event->SetWaveAngle(*reinterpret_cast<int8_t*>(file_position));
  file_position += sizeof(int8_t);
  
  // "TOF bins"
  for (int tofbn=0 ; tofbn<m_nbTOFBins ; tofbn++)
  {
    event->SetEventValue(tofbn, *reinterpret_cast<FLTNB*>(file_position));
    file_position += sizeof(FLTNB);
  }

  if (m_hasAdditiveCorrections)
  {
	event->SetAdditiveCorrections(*reinterpret_cast<FLTNB*>(file_position));
	file_position += sizeof(FLTNB);
  }

  // Return the updated event
  return m2p_BufferEvent[a_th];
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

void iDataFileAOT::DescribeSpecific()
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  if (m_verbose==0) return;
  // Describe the datafile
  Cout("iDataFileAOT::DescribeSpecific() -> Here is some specific content of the AOT datafile" << endl);
  Cout("  --> Acquisition frequency (Hz): " << m_acquisitionFrequencyInHz << endl);
  Cout("  --> Number of 'TOF bins': " << m_nbTOFBins << endl);
  Cout("  --> Number of transducers: " << m_nbUSTransducers << endl);
  Cout("  --> Number of groups of 8 transducers: " << m_nbGroupsUSTransducers << endl);
  Cout("  --> Number of TOF bins: " << m_nbTOFBins << endl);
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::CheckSpecificParameters()
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  // Error if m_dataType != AOT
  if (m_dataType != TYPE_AOT)
  {
    Cerr("***** iDataFileAOT::CheckSpecificParameters() -> Data type should be AOT !'" << endl);
    return 1;
  }
  // Check the provided acquisition frequency
  if (m_acquisitionFrequencyInHz <= 0.)
  {
    Cerr("***** iDataFileAOT::CheckSpecificParameters() -> Acquisition frequency is not positive!'" << endl);
    return 1;
  }
  // Check the number of acquisitions per event
  if (m_nbTOFBins == 0)
  {
    Cerr("***** iDataFileAOT::CheckSpecificParameters() -> Number of 'TOF bins' is equal to zero!'" << endl);
    return 1;
  }
  // Check the number of US transducers
  if (m_nbUSTransducers == 0)
  {
    Cerr("***** iDataFileAOT::CheckSpecificParameters() -> Number of US transducers is equal to zero!'" << endl);
    return 1;
  }
  // Check the number of groups of 8 US transducers
  if (m_nbGroupsUSTransducers == 0)
  {
    Cerr("***** iDataFileAOT::CheckSpecificParameters() -> Number of groups of US transducers is equal to zero!'" << endl);
    return 1;
  }
  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::CheckFileSizeConsistency()
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)

  // Create the stream
  fstream* p_file = new fstream( m_dataFileName.c_str(), ios::binary| ios::in );
  // Check that datafile exists
  if (!p_file->is_open()) 
  {
    Cerr("***** iDataFileAOT::CheckFileSizeConsistency() -> Failed to open input file '" << m_dataFileName.c_str() << "' !" << endl);
    Cerr("                                                 (Provided in the data file header: " << m_headerFileName << ")" << endl);
    return 1;
  }
  // Get file size in bytes
  p_file->seekg(0, ios::end);
  int64_t sizeInBytes = p_file->tellg();
  // Close stream and delete it
  p_file->close();
  delete p_file;
  // Check datafile self-consistency
  if (m_nbEvents*m_sizeEvent != sizeInBytes)
  {
    Cerr("--------------------------------------------------------------------------------------------------------------------------------------" << endl);
    Cerr("***** iDataFileAOT::CheckFileSizeConsistency() -> DataFile size is not consistent with the information provided by the user/datafile !" << endl);
    Cerr("  --> Expected size : "<< m_nbEvents*m_sizeEvent << endl);
    Cerr("  --> Actual size : "<< sizeInBytes << endl << endl);
    // Cerr("      ADDITIONAL INFORMATION ABOUT THE DATAFILE INITIALIZATION : " << endl);
    return 1;
  }
  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::CheckSpecificConsistencyWithAnotherDataFile(vDataFile* ap_DataFile)
{
  DEBUG_VERBOSE(m_verbose,VERBOSE_DEBUG_LIGHT)
  // Dynamic cast the vDataFile to a iDataFileAOT
  iDataFileAOT* p_data_file = (dynamic_cast<iDataFileAOT*>(ap_DataFile));
  // Check acquisition frequency
  if (m_acquisitionFrequencyInHz!=p_data_file->GetAcquisitionFrequencyInHz())
  {
    Cerr("***** iDataFileAOT::CheckSpecificConsistencyWithAnotherDataFile() -> Acquisition frequencies is inconsistent !" << endl);
    return 1;
  }
  // Check number of acquisitions per event
  if (m_nbTOFBins!=p_data_file->GetNbTOFBins())
  {
    Cerr("***** iDataFileAOT::CheckSpecificConsistencyWithAnotherDataFile() -> Numbers of acquisitions per event is inconsistent !" << endl);
    return 1;
  }
  // Check number of US transducers
  if (m_nbUSTransducers!=p_data_file->GetNbUSTransducers())
  {
    Cerr("***** iDataFileAOT::CheckSpecificConsistencyWithAnotherDataFile() -> Numbers of US transducers are inconsistent !" << endl);
    return 1;
  }
  // Check number of groups of US transducers
  if (m_nbUSTransducers!=p_data_file->GetNbGroupsUSTransducers())
  {
    Cerr("***** iDataFileAOT::CheckSpecificConsistencyWithAnotherDataFile() -> Numbers of groups of US transducers are inconsistent !" << endl);
    return 1;
  }
  // End
  return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::PROJ_InitFile()
{
  Cerr("***** iDataFileAOT::PROJ_InitFile() -> error, function not implemented yet !" << endl);
  return 1;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::WriteEvent(vEvent* ap_Event, int a_th)
{
  Cerr("***** iDataFileAOT::WriteEvent() -> error, function not implemented yet !" << endl);
  return 1;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::WriteHistoEvent(iEventHistoAOT* ap_Event, int a_th)
{
  Cerr("***** iDataFileAOT::WriteHistoEvent() -> error, function not implemented yet !" << endl);
  return 1;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::WriteHeader()
{
  
  Cerr("***** iDataFileAOT::WriteHeader() -> error, function not implemented yet !" << endl);
  return 1;
  
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int iDataFileAOT::PROJ_GetScannerSpecificParameters()
{
  Cerr("***** iDataFileAOT::PROJ_GetScannerSpecificParameters() -> error, function not implemented yet !" << endl);
  return 1;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
