
/*!
  \file
  \ingroup  projector
  \brief    Implementation of class oSystemMatrix
*/

#include "oSystemMatrix.hh"
#include "vScanner.hh"
#include "vEvent.hh"
#include "vDataFile.hh"
#include "iDataFileAOT.hh"
#include <sys/time.h>
#include <sys/resource.h>

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

oSystemMatrix::oSystemMatrix()
{
  // Default all members
  m_compatibleWithSPECTAttenuationCorrection = false;
  map<pair<int,int>,FLTNB**> m3p_voxelWeightsMap;
  map<string, int> mp_identifierMap;
  m_nbTOFBins = 0;

}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

oSystemMatrix::~oSystemMatrix()
{
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int oSystemMatrix::Project(int a_direction, oProjectionLine* ap_ProjectionLine, string a_eventIdentifier)
{
	ap_ProjectionLine->AddLine(a_direction, m_nbTOFBins, m2p_voxelWeightsMap[{mp_identifierMap[a_eventIdentifier],0}]);
	return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================

int oSystemMatrix::LoadData(vDataFile* ap_DataFile, INTNB nbVoxels, string& a_pathToSystemMatrixDirectory)
{
	// Check the provided parameters
	if (ap_DataFile->GetDataType() != TYPE_AOT)
	{
		Cerr("***** oSystemMatrix::LoadSystemMatrix -> Only works with AOT !");
		return 1;
	}
	iDataFileAOT* p_DataFileAOT = (static_cast<iDataFileAOT*>(ap_DataFile));

	int64_t nbEvents = p_DataFileAOT->GetSize();
	m_nbTOFBins = p_DataFileAOT->GetNbTOFBins();
	m_nbVoxels = nbVoxels;
	// Check the dimensions
	if (nbEvents < 1)
	{
		Cerr("***** oSystemMatrix::LoadSystemMatrix() -> The datafile size is incorrect (<1)!");
		return 1;
	}
	if (m_nbTOFBins < 1)
	{
		Cerr("***** oSystemMatrix::LoadSystemMatrix() -> The number of acquisitions per event is incorrect (<1)!");
		return 1;
	}
	if (m_nbVoxels < 1)
	{
		Cerr("***** oSystemMatrix::LoadSystemMatrix() -> The number of voxels is incorrect (<1)!");
		return 1;
	}

	// Check if the maximum number of simultaneously opened files is greater than the nb of events (+ a safety margin)
	struct rlimit open_file_limit;
	getrlimit(RLIMIT_NOFILE, &open_file_limit);
	uint64_t desired_open_file_limit = nbEvents+30;
	if (open_file_limit.rlim_cur < desired_open_file_limit)
	{
		// Check if it is possible to set the maximum number of simultaneously opened files at the nb of events (+ a safety margin)
		if (open_file_limit.rlim_max < desired_open_file_limit)
		{
			Cerr("***** oSystemMatrix::LoadSystemMatrix() -> The maximum number of simultaneously opened files is too low!");
			return 1;
		}
		open_file_limit.rlim_cur = desired_open_file_limit;
		if (setrlimit(RLIMIT_NOFILE, &open_file_limit))
		{
			Cerr("***** oSystemMatrix::LoadSystemMatrix() -> An error occurred while setting the new maximum number of simultaneously opened files!");
			return 1;
		}
	}

	for (int64_t en=0 ; en<nbEvents; en++)
	{
		iEventHistoAOT* event = static_cast<iEventHistoAOT*>(p_DataFileAOT->GetEvent(en, 0));
		string eventIdentifier; event->GetEventIdentifier(eventIdentifier);
		// Add entry to map
		mp_identifierMap[eventIdentifier] = static_cast<int>(en);

		// Check if the line exists and its size
		string pathToLine = a_pathToSystemMatrixDirectory + "field_" + eventIdentifier + ".img";
		ifstream is (pathToLine, ifstream::binary);
		if (is)
		{
			is.seekg (0, is.end);
			int length = is.tellg();
			if (length < 0 || length != m_nbTOFBins*m_nbVoxels*((int)sizeof(FLTNB)))
			{
				Cerr("***** oSystemMatrix::LoadSystemMatrix -> The size of " << pathToLine << " is different from what is expected!" << " Time: " << m_nbTOFBins <<" Nb Voxel :" << m_nbVoxels << ((int)sizeof(FLTNB)));
				return 1;
			}
			is.close();
		}
		else
		{
			Cerr("***** oSystemMatrix::LoadSystemMatrix() -> " << pathToLine << " does not exist!" << endl);
			return 1;
		}

		// Create mapped file
		m2p_mappedFiles[{en,0}] = new oMemoryMapped();
		// Open the file
		if (m2p_mappedFiles[{en,0}]->Open(pathToLine.c_str(), oMemoryMapped::WholeFile, oMemoryMapped::Normal))
		{
			Cerr("***** oSystemMatrix::LoadSystemMatrix() -> Failed to open data file '" << pathToLine << "'!" << endl);
			return 1;
		}
		// Get raw pointer to mapped memory
		m2p_voxelWeightsMap[{en,0}] = (char*)m2p_mappedFiles[{en,0}]->GetData();
	}
	return 0;
}

// =====================================================================
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// =====================================================================
