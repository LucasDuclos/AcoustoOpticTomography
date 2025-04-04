
/*!
  \file
  \ingroup  projector
  \brief    Declaration of class oSystemMatrix
*/

#ifndef OSYSTEMMATRIX_HH
#define OSYSTEMMATRIX_HH 1

#include "gVariables.hh"
#include "oProjectionLine.hh"
#include "vScanner.hh"
#include "vEvent.hh"
#include "vDataFile.hh"
#include "oMemoryMapped.hh"

/**
 * \ingroup projector
 * @defgroup SYSTEM_MATRIX_KEYWORD System matrix keyword
 *
 *    \brief Keyword for a pre-computed and loaded system matrix \n
 *           Defined in oSystemMatrix.hh
 */
/**@{*/
/** String constant corresponding to a key word used as a projector
    name for pre-computed and loaded system matrix instead of using
    an on-the-fly projector */
#define SYSTEM_MATRIX_KEYWORD "matrix" 
/** @} */


/*!
  \class   oSystemMatrix
  \brief   This class is designed to manage pre-computed system matrices
  \details This class is basically a container for pre-computed system matrices
           able to read/write oProjectionLines. It can be used during the
           reconstruction process by the oProjectorManager.
           Everything needs to be implemented !
*/
class oSystemMatrix
{
  // -------------------------------------------------------------------
  // Constructor & Destructor
  public:
    /*!
      \fn      public oSystemMatrix::oSystemMatrix()
      \brief   The constructor of oSystemMatrix
      \details This is the default and unique constructor. It does not take any parameter and
               its role is only to affect default values to each member of the class.
    */
    oSystemMatrix();
    /*!
      \fn      virtual public oSystemMatrix::~oSystemMatrix()
      \brief   The destructor of oSystemMatrix
      \details This is the default and unique destructor. It does not take any parameter and
               its role is only to free or delete all structures that were built by this class.
               It is virtual, so that it is automatically called when a child object is deleted.
    */
    ~oSystemMatrix();

  // -------------------------------------------------------------------
  // Public member functions
  public:
    /*!
      \fn      public int oSystemMatrix::Project()
      \param   int a_direction
      \param   oProjectionLine* ap_ProjectionLine
      \param   string a_eventIdentifier
      \brief   A function used to compute the projection elements with respect to the provided parameters
      \details This function is used to fill the provided oProjectionLine with the system matrix elements associated
               to the provided identifier.
      \return  An integer reflecting the projection status; 0 if no problem, another value otherwise.
    */
    int Project(int a_direction, oProjectionLine* ap_ProjectionLine, string a_eventIdentifier);
    /*!
      \fn      public int oSystemMatrix::Project()
      \param   vDataFile* ap_DataFile
      \param   INTNB nbVoxels
      \param   string& a_pathToSystemMatrixDirectory
      \brief   A function used to load the system matrix and to set some parameters with respect to the provided datafile
      \return  An integer reflecting the status; 0 if no problem, another value otherwise.
    */
    int LoadData(vDataFile* ap_DataFile, INTNB nbVoxels, string& a_pathToSystemMatrixDirectory);


  // -------------------------------------------------------------------
  // Private member functions
  private:


  // -------------------------------------------------------------------
  // Public Get & Set functions
  public:
    /*!
      \fn      inline public bool oSystemMatrix::GetCompatibilityWithSPECTAttenuationCorrection()
      \return  m_compatibleWithSPECTAttenuationCorrection
    */
    inline bool GetCompatibilityWithSPECTAttenuationCorrection()
           {return m_compatibleWithSPECTAttenuationCorrection;}


  // -------------------------------------------------------------------
  // Data members
  private:
    map<pair<int,int>,INTNB>  mp_nbVoxels;           	/*!< The map associating pairs of indices to the number of contributing voxels */
    map<pair<int,int>,INTNB*> m2p_voxelIndicesMap;   	/*!< The map associating pairs of indices to the list of voxels indices */
    map<pair<int,int>,char*> m2p_voxelWeightsMap;   	/*!< The map associating pairs of indices to the raw pointers to mapped memory */
	map<pair<int,int>,oMemoryMapped*> m2p_mappedFiles;  /*!< The map associating pairs of indices to the list of voxels weights */
	map<string, int> mp_identifierMap; 					/*!< The map associating the line of system matrix to the event identifier */
    bool m_compatibleWithSPECTAttenuationCorrection; 	/*!< Boolean that says if the projector is compatible with SPECT attenuation correction */
	int m_nbTOFBins;									/*!< The number of "TOF bins" (number of acquisitions per event) */
	int m_nbVoxels;										/*!< The number of voxels */
};

#endif
