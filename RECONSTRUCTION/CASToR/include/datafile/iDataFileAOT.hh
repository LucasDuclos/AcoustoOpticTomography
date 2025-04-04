
/*!
  \file
  \ingroup  datafile
  \brief    Declaration of class iDataFileAOT
*/

#ifndef IDATAFILEAOT_HH
#define IDATAFILEAOT_HH 1

#include "gVariables.hh"
#include "vDataFile.hh"

/*!
  \class   iDataFileAOT
  \brief   Inherit from vDataFile. Class that manages the reading of a AOT input file (header + data).
  \details It contains several arrays corresponding to the different kind of informations the data file could contain. \n
           As many booleans as arrays say if the data are here or not. The data file can be either completely loaded, or read event by event during reconstruction. \n
           MPI is coming here to cut the data file into peaces (also either can be loaded or read on-the-fly).
*/
class iDataFileAOT : public vDataFile
{
  // -------------------------------------------------------------------
  // Constructor & Destructor
  public:
    /*!
      \brief   iDataFileAOT constructor. 
               Initialize the member variables to their default values.
    */
    iDataFileAOT();
    /*!
      \brief   iDataFileAOT destructor. 
    */
    ~iDataFileAOT();

  // -------------------------------------------------------------------
  // Public member functions
  public:
    /*!
      \fn      iDataFileAOT::ReadSpecificInfoInHeader()
      \param   bool a_affectQuantificationFlag
      \brief   Read through the header file and gather specific AOT information.
      \details If the parameter flag is on, then affect the quantification factors from the oImageDimensionsAndQuantification after
               reading relevant information
      \return  0 is success, positive value otherwise
    */
    int ReadSpecificInfoInHeader(bool a_affectQuantificationFlag);
    /*!
      \fn      iDataFileAOT::WriteHeader()
      \brief   Generate a header file according to the data output information.
      \return  0 if success, and positive value otherwise.
    */
    int WriteHeader();
    /*!
      \fn      iDataFileAOT::ComputeSizeEvent()
      \brief   Computation of the size of each event according to the mandatory/optional correction fields
      \return  0 is success, positive value otherwise
    */
    int ComputeSizeEvent();
    /*!
      \fn      iDataFileAOT::PrepareDataFile()
      \brief   Store different kind of information inside arrays (data relative to specific correction as well as basic raw data for the case data is loaded in RAM) \n
               Use the flag provided by the user to determine how the data has to be sorted (preloaded or read on the fly)
      \return  0 is success, positive value otherwise
    */
    int PrepareDataFile();
    /*!
      \fn      iDataFileAOT::WriteEvent()
      \param   ap_Event : event containing the data to write
      \param   a_th : index of the thread from which the function was called
      \brief   Write event according to the chosen type of data
      \return  0 if success, and positive value otherwise.
    */
    int WriteEvent(vEvent* ap_Event, int a_th);
    /*!
      \fn      iDataFileAOT::GetEventSpecific()
      \param   ap_buffer : address pointing to the event to recover
      \param   a_th : index of the thread from which the function was called
      \brief   Read an event from the position pointed by 'ap_buffer', parse the generic or modality-specific information, and store them in the (multithreaded) 'm2p_BufferEvent' object
      \return  the thread-specific  'm2p_BufferEvent' object containing the modality-specific information for the event
    */
    vEvent* GetEventSpecific(char* ap_buffer, int a_th);
    /*!
      \fn      iDataFileAOT::DescribeSpecific()
      \brief   Implementation of the pure virtual eponym function that simply prints info about the datafile
    */
    void DescribeSpecific();

  // -------------------------------------------------------------------
  // Public Get & Set functions
  public:
    /*!
      \fn      iDataFileAOT::GetAcquisitionFrequencyInHz()
      \return  m_acquisitionFrequencyInHz
    */
    inline FLTNB GetAcquisitionFrequencyInHz()
           {return m_acquisitionFrequencyInHz;}
        /*!
      \fn      iDataFilePET::GetNbTOFBins() 
      \return  number of TOF bins in the acquisition
    */
    inline int GetNbTOFBins()
           {return m_nbTOFBins;}
    /*!
      \fn      iDataFileAOT::GetNbUSTransducers()
      \return  m_nbUSTransducers
    */
    inline uint16_t GetNbUSTransducers()
           {return m_nbUSTransducers;}
    /*!
      \fn      iDataFileAOT::GetNbGroupsUSTransducers()
      \return  m_nbGroupsUSTransducers
    */
    inline uint16_t GetNbGroupsUSTransducers()
           {return m_nbGroupsUSTransducers;}
    /*!
      \fn      iDataFileAOT::SetAcquisitionFrequencyInHz() 
      \param   a_acquisitionFrequencyInHz
      \brief   Set the acquisition frequency of the data
    */
    inline void SetAcquisitionFrequencyInHz(FLTNB a_acquisitionFrequencyInHz)
           {m_acquisitionFrequencyInHz = a_acquisitionFrequencyInHz;}
    /*!
      \fn      iDataFileAOT::SetNbUSTransducers() 
      \param   a_nbUSTransducers
      \brief   Set the number of US transducers
    */
    inline void SetNbUSTransducers(uint16_t a_nbUSTransducers)
           {m_nbUSTransducers = a_nbUSTransducers;}
    /*!
      \fn      iDataFileAOT::SetNbGroupsUSTransducers() 
      \param   a_nbGroupsUSTransducers
      \brief   Set the number of groups of 8 US transducers
    */
    inline void SetNbGroupsUSTransducers(uint16_t a_nbGroupsUSTransducers)
           {m_nbGroupsUSTransducers = a_nbGroupsUSTransducers;}

  // -------------------------------------------------------------------
  // Public functions dedicated to the projection script
  public:
    /*!
      \fn      iDataFileAOT::PROJ_InitFile()
      \brief   Initialize the fstream objets for output writing as well as some other variables specific to the Projection script (Event-based correction flags, Estimated size of data file)
      \return  0 if success, and positive value otherwise.
    */
    int PROJ_InitFile();
    /*!
      \fn      iDataFileAOT::PROJ_GetScannerSpecificParameters()
      \brief   Get AOT specific parameters for projections from the scanner object, through the scannerManager.
      \return  0 if success, positive value otherwise
    */
    int PROJ_GetScannerSpecificParameters();

  // -------------------------------------------------------------------
  // Private member functions
  private:
    /*!
      \fn      int iDataFileAOT::SetSpecificParametersFrom()
      \brief   Initialize all parameters specific to AOT from the provided datafile.
      \return  0 if success, and positive value otherwise
    */
    int SetSpecificParametersFrom(vDataFile* ap_DataFile);
    /*!
      \fn      iDataFileAOT::CheckSpecificParameters()
      \brief   Check parameters specific to AOT data
      \return  0 if success, and positive value otherwise.
    */
    int CheckSpecificParameters();
    /*!
      \fn      iDataFileAOT::WriteHistoEvent()
      \param   ap_Event : event containing the data to write
      \param   a_th : index of the thread from which the function was called
      \brief   Write a AOT histogram event
      \return  0 if success, and positive value otherwise.
    */
    int WriteHistoEvent(iEventHistoAOT* ap_Event, int a_th);
    /*!
      \fn      iDataFileAOT::CheckFileSizeConsistency()
      \brief   This function is implemented in child classes \n
               Check if file size is consistent.
      \return  0 if success, and positive value otherwise.
    */
    int CheckFileSizeConsistency();
    /*!
      \fn      iDataFileAOT::CheckSpecificConsistencyWithAnotherDataFile()
      \param   vDataFile* ap_DataFile
      \brief   Check consistency between 'this' and the provided datafile, for specific characteristics.
      \details Implementation of the pure virtual function from vDataFile. It checks correction flags, etc.
      \return  0 if the provided datafile is consistent with 'this', another value otherwise
    */
    int CheckSpecificConsistencyWithAnotherDataFile(vDataFile* ap_DataFile);

  // -------------------------------------------------------------------
  // Data members
  private:
	FLTNB m_acquisitionFrequencyInHz;		/*!< Acquisition frequency. Default = 0 */
	uint16_t m_nbUSTransducers;				/*!< Number of US transducers. Default = 0 */
	uint16_t m_nbGroupsUSTransducers;		/*!< Number of groups of 8 US transducers. Default = 0 */
    int m_nbTOFBins;                     	/*!< Number of "TOF bins" (ie. acquisitions, name has to be changeed) for histogram mode. Default = 1 */  
    bool m_hasAdditiveCorrections;			/*!< Flag to say if the data have additive corrections. Default = false */
};

#endif
