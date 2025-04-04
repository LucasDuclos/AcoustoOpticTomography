
/*!
  \file
  \ingroup  datafile
  \brief    Declaration of class iEventHistoAOT
*/

#ifndef IEVENTHISTOAOT_HH
#define IEVENTHISTOAOT_HH 1

#include "vEvent.hh"
#include "iEventHistoAOT.hh"

/*!
  \class   iEventHistoAOT
  \brief   Inherit from iEventHistoAOT. Class for AOT list-mode events
  \details It manages data and functions specific to list mode AOT.
*/
class iEventHistoAOT : public vEvent
{
  // -------------------------------------------------------------------
  // Constructor & Destructor
  public:
    /*!
      \brief   iEventHistoAOT constructor. 
               Initialize the member variables to their default values.
    */
    iEventHistoAOT();    
    /*!
      \brief   iEventHistoAOT destructor. 
    */
    ~iEventHistoAOT();

  // -------------------------------------------------------------------
  // Public member functions
  public:
    /*!
      \fn      inline int iEventHistoAOT::AllocateSpecificData()
      \brief   Function allowing the allocation of specific data. Return 0 by default for iEventHistoAOT
      \return  0 is success, positive value otherwise
    */
    int AllocateSpecificData();
    /*!
      \fn      void iEventHistoAOT::Describe()
      \brief   This function can be used to get a description of the event printed out
    */
    void Describe();
    /*!
      \fn      void iEventHistoAOT::GetEventName()
      \param   a_eventName
      \brief   Get the name of the system matrix file corresponding to the event
      \return  0 is success, positive value is otherwise
    */
    int GetEventIdentifier(string& a_eventName);

  // -------------------------------------------------------------------
  // Public Get & Set functions
  public:
    /*!
      \fn      inline FLTNB iEventHistoAOT::GetEventValue()
      \param   a_bin
      \return  the event value corresponding to the specific "TOF bin" passed as parameter
    */
    inline FLTNB GetEventValue(int a_bin)
           {return mp_eventValue[a_bin];}
    /*!
      \fn      inline uint16_t iEventHistoAOT::GetEventNbTOFBins()
      \return  the number of TOF bins in the Event
    */
    inline uint16_t GetEventNbTOFBins()
           {return m_eventNbTOFBins;}
    /*!
      \fn      inline INTNB vEvent::GetNbValueBins()
      \brief   Get the number of event value bins. Redundant function but used in the code.
      \return  Number of TOF bins here
    */
    inline INTNB GetNbValueBins()
           {return m_eventNbTOFBins;}
    /*!
      \fn      inline uint8_t iEventHistoAOT::GetUSTransducersActivation()
      \param   a_transducersGroupNb
      \return  the activation values corresponding to the specific group number of US transducer passed as parameter
    */
    inline uint8_t GetUSTransducersActivation(int a_transducersGroupNb)
           {return mp_USTransducersActivation[a_transducersGroupNb];}
    /*!
      \fn      inline uint16_t iEventHistoAOT::GetNbUSTransducers()
      \return  the number of US transducers used in the event
    */
    inline uint16_t GetNbUSTransducers()
           {return m_nbUSTransducers;}
    /*!
      \fn      inline uint16_t iEventHistoAOT::GetNbGroupsUSTransducers()
      \return  the number of groups of 8 US transducers used in the event
    */
    inline uint16_t GetNbGroupsUSTransducers()
           {return m_nbGroupsUSTransducers;}
    /*!
      \fn      inline FLTNB iEventHistoAOT::GetWaveAngle()
      \return  the angle of the plane wave in the event
    */
    inline int8_t GetWaveAngle()
           {return m_waveAngle;}
    /*!
      \fn      inline FLTNB iEventHistoAOT::GetAdditiveCorrections()
      \param   a bin (not used)
    */
    inline FLTNB GetAdditiveCorrections(int a_bin)
		  {return m_additiveCorrections;}
    /*!
      \fn      inline void iEventHistoAOT::SetEventValue()
      \param   a_bin (0 if noTOF)
      \param   a_value
      \brief   Cast the FLTNBDATA value passed in parameters in FLTNB, and use it to set the event value of the specific TOF bin
    */
    inline void SetEventValue(int a_bin, FLTNBDATA a_value)
           {mp_eventValue[a_bin] = (FLTNB)a_value;} 
    /*!
      \fn      inline FLTNB iEventHistoAOT::GetEventScatRate()
      \param   a_value
      \return  the scatter correction rate in 1/s for this histo-mode event for the specific TOF bin
    */
    inline void SetEventNbTOFBins(uint16_t a_value)
           {m_eventNbTOFBins = a_value;}
    /*!
      \fn      inline void iEventHistoAOT::SetUSTransducersActivation()
      \param   a_transducerGroupNb
      \param   a_value
      \brief   Set the activation value of the specific group number of US transducer
    */
    inline void SetUSTransducersActivation(int a_transducerGroupNb, uint8_t a_value)
           {mp_USTransducersActivation[a_transducerGroupNb] = a_value;}
    /*!
      \fn      inline void iEventHistoAOT::SetNbUSTransducers()
      \param   a_value
    */
    inline void SetNbUSTransducers(uint16_t a_value)
           {m_nbUSTransducers = a_value;}
    /*!
      \fn      inline void iEventHistoAOT::SetNbGroupsUSTransducers()
      \param   a_value
    */
    inline void SetNbGroupsUSTransducers(uint16_t a_value)
           {m_nbGroupsUSTransducers = a_value;}
    /*!
      \fn      inline void iEventHistoAOT::SetNbUSTransducers()
      \param   a_value
    */
    inline void SetWaveAngle(FLTNB a_value)
           {m_waveAngle = a_value;}
    /*!
      \fn      void iEventHistoAOT::SetAdditiveCorrections()
      \param   a_additiveCorrection
    */
    inline void SetAdditiveCorrections(FLTNB a_additiveCorrection)
		  {m_additiveCorrections = a_additiveCorrection;}
	// Other correction functions, which are not used
    /*!
      \fn      inline FLTNB iEventHistoAOT::GetMultiplicativeCorrections()
    */
    inline FLTNB GetMultiplicativeCorrections()
		  {return 1.;}
    /*!
      \fn      inline void iEventHistoAOT::MultiplyAdditiveCorrections()
      \param   FLTNB a_factor
    */
    inline void MultiplyAdditiveCorrections(FLTNB a_factor) {};
    
  // -------------------------------------------------------------------
  // Private member functions
  private:

  // -------------------------------------------------------------------
  // Data members
  private:
    FLTNB* mp_eventValue; 					/*!< Pointer containing the amount of data in each potential "TOF bin". Default value =NULL */
    uint16_t m_eventNbTOFBins; 				/*!< Number of "TOF bins" in the Event. A "TOF bin" is an acquisition. Default value =0 */
    uint8_t* mp_USTransducersActivation;	/*!< Pointer containing the activation values of the US transducers in the event (grouped by 8 and zero-padded if necessary). Default value =NULL*/
	uint16_t m_nbUSTransducers;				/*!< Number of US transducers. Default value =0*/
	uint16_t m_nbGroupsUSTransducers;		/*!< Number of groups of 8 US transducers. Default value =0*/
	int8_t m_waveAngle;						/*!< Wave angle of the acquisition. Default value =0*/
	FLTNB m_additiveCorrections;				/*!< Additive correction of the acquisition. Default value =0*/
};

#endif
