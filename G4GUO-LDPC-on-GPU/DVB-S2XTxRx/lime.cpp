#include <pthread.h>
#include "lime/LimeSuite.h"
#include "dvbs2_rx.h"
#include <iostream>
#define USE_MATH_DEFINES
#include <math.h>
#include <unistd.h>
#include <stdio.h>

using namespace std;

static lms_device_t* m_lms_device;
static int m_n_devices;
static lms_stream_t m_rx_streams[N_CHANS];
static lms_stream_t m_tx_streams[N_CHANS];
static lms_stream_meta_t m_rx_metadata[N_CHANS];
static lms_stream_meta_t m_tx_metadata[N_CHANS];

#define CH_0 0

SComplex samples[1024*128];

static int error()
{
    //print last error message
    cout << "ERROR:" << LMS_GetLastErrorMessage();
    if (m_lms_device != NULL)
        LMS_Close(m_lms_device);
    exit(-1);
}


int lime_rx_buffer(SComplex *s, int chan, int len ){
    chan = 0;
	int n = LMS_RecvStream(&m_rx_streams[chan], s, len, &m_rx_metadata[chan], 100000000);
	if(n < 0 ) error();

/*
	lms_stream_status_t status;
	LMS_GetStreamStatus(&m_rx_streams[0], &status); //Obtain RX stream stats

	cout << "RX Overrun: " << status.overrun << endl;
	cout << "RX Underrun: " << status.underrun << endl;
	cout << "RX S Rate: " << status.sampleRate / 1e6 << " MB/s" << endl;
	cout << "RX L Rate: " << status.linkRate / 1e6 << " MB/s" << endl;
	cout << "RX 0 FIFO: " << (100 * status.fifoFilledCount) / status.fifoSize << "%" << endl; //percentage of TX 0 fifo filled
	cout << "RX 0 FIFO Size : " << status.fifoSize << endl; //percentage of TX 0 fifo filled
*/
	return n;
}
int lime_rx_buffer(FComplex *s, int chan, int len ){

	int n = LMS_RecvStream(&m_rx_streams[chan], s, len, &m_rx_metadata[chan], 1000);

	return n;
}
int lime_rx_samples(FComplex *s, int len ){

	int n = LMS_RecvStream(&m_rx_streams[CH_0], s, CH_0, &m_rx_metadata[CH_0], 1000);

	return n;
}
int lime_tx_buffer(SComplex *s, int chan, int len ){

	int n = LMS_SendStream(&m_tx_streams[chan], s, len, &m_tx_metadata[chan], 1000);

	return n;
}
int lime_txrx_buffer(SComplex *tx, SComplex *rx, int chan, int len ){
	int n;
	n = LMS_SendStream(&m_tx_streams[chan], tx, len, &m_tx_metadata[chan], 1000);
	n = LMS_RecvStream(&m_rx_streams[chan], rx, len, &m_rx_metadata[chan], 1000);
    return n;
}
/*
int lime_rxtx_buffer(SComplex *rx, SComplex *tx, int len ){
	lms_stream_status_t status;
	lms_stream_meta_t rx_metadata; //Use metadata for additional control over sample receive function behaviour
	lms_stream_meta_t tx_metadata; //Use metadata for additional control over sample send function behaviour
    rx_metadata.flushPartialPacket = false; //Do not discard data remainder when read size differs from packet size
    rx_metadata.waitForTimestamp = false; //Do not wait for specific timestamps
	tx_metadata.flushPartialPacket = false; //Do not discard data remainder when read size differs from packet size
	tx_metadata.waitForTimestamp = false; //Enable synchronization to HW timestamp

//    int n = LMS_RecvStream(&m_rx_streams[0], rx, len, &rx_metadata, 1000);

	//tx_metadata.timestamp = rx_metadata.timestamp + 1024 * 8;
    //tx_metadata.timestamp = rx_metadata.timestamp + (1024 * 4);
	//	cout << "RX samples: " << n << endl;

	//LMS_GetStreamStatus(&m_tx_streams[0], &status); //Obtain TX stream stats
	//cout << "TX 0 FIFO: " << (100 * status.fifoFilledCount) / status.fifoSize << "%" << endl; //percentage of TX 0 fifo filled
    //if(status.fifoFilledCount >= status.fifoSize - 1024 ) usleep(10);
    int n = LMS_SendStream(&m_tx_streams[0], tx, len, &tx_metadata, 1000);
 //	cout << "RX timestamp: " << rx_metadata.timestamp << endl;
//	cout << "TX timestamp: " << tx_metadata.timestamp << endl;


    static int count;

    if((++count)%1000 == 0 ){

	LMS_GetStreamStatus(&m_tx_streams[0], &status); //Obtain TX stream stats

	cout << "TX Overrun: " << status.overrun << endl;
	cout << "TX Underrun: " << status.underrun << endl;
	cout << "TX S Rate: " << status.sampleRate / 1e6 << " MB/s" << endl;
	cout << "TX L Rate: " << status.linkRate / 1e6 << " MB/s" << endl;
	cout << "TX 0 FIFO: " << (100 * status.fifoFilledCount) / status.fifoSize << "%" << endl; //percentage of TX 0 fifo filled
	cout << "TX 0 FIFO Size : " << status.fifoSize << endl; //percentage of TX 0 fifo filled

	LMS_GetStreamStatus(&m_rx_streams[0], &status); //Obtain RX stream stats

	cout << "RX Overrun: " << status.overrun << endl;
	cout << "RX Underrun: " << status.underrun << endl;
	cout << "RX S Rate: " << status.sampleRate / 1e6 << " MB/s" << endl;
	cout << "RX L Rate: " << status.linkRate / 1e6 << " MB/s" << endl;
	cout << "RX 0 FIFO: " << (100 * status.fifoFilledCount) / status.fifoSize << "%" << endl; //percentage of TX 0 fifo filled
	cout << "RX 0 FIFO Size : " << status.fifoSize << endl; //percentage of TX 0 fifo filled

	}

    return 0;
}
*/
void lime_set_transmit_bw( double bw, int chan ){
	double tx_bw = bw/2;
	double rx_bw = bw;

	if(LMS_SetLPFBW( m_lms_device, LMS_CH_TX, chan, tx_bw)    < 0 ) error();
//	if(LMS_SetLPFBW( m_lms_device, LMS_CH_RX, chan, rx_bw)    < 0 ) error();
	if(LMS_Calibrate(m_lms_device, LMS_CH_TX, chan, tx_bw, 0) < 0 ) error();
//	if(LMS_Calibrate(m_lms_device, LMS_CH_RX, chan, rx_bw, 0) < 0 ) error();

}
int lime_open( double freq, double srate )
{
    // This blocks
	// Set all device handles to NULL
	m_lms_device = NULL;
    // Sample rate is twice the symbol rate
    //Find devices
    int n;
    lms_info_str_t list[N_RADIOS]; //should be large enough to hold all detected devices

    if ((m_n_devices = LMS_GetDeviceList(list)) < 0) error();//NULL can be passed to only get number of devices

    cout << "Number of Devices found: " << m_n_devices << endl; //print number of devices

    if (m_n_devices < 1) return -1;

    //open the first device
    if (LMS_Open(&m_lms_device, list[0], NULL)) error();

    //Initialize device with default configuration
    //Do not use if you want to keep existing configuration
    if (LMS_Init(m_lms_device) != 0) error();
    //if (LMS_Reset(m_lms_device) != 0) error();

    lms_name_t ant_list[10];

    cout << endl;

    if((n=LMS_GetAntennaList(m_lms_device, LMS_CH_TX, 0, ant_list)) <0 ) error();
    for( int i = 0; i < n; i++ ){
        cout << "Tx Antenna: " << ant_list[i] << endl;
    }

    cout << endl;

    if((n=LMS_GetAntennaList(m_lms_device, LMS_CH_RX, 0, ant_list)) <0 ) error();
    for( int i = 0; i < n; i++ ){
        cout << "Rx Antenna: " << ant_list[i] << endl;
    }

    cout << endl;

    //Get number of channels
    if ((n = LMS_GetNumChannels(m_lms_device, LMS_CH_RX)) < 0) error();
    cout << "Number of available RX channels: " << n << endl;
    if ((n = LMS_GetNumChannels(m_lms_device, LMS_CH_TX)) < 0) error();
    cout << "Number of available TX channels: " << n << endl;

    //Enable RX channel
    //Channels are numbered starting at 0
    if (LMS_EnableChannel(m_lms_device, LMS_CH_RX, 0, true) != 0) error();
    //if (LMS_EnableChannel(m_lms_device, LMS_CH_RX, 1, true) != 0) error();

    //Enable TX channels
   // if (LMS_EnableChannel(m_lms_device, LMS_CH_TX, 0, true) != 0) error();
   // if (LMS_EnableChannel(m_lms_device, LMS_CH_TX, 1, true) != 0) error();

    //Set RX center frequency to 1 GHz
    //Automatically selects antenna port
//    if (LMS_SetLOFrequency(m_lms_device, LMS_CH_RX, 0, 1249000000) != 0) error();
    if (LMS_SetLOFrequency(m_lms_device, LMS_CH_RX, 0, freq) != 0) error();
    //if (LMS_SetLOFrequency(m_lms_device, LMS_CH_RX, 1, freq) != 0) error();

    //Set TX center frequency to 1 GHz
    //Automatically selects antenna port
    //if (LMS_SetLOFrequency(m_lms_device, LMS_CH_TX, 0, 1.249e9) != 0) error();
    //if (LMS_SetLOFrequency(m_lms_device, LMS_CH_TX, 1, 1.249e9) != 0) error();

    // Preferred oversampling in RF 4x
    // This set sampling rate for all channels
    if (LMS_SetSampleRate(m_lms_device, srate, 4) != 0) error();

    double nf[16];
    nf[0] = 1000000;
    nf[1] = 3000000;
    nf[2] = 3000000;
    nf[3] = 3000000;
    nf[4] = 3000000;
    nf[5] = 3000000;
    nf[6] = 3000000;
    nf[7] = 3000000;
    nf[8] = 3000000;
    nf[9] = 3000000;
    nf[10] = 3000000;
    nf[11] = 3000000;
    nf[12] = 3000000;
    nf[13] = 3000000;
    nf[14] = 3000000;
    nf[15] = 3000000;

 //   if (LMS_SetNCOFrequency(m_lms_device, LMS_CH_TX, 0, nf, 0) != 0) error();

    //Set RX gain
    if (LMS_SetNormalizedGain(m_lms_device, LMS_CH_RX, 0, 1.0) != 0) error();
//    if (LMS_SetNormalizedGain(m_lms_device, LMS_CH_RX, 1, 1.0) != 0) error();
    if (LMS_SetGaindB(        m_lms_device, LMS_CH_RX, 0, 50) != 0) error();

    //Set TX gain
    if (LMS_SetNormalizedGain(m_lms_device, LMS_CH_TX, 0, 0.3) != 0) error();
//    if (LMS_SetNormalizedGain(m_lms_device, LMS_CH_TX, 1, 0.1) != 0) error();

    if (LMS_SetAntenna(m_lms_device, LMS_CH_TX, 0, 1) !=0) error();
    if (LMS_SetAntenna(m_lms_device, LMS_CH_RX, 0, 1) !=0) error();

//    lime_set_transmit_bw(  10000000, 0 );

   // uint16_t rval;
   // LMS_ReadLMSReg(m_lms_device, 0x0208, &rval);
   // printf("Reg VAL %.4X\n",rval);
   // LMS_ReadLMSReg(m_lms_device, 0x0020, &rval);
   // printf("Reg VAL %.4X\n",rval);


//	LMS_WriteLMSReg(m_lms_device, uint32_t address, rval);

    //Enable test signals generation in RX channels
    //To receive data from RF, remove these lines or change signal to LMS_TESTSIG_NONE
    //if (LMS_SetTestSignal(m_lms_device, LMS_CH_RX, 0, LMS_TESTSIG_NCODIV4, 0, 0) != 0) error();
    //if (LMS_SetTestSignal(m_lms_device, LMS_CH_RX, 1, LMS_TESTSIG_NCODIV8F, 0, 0) != 0) error();

    //Streaming Setup

    //Initialize streams
    //All streams setups should be done before starting streams. New streams cannot be set-up if at least stream is running.
    for (int i = 0; i < N_CHANS; i++)
    {
       	m_rx_streams[i].channel = i; //channel number
       	m_rx_streams[i].fifoSize = 1024 * 128; //fifo size in samples
       	m_rx_streams[i].throughputVsLatency = 1.0; //optimize for maximum throughput
       	m_rx_streams[i].isTx = false; //RX channel
       	m_rx_streams[i].dataFmt = lms_stream_t::LMS_FMT_I16; //16-bit integers
       	if (LMS_SetupStream(m_lms_device, &m_rx_streams[i]) != 0) error();

       	m_tx_streams[i].channel = i; //channel number
       	m_tx_streams[i].fifoSize = 1024 * 128; //fifo size in samples
       	m_tx_streams[i].throughputVsLatency = 1.0; //optimize for  throughput
       	m_tx_streams[i].isTx = true; //TX channel
       	m_tx_streams[i].dataFmt = lms_stream_t::LMS_FMT_I16; //16-bit integers
       	if (LMS_SetupStream(m_lms_device, &m_tx_streams[i]) != 0) error();

        m_rx_metadata[i].flushPartialPacket = false; //Do not discard data remainder when read size differs from packet size
        m_rx_metadata[i].waitForTimestamp = false; //Do not wait for specific timestamps
    	m_tx_metadata[i].flushPartialPacket = false; //Do not discard data remainder when read size differs from packet size
        m_tx_metadata[i].waitForTimestamp = false; //Do not wait for specific timestamps

    }
 	//Start streaming
	for (int i = 0; i < N_CHANS; i++)
	{
		if (LMS_StartStream(&m_rx_streams[i]) != 0) error();
		//if (LMS_StartStream(&m_tx_streams[i]) != 0) error();
	}

	// Test code
/*
	static float a;
	SComplex *rx = new SComplex[1024];
	for(int i = 0; i < 100000; i++){
		SComplex *tx = new SComplex[1024];
		for( int x = 0; x < 1024; x++){
			tx[x].re = (short)(cos(a)*32768*0.5);
			tx[x].im = (short)(sin(a)*32768*0.5);
			a += 0.1*2*M_PI;
			if(a > 2*M_PI) a -= 2*M_PI;
		}
		lime_tx_buffer( rx, 0, 1024 );
		delete [] tx;
	}
	delete [] rx;
	cout << "Samples done" << endl;
	lime_terminate();

exit(0);
*/
    return 0;
}

void lime_set_rx_sr( double srate ){
    LMS_SetSampleRate(m_lms_device, srate, 4);
}

double lime_get_rx_sr(void){
    double host_Hz;
    double rf_Hz;
    LMS_GetSampleRate(m_lms_device, LMS_CH_RX, CH_0, &host_Hz, &rf_Hz);
    return host_Hz;
}

void lime_close(void)
{
    //Stop TX streaming
    for( int device = 0; device < m_n_devices; device++)
    {
    	for (int i = 0; i < N_CHANS; i++)
    	{
    		LMS_StopStream(&m_tx_streams[i]);
    		LMS_DestroyStream(m_lms_device, &m_tx_streams[i]);
    	}
    }
    //Stop RX streaming
    for (int i = 0; i < N_CHANS; i++)
    {
    	LMS_StopStream(&m_rx_streams[i]); //stream is stopped but can be started again with LMS_StartStream()
    	LMS_DestroyStream(m_lms_device, &m_rx_streams[i]); //stream is deallocated and can no longer be used
    }

    //Close device
    LMS_Close(m_lms_device);
}
