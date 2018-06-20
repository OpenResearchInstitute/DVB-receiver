#ifndef __DVBS2_RX_H__
#define  __DVBS2_RX_H__
#define USE_MATH_DEFINES

#include <math.h>
#include <stdint.h>
#include <string.h>

#pragma once

// Number of simultaneous LDPC frames


// LDPC

typedef int8_t LLR;
#define NP_FRAMES 128

//#define CV 127
#define CV 127.0
//#define UCLAMP(x) ((LLR)((fabs(x*8) > CV) ? x >= 0 ? CV : -CV : x*8))
#define UCLAMP(x) ((LLR)((fabs(x*8) > CV) ? x >= 0 ? CV : -CV : x >= 0 ? (x*8)+1 : (x*8)-1))

//#define CLAMP(x) ((LLR)((fabs(x) > CV) ? x >= 0 ? CV : -CV : x))
//#define CLAMP(x) (x)
//#define UCLAMP(x) ((LLR)((fabs(x) > CV) ? x >= 0 ? CV : -CV : x))

//#define BUILD_BCH_TABLES

typedef uint8_t Bit;
typedef uint8_t Bits;


// LimeSDR defines
#define N_RADIOS 1
#define N_CHANS 1

// Samples per symbol
#define SAMPLES_PER_SYMBOL 2

// Equaliser size
#define KN 6


typedef struct{
	int16_t re;
	int16_t im;
}SComplex;

typedef struct{
	float re;
	float im;
}FComplex;

//
#define CERROR(a,b) (((a.re-b.re)*(a.re-b.re))+((a.im-b.im)*(a.im-b.im)))

#define GPU_BLOCK_SIZE 1024
#define S2_BLOCK_N 90
#define S2_SPS_N 2

#define RX_SAMPLE_HISTORY 200
#define TRACK_RANGE 8

#define FRAME_SIZE_NORMAL 64800
#define FRAME_SIZE_MEDIUM 32400
#define FRAME_SIZE_SHORT  16200

#define PREAM_N 90
#define VLPREAM_N 900

typedef enum{
	// Code rates S2
	cR_1_4,
	cR_1_3,
	cR_2_5,
	cR_1_2,
	cR_3_5,
	cR_2_3,
	cR_3_4,
	cR_4_5,
	cR_5_6,
	cR_8_9,
	cR_9_10,
	// Code rates S2X
	cR_2_9,
	cR_13_45,
	cR_9_20,
	cR_11_20,
	cR_26_45,
	cR_28_45,
	cR_23_36,
	cR_25_36,
	cR_13_18,
	cR_7_9,
	cR_90_180,
	cR_96_180,
	cR_100_180,
	cR_104_180,
	cR_116_180,
	cR_124_180,
	cR_128_180,
	cR_132_180,
	cR_135_180,
	cR_140_180,
	cR_154_180,
	cR_18_30,
	cR_20_30,
	cR_22_30,
	cR_11_45,
	cR_4_15,
	cR_14_45,
	cR_7_15,
	cR_8_15,
	cR_32_45,
	cR_1_5
}CodeRate;



// Frame type
typedef enum{
	frame_NORMAL,
	frame_MEDIUM,
	frame_SHORT,
	frame_UNKNOWN
}FrameType;

// LDPC tables
extern uint16_t ldpc_tab_1_4N[45][13];
extern uint16_t ldpc_tab_1_3N[60][13];
extern uint16_t ldpc_tab_2_5N[72][13];
extern uint16_t ldpc_tab_1_2N[90][9];
extern uint16_t ldpc_tab_3_5N[108][13];
extern uint16_t ldpc_tab_2_3N[120][14];
extern uint16_t ldpc_tab_3_4N[135][13];
extern uint16_t ldpc_tab_4_5N[144][12];
extern uint16_t ldpc_tab_5_6N[150][14];
extern uint16_t ldpc_tab_8_9N[160][5];
extern uint16_t ldpc_tab_9_10N[162][5];
extern uint16_t ldpc_tab_1_4S[9][13];
extern uint16_t ldpc_tab_1_3S[15][13];
extern uint16_t ldpc_tab_2_5S[18][13];
extern uint16_t ldpc_tab_1_2S[20][9];
extern uint16_t ldpc_tab_3_5S[27][13];
extern uint16_t ldpc_tab_2_3S[30][14];
extern uint16_t ldpc_tab_3_4S[33][13];
extern uint16_t ldpc_tab_4_5S[35][4];
extern uint16_t ldpc_tab_5_6S[37][14];
extern uint16_t ldpc_tab_8_9S[40][5];
// Table B.1: LDPC code identifier: 2/9 (nldpc = 64,800)
extern uint16_t ldpc_tab_2_9NX[40][12];
// Table B.2: LDPC code identifier: 13/45 (nldpc = 64,800)
extern uint16_t ldpc_tab_13_45NX[52][13];
//Table B.3: LDPC code identifier: 9/20 (nldpc = 64,800)
extern uint16_t ldpc_tab_9_20NX[82][13];
//Table B.4: LDPC code identifier: 11/20 (nldpc = 64,800)
extern uint16_t ldpc_tab_11_20NX[99][14];
// Table B.5:LDPC code identifier: 26/45 (nldpc = 64,800)
extern uint16_t ldpc_tab_26_45NX[104][14];
// Table B.6: LDPC code identifier: 28/45 (nldpc= 64,800)
extern uint16_t ldpc_tab_28_45NX[112][12];
// Table B.7: LDPC code identifier: 23/36 (nldpc = 64,800)
extern uint16_t ldpc_tab_23_36NX[117][12];
// Table B.8: LDPC code identifier: 25/36 (nldpc = 64,800)
extern uint16_t ldpc_tab_25_36NX[125][12];
// Table B.9:LDPC code identifier: 13/18 (nldpc = 64,800)
extern uint16_t ldpc_tab_13_18NX[130][11];
// Table B.10: LDPC code identifier: 7/9 (nldpc = 64,800)
extern uint16_t ldpc_tab_7_9NX[140][13];
//Table B.11: LDPC code identifier: 90/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_90_180NX[90][19];
//Table B.12: LDPC code identifier: 96/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_96_180NX[96][21];
// Table B.13: LDPC code identifier: 100/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_100_180NX[100][17];
// Table B.14: LDPC code identifier: 104/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_104_180NX[104][19];
// Table B.15: LDPC code identifier: 116/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_116_180NX[116][19];
// Table B.16: LDPC code identifier: 124/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_124_180NX[124][17];
// Table B.17: LDPC code identifier: 128/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_128_180NX[128][16];
// Table B.18: LDPC code identifier: 132/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_132_180NX[132][16];
// Table B.19: LDPC code identifier: 135/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_135_180NX[135][15];
// Table B.20: LDPC code identifier: 140/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_140_180NX[140][16];
// Table B.21: LDPC code identifier: 154/180 (nldpc = 64,800)
extern uint16_t ldpc_tab_154_180NX[154][14];
// Table B.22: LDPC code identifier: 18/30 (nldpc = 64,800)
extern uint16_t ldpc_tab_18_30NX[108][20];
// Table B.23: LDPC code identifier: 20/30 (nldpc = 64,800)
extern uint16_t ldpc_tab_20_30NX[120][17];
// Table B.24: LDPC code identifier: 22/30 (nldpc = 64,800)
extern uint16_t ldpc_tab_22_30NX[132][16];
// Addresses of parity bit accumulators for nldpc = 16,200 and nldpc = 32,400
// Table C.1: LDPC code identifier: 11/45 (nldpc = 16,200)
extern uint16_t ldpc_tab_11_45SX[11][11];
// Table C.2: LDPC code identifier: 4/15 (nldpc = 16,200)
extern uint16_t ldpc_tab_4_15SX[12][22];
// Table C.3: LDPC code identifier: 14/45 (nldpc = 16,200)
extern uint16_t ldpc_tab_14_45SX[14][13];
// Table C.4: LDPC code identifier: 7/15 (nldpc = 16,200)
extern uint16_t ldpc_tab_7_15SX[21][25];
// Table C.5: LDPC code identifier: 8/15 (nldpc = 16 200)
extern uint16_t ldpc_tab_8_15SX[24][22];
// Table C.6: LDPC code identifier: 26/45 (nldpc = 16,200)
extern uint16_t ldpc_tab_26_45SX[26][14];
// Table C.7: LDPC code identifier: 32/45 (nldpc = 16,200)
extern uint16_t ldpc_tab_32_45SX[32][13];
// Table C.8: LDPC code identifier: 1/5 (nldpc = 32,400)
extern uint16_t ldpc_tab_1_5MX[18][14];
// Table C.9: LDPC code identifier: 11/45 (nldpc = 32,400)
extern uint16_t ldpc_tab_11_45MX[22][11];
// Table C.10:LDPC code identifier: 1/3 (nldpc = 32,400)
extern uint16_t ldpc_tab_1_3MX[30][13];


// Constellation
typedef enum{ 
	m_RESERVED,
	m_SET_1,
	m_SET_2,
	m_BPSK, 
	m_QPSK,
	m_8PSK,
	m_8APSK,
	m_2_4_2APSK,
	m_16APSK,
	m_8_8APSK,
	m_4_12APSK,
	m_32APSK,
	m_4_8_4_16APSK,
	m_4_12_16rbAPSK,
	m_64APSK,
	m_16_16_16_16APSK,
	m_4_12_20_28APSK,
	m_8_16_20_20APSK,
	m_128APSK,
	m_256APSK
}Modulation;

// De-Interleaver type
typedef enum{
	I_0_N,
	I_0_S,
	I_0_M,
	I_00_N,
	I_00_M,
	I_00_S,
	I_012_N,
	I_012_S,
	I_102_N,
	I_102_S,
	I_210_N,
	I_210_S,
	I_0123_N,
	I_0123_S,
	I_0321_N,
	I_2103_N,
	I_2103_S,
	I_2130_S,
	I_2301_N,
	I_2310_N,
	I_3201_N,
	I_3201_S,
	I_3210_N,
	I_3012_N,
	I_3021_N,
	I_01234_N,
	I_21430_N,
	I_01234_S,
	I_40312_N,
	I_40213_N,
	I_41230_N,
	I_41230_S,
	I_10423_N,
	I_10423_S,
	I_201543_N,
	I_124053_N,
	I_4130256_N,
	I_421053_N,
	I_305214_N,
	I_520143_N,
	I_4250316_N,

	I_40372156_N,
	I_01234567_N,
	I_46320571_N,
	I_75642301_N,
	I_50743612_N

}InterleaveType;

typedef enum{
	MODCOD_OK,
	MODCOD_DUMMY,
	MODCOD_TYPE1,
	MODCOD_TYPE2,
	MODCOD_RESERVED,
	MODCOD_ERROR
}ModcodStatus;

typedef enum{
	BCH_N8,
	BCH_N10,
	BCH_N12,
	BCH_S12,
	BCH_M12
}BchTypes;

typedef enum{
	DVB_S2,
	DVB_S2X,
	DVB_S2XVLSNR
}DvbS2Type;

typedef enum{
	TEST_SDR,
	LIME_SDR,
	PLUTO_SDR,
	ERROR_SDR,
	HELP_SDR
}SdrType;

//
// Base Band Header
//
#define BB_TS 3
#define BB_GP 0
#define BB_GC 1

#define BB_SIS 1
#define BB_MIS 0

#define BB_CCM 1
#define BB_ACM 0

#define BB_RO_35 0b000
#define BB_RO_25 0b001
#define BB_RO_20 0b010
#define BB_RO_15 0b100
#define BB_RO_10 0b101
#define BB_RO_05 0b110

typedef struct{
	unsigned int ts_gs   : 2;
	unsigned int sis_mis : 1;
	unsigned int ccm_acm : 1;
	unsigned int issyi   : 1;
	unsigned int npd     : 1;
	unsigned int ro      : 3;
	uint8_t      mat2;
	uint16_t     upl;
	uint16_t     dfl;
	uint8_t      sync;
	uint16_t     syncd;
	unsigned int crc_ok  : 1;
}BBHeader;

typedef enum{
	RX_UNLOCKED_COARSE,
	RX_UNLOCKED_FINE,
	RX_LOCKED_DVB_S2,
	RX_LOCKED_DVB_S2P,
	RX_LOCKED_DVB_S2X,
	RX_LOCKED_DVB_S2XP,
	RX_LOCKED_DVB_S2XV,
	RX_LOCKED_DVB_S2XVP
}RxLock;

typedef struct{
	uint8_t modcod;
	int q;
	DvbS2Type s2_type;
	FrameType frame_type;
	BBHeader  bbh;
	int pilots;
	uint32_t nsyms;// Number of usuable data symbols
	int bsyms;// Number of info and pilot syms
	int fsyms;// Number of preamble + info + pilot syms
	int nsams;// number of samples in the frame.
	int nblocks; // A block is 90 syms
	Modulation mod;
	Modulation mod_class;
	CodeRate code_rate;
	InterleaveType itype;
	BchTypes bch;
	int pbch;
	int kbch;
	int nbch;
	int nldpc;// Number of bits in the LDPC block
	int nuldpc;// Number of data bits in the LDPC block
	int pldpc; // Number of parity bits the the LDPC block
	// Fixed area
	SdrType sdr_type;
	int ldpc_iterations;
	int np_frame_count;
	// working area
	RxLock lock;
	int sample_pointer;
	uint8_t fn;       // Frame buffer number
	double req_syrate;// Requested symbol rate
	double req_sarate;// Requested sample rate 2* symbol rate
	double act_sarate;// Actual received sample rate
	double req_freq;  // Requested frequency
	float phase_acc;  // Phase accumulator
	float phase_delta;
	char  format_text[80];
}RxFormat;

//
// Used in receiver, these only work for DVB-S2
//
#define BCONS 15
typedef struct{
	uint16_t size;
	uint16_t node[BCONS];
}LdpcBTab;

#define CCONS 31
typedef struct{
	uint16_t size;
	uint16_t node[CCONS];
}LdpcCTab;

#define MCONS 32
// Used to hold messages
typedef struct{
	LLR m[MCONS];
}LdpcMTab;


#define D_PI (2.0*M_PI)
#define SAMPLES_PER_SYMBOL 2

#define cmultReal(x,y)     ((x.re*y.re)-(x.im*y.im))
#define cmultImag(x,y)     ((x.re*y.im)+(x.im*y.re)) 
#define cmultRealConj(x,y) ((x.re*y.re)+(x.im*y.im))
#define cmultImagConj(x,y) ((x.im*y.re)-(x.re*y.im)) 


#define NORMAL_FRAME_BITS 64800
#define MEDIUM_FRAME_BITS 32400
#define SHORT_FRAME_BITS  16200

#define FEC_GPU_BLOCK_SIZE 360
#define LDPC_ITERATIONS 6

#define CHECK(call) { \
	cudaError_t err; \
	if((err=(call)) != cudaSuccess){ \
		fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
				__FILE__, __LINE__); \
		exit(1); \
	} \
}


// Test bench
void test(const char *filename, float sn);
void device_display_array(FComplex *ds, int len );
void device_display_array(float *ds, int len );
void device_display_array(uint8_t *ds, int len );
void host_display_array(FComplex *b, int len );
void host_display_array(float *b, int len );
void host_display_array(uint8_t *b, int len );
void device_display_array(uint16_t *ds, int len );
void host_display_array(uint16_t *b, int len );
void device_display_array(float *da, float *db, int len );

// RRC filter
short *rrc_make_filter(float roff, int ratio, int taps);

void    pl_decode_open(void);
uint8_t pl_decode(FComplex *in );
ModcodStatus pl_new_modcod( uint8_t modcod);
uint8_t pl_decode_preamble(FComplex *in);
ModcodStatus modcod_decode(uint8_t modcod);
void dvbs2_initialise(void);
bool isvalid_modcod(uint32_t modcod );

//void initialise_ldpc_decoder(void);

// Demapping functions
void     descrambler_open(void);
FComplex scramble_symbol(FComplex x, int n);
FComplex descramble_symbol(FComplex x, int n);
void     bb_derandomise(uint8_t *frame, int len);

// De-interleaver
typedef struct{
	uint8_t bit;// decoded bit
	LLR m;// Bit metric
}RxFrame;

//extern void (*deinterleaver)(float *m);
//void deinterleaver_new_frame(RxFrame *r, RxFormat *f);
//void set_deinterleaver(InterleaveType type);

// De mapper and de-interleaver
extern void (*demapin)(FComplex *d_s, LLR *d_m, float sn, uint8_t inter, size_t len);
void demapin_set(void);
void demap_update_constellation(FComplex *c, int len);

// Equaliser
void eq_reset(void);
FComplex eq_equalize_train_known(FComplex *in, FComplex train);
extern FComplex (*eq_data)(FComplex *in );
void eq_set_modulation(void);
void eq_open(void);
void eq_course_preamble( FComplex *in, FComplex *out);
void eq_stats(float &variance, float &mer);
FComplex eq_phase_correction(void);
void eq_de_rotate_estimate(FComplex *s, FComplex *r, int nr );
void eq_de_rotate_apply_inplace(FComplex *s, int nr );

// Constellation tables for the DFE
void contab_set(void);
void contab_open(void);
void contab_close(void);

// Receiver
void init_rx_stage1(void);
void rxs1_samples(FComplex *s, int n);
int  rx_frame_new_samples( FComplex *s, int n, RxFormat *f);

// Lime
int lime_open(double freq, double srate);
void lime_close(void);
int lime_tx_buffer(SComplex *s, int chan, int len );
int lime_rx_buffer(SComplex *s, int chan, int len );
int lime_rx_buffer(FComplex *s, int chan, int len );
int lime_rx_samples(FComplex *s, int len );
int lime_txrx_buffer(SComplex *tx, SComplex *rx, int chan, int len );
void lime_set_rx_sr( double srate );
double lime_get_rx_sr(void);

// adalm-pluto
int  pluto_open( double frequency, double srate );
void pluto_close(void);
int  pluto_rx_samples( SComplex **s);
void pluto_set_rx_sr(double sr);
double pluto_get_rx_sr(void);
void pluto_load_rrc_filter(float roff );

// BCH
void bch_decode_open(void);
void bch_decode_close(void);
extern int  (*bch_device_decode)(Bit *din, uint8_t *out);
void bch_set_device_decode(void);
int host_bch_pack_bits(uint8_t *out, Bit *in,  int len);
int  bch_host_byte_decode(uint8_t *in);

// BCH byte lookup table

typedef struct{
	uint64_t bch_r[3];
}BCHLookup;

void bch_d_copy_lookup(BCHLookup *lu);
void bch_select_device_lookup_table(void);
void bch_device_nframe_check( uint8_t *din, uint8_t *dout );

bool bch_h_byte_n_8_parity_check(  uint8_t *in, int len );
bool bch_h_byte_n_10_parity_check( uint8_t *in, int len );
bool bch_h_byte_n_12_parity_check( uint8_t *in, int len );
bool bch_h_byte_m_12_parity_check( uint8_t *in, int len );
bool bch_h_byte_s_12_parity_check( uint8_t *in, int len );
bool bch_d_byte_n_8_parity_check(  uint8_t *in, int len );
bool bch_d_byte_n_10_parity_check( uint8_t *in, int len );
bool bch_d_byte_n_12_parity_check( uint8_t *in, int len );
bool bch_d_byte_m_12_parity_check( uint8_t *in, int len );
bool bch_d_byte_s_12_parity_check( uint8_t *in, int len );

void bch_open_b( void );

// LDPC
void ldpc2_decode_open(void);
void ldpc2_decode_close(void);
void ldpc2_decode_set_fec(void);
void ldpc2_decode(LLR *d_m, uint8_t *d_bytes);
void ldpc2_decode(LLR *d_m, uint8_t *d_bytes, uint8_t *d_checks);
// Use callback
void ldpc2_decode(LLR *d_m );

extern void (*ldpc_decode)(LLR *d_m, Bit *d_bits);

void ldpc_decode_lookup_generate(void);
void ldpc_decode_open(void);
void ldpc_decode_close(void);
void bch_set_decoder(void);

// Preamble
void preamble_open(void);
void preamble_close(void);
void preamble_coeffs(float *in);
int  preamble_hunt_coarse( FComplex *in, int len );
int  preamble_hunt_fine( FComplex *in, int len, RxFormat *fmt, int &offset);

// Receiver A
//
// This updates the rate when we are hunting for a preamble
// It is passed the received frame length in samples and the
// frame length in samples of what we actually receive.
//
void rx_update_symrate( int rx, int ex );
//
// This uses the offset from the start of the symbol to
// generate the correction. This is usually called after sync has been
// acheived by the receiver.
//
void rx_update_symrate( int off );
// This applies the average correction of the samples
void rx_apply_symrate_adjust( void );
//
// Adjust the symbol rate by the rate amount
void rx_adjust_symrate( int rate );
//
// This returns the ratio of expected to received frame rate
// A value of 1.0 indicates we have achieved a symbol rate match
//
double rx_average_symrate(void);
// Reset the nuber of received symrate updates
void rx_reset_symrate(void);
// Return the number of sym rates;
int rx_symrate_count(void);
// Frequency error update
void rx_update_ferror( double ferror );
// This applies the average correction of the samples
void rx_apply_ferror_adjust( double frac );
// Return the current measured frequency error
double rx_average_ferror(void);


// Open the receiver
void receiver_open(void);
// Close the receiver
void receiver_close(void);
// Send a new block of samples to the receiver
void receiver_samples( FComplex *in, int len );
// Receiver has lost the signal and needs to go back into hunting
void receiver_los(void);
// Indicate the lock state of the receiver
bool receiver_islocked(void);

// Receiver B
void rxb_open(void);
void rxb_close(void);
void rxb_aos(void);
void rxb_output_serial( uint8_t *m, uint8_t *checks );
void rxb_dvbs2_process( FComplex *s, int len );
void rxb_dvbs2_pilots_process( FComplex *s, int len );
void rxb_dvbs2x_process( FComplex *s, int len );
void rxb_dvbs2x_pilots_process( FComplex *s, int len );
void rxb_dvbs2xvlsnr_process( FComplex *s, int len );
void rxb_dvbs2xvlsnr_pilots_process( FComplex *s, int len );

// BB Header
void bb_header_decode(uint8_t *b, BBHeader *h);
void bb_header_open(void);
uint8_t bb_calc_crc8( uint8_t *b, int len );
void bbh_crc_table(void);

void bb_build_crc8_table( void );

// Output routines
void data_output_transportstream(uint8_t *in, int len, BBHeader *h);
void data_output_open(void);
void data_output_close(void);

// Hardware library
void hw_adjust_sarate( double srate );
double hw_get_sarate( void );
void hw_set_rolloff( float ro);

// Control interface
void display_status(char *text, int n);

// Stats
void stats_open(void);
void stats_close(void);
void stats_bbh_errors_reset(void);
int  stats_bbh_errors_read(void);
void stats_bbh_errors_update(int n);
void stats_bch_errors_reset(void);
int  stats_bch_errors_read(void);
void stats_bch_errors_update(int n);
void stats_freq_error_update(float error);
void stats_freq_error_reset(void);
float stats_freq_error_read(void);
void stats_mag_update(float mag);
void stats_mag_reset(void);
float stats_mag_read(void);
void stats_update(void);
bool stats_update_read(void);
void stats_tp_rx_update(uint32_t n);
void stats_tp_rx_reset(void);
uint64_t stats_tp_rx_read(void);
void stats_tp_er_update(uint32_t n);
void stats_tp_er_reset(void);
uint64_t stats_tp_er_read(void);
void stats_ldpc_fes_update(uint32_t fes);
uint32_t stats_ldpc_fes_read(void);
void stats_mer_update(float mer);
float stats_mer_read(void);

// Frequency zigzag routines
void zigzag_reset(void);
double zigzag_delta(void);
void zigzag_set_inc_and_max(double freq, uint16_t max);

#endif
