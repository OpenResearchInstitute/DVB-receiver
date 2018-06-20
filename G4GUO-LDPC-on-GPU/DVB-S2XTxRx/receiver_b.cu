#include <stdio.h>
#include <queue>
#include <deque>
#include <list>
#include <queue>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include "dvbs2_rx.h"

using namespace std;

extern FComplex g_preamble[90];
extern FComplex vl_snr_preamble[900];
extern RxFormat g_format;
extern float g_pream[57];

#define SYM1 0.7071067f

FComplex *m_frame[2];

static int32_t  m_sidx; // Symbol index into frame
static uint32_t m_pidx; // Payload index into frame
static uint32_t m_scdx; // scrambler index index
static uint32_t m_pbidx;// Pilot block counter
static FComplex m_pilots = { SYM1,  SYM1 };
static int m_fails;

// Save the received preamble
static FComplex m_pream[PREAM_N];
static FComplex m_pilot_syms[36];
static FComplex m_pilot_refs[36];
// output  queue
static queue <uint8_t *> m_output_q;
// Mutexes
static pthread_mutex_t mutex_output;

static int modcod_errors;

// Output thread
//
// Array to store received data
//
static uint8_t m_last_ts_gs;
//
// Called from the LDPC callback
//
void rxb_output_serial( uint8_t *m, uint8_t *checks ){
    int n_checks = 0;
    int es = 0;
    int eb = 0;
	for( uint32_t i = 0; i < NP_FRAMES; i++){
        // Do the BCH decoding
		uint8_t *b = &m[i*(g_format.nbch/8)];
		if(checks[i]){
			n_checks++;
		    // We can only correct a certain number of BCH errors as it is so expensive
		    if( n_checks < 50 ){
                es += bch_host_byte_decode( b );
		    }
		}
        // De Randomise the BB frame
        int len = g_format.kbch/8;
        bb_derandomise( b, len );
        // We now have a octet stream for further processing
        BBHeader h;
        bb_header_decode( b, &h);
//        for( int i = 0; i < 10; i++) printf("%.2X ",b[i]);
//        printf("\n");
        // Remove the header length
        len = len - 10;
        if(h.crc_ok){
            m_last_ts_gs = h.ts_gs;
            g_format.bbh = h;
            switch(h.ts_gs){
            case BB_TS:
	            data_output_transportstream( &b[10], len, &h);
	            break;
            default:
	            break;
            }
        }else{
             switch(m_last_ts_gs){
            case BB_TS:
	            data_output_transportstream(&b[10], len, &h);
	            break;
            default:
	            break;
            }
            eb++;
        }
    }
	stats_ldpc_fes_update(n_checks);
    stats_bch_errors_update(es);
    stats_bbh_errors_update(eb);
    // Decide when to loose the frame sync
	if(eb >= (NP_FRAMES-10) )
		m_fails++;
	else
		m_fails = 0;
	if(m_fails > 5 ){
		m_fails = 0;
		receiver_los();
	}
	n_checks = 0;
    stats_update();
}
/*
void rx_frame_pilots_vlsnr( FComplex *s, RxFormat *f){
	Demod dm;
	int isam; // sample index
	int iscr; // scambler index
	int ipre; // preamble index

	isam = iscr = ipre = 0;

	eq_reset();
	deinterleaver_new_frame(m_rframe, f);
	//
	// train on the preamble
	//
    if( f->isvlsnr == 0){
    	// standard preamble
    	for( int i = 0; i < 90; i++){
			eq_equalize_train_known(&s[isam], m_preamble[ipre]);
			isam += 2;
			ipre += 1;
    	}
    }else{
    	// Very low SNR preamble
    	for( int i = 0; i < 900; i++){
    		eq_equalize_train_known(&s[isam], vl_snr_preamble[ipre]);
			isam += 2;
			ipre += 1;
    	}
    }

    // See whether we have pilots or not
	if( f->pilots == 0 ){
		// No Pilot symbols
		for( int i = 0; i < f->nblocks; i++){
			for( int j = 0; j < 90; j++){
				dm = eq_equalize_data(&s[isam], iscr);
				deinterleaver(dm.m);
				isam += 2;
				iscr += 1;
			}
		}
	}else{
		// There are Pilot symbols
		// pilot block is 16 blocks + pilot symbols
		int pilot_blocks     = f->nblocks/16;
		int remainder_blocks = f->nblocks%16;
		// See if there is an exact multiple of 16 x 90 symbols
		if( remainder_blocks == 0 ){
			// Exact fit so no need to final pilots
			pilot_blocks--;
			remainder_blocks = 16;
		}

		for( int i = 0; i < pilot_blocks; i++){
			for( int j = 0; j < 16; j++){
				for( int k = 0; k < 90; k++){
					dm = eq_equalize_data(&s[isam], iscr);
					deinterleaver(dm.m);
					isam += 2;
					iscr += 1;
				}
			}
			// Add pilot symbols
			for( int k = 0; k < 36; k++){
				eq_equalize_train_known(&s[isam], scramble_symbol(m_pilots[0], iscr));
				isam += 2;
				iscr += 1;
			}
		}
		// Last block in frame (no pilots to follow)
		for( int j = 0; j < remainder_blocks; j++){
			for( int k = 0; k < 90; k++){
				dm = eq_equalize_data(&s[isam], iscr);
				deinterleaver(dm.m);
				isam += 2;
				iscr += 1;
			}
		}
	}
	f->sample_pointer = 0;
	f->samples_left_to_write = f->samples_per_frame;
}
*/
//
// RX2 stuff
//
// Process frequency 'error' caused by symbol rate error

static double m_sr_error; // Accumulator
static int    m_e_count;  // Nr consecutive errors before a re-sync

void rxb_symbol_rate_error(void){

//	double e = m_sr_error*g_format.act_sarate/(NP_FRAMES * g_format.nsams * 2);
	double e = m_sr_error/(NP_FRAMES * g_format.nsams * 2);

	// Tracing
//	double d = m_sr_error;
//	printf("E error %f %f\n",d, e);

	m_sr_error = 0;

	if(fabs(e) > (TRACK_RANGE/4))
		m_e_count--;
	else
		m_e_count = 20;

	if(m_e_count <= 0){
		//
	    // Adjust the symbol rate to remove the error
		// The delay is to allow the system to recover after a symbol rate change
		//
		rx_adjust_symrate( -e );
		//receiver_los();
	    m_e_count = 20;
	}
}
//
// Device data
//
static Bit      *d_bytes;
static FComplex *d_s;
static LLR      *d_m;
//
// End of frame actions
//
void rxb_end_of_frame( void ){
	if(++g_format.np_frame_count >= NP_FRAMES){
		g_format.np_frame_count = 0;
	    // Copy the decoded symbols from host memory space to device memory space
		CHECK(cudaMemcpyAsync( d_s, m_frame[g_format.fn], sizeof(FComplex)*g_format.nsyms*NP_FRAMES, cudaMemcpyHostToDevice));
	    // Demap and deinterleave (if necessary) the symbols
		float variance,mer;
		eq_stats(variance,mer);
		stats_mer_update(mer);
		variance = 1.0;
	    demapin( d_s, d_m, variance, g_format.itype, g_format.nsyms );
        // Decode
	    ldpc2_decode( d_m );
	    g_format.fn = (g_format.fn+1)%2;// Toggle between the two memory banks
	    rxb_symbol_rate_error();
	    // Reset equaliser for next 128 frames
	    eq_reset();
	}
	// Use the equaliser error to correct the frequency offse
    // Reset all the frame pointers
	m_pidx  = g_format.np_frame_count;
	m_sidx  = 0;
	m_scdx  = 0;
	m_pbidx = 0;
}

/////////////////////////////////////////////////////////////////////////////////
//
// Time and space tracking routines
//
/////////////////////////////////////////////////////////////////////////////////

//
// Start of block tracking
//
void rxb_track_sob(void){
}
//
// End of block tracking
//
void rxb_track_eob(void){
}

void rxb_frequency_error(FComplex *in){
    FComplex suma,sumb;

    FComplex *s = in;

    int j = 0;

    suma.re = suma.im = sumb.re = sumb.im = 0;

    // Differentially decode
    for( int cn = 0; cn < 25; cn++){
        suma.re   += cmultImagConj(s[j],s[j+2])*g_pream[cn];
        suma.im   += cmultRealConj(s[j],s[j+2])*g_pream[cn];
        j += 2;
    }
    j = 52;
    for( int cn = 25; cn < 57; cn++){
        sumb.re   += cmultImagConj(s[j],s[j+2])*g_pream[cn];
        sumb.im   += cmultRealConj(s[j],s[j+2])*g_pream[cn];
        j += 4;
    }
    printf("%f %f     %f %f\n",suma.re, suma.im, sumb.re, sumb.im);
    suma.re=suma.im = 0;

//    if(fabs(suma.re - sumb.re) > fabs(suma.re + sumb.re)){
    if(sumb.re < 0){
    	suma.re = suma.re - sumb.re;
       	suma.im = suma.im - sumb.im;
    }else{
    	suma.re = suma.re + sumb.re;
     	suma.im = suma.im + sumb.im;
    }

//suma.im -= 0.00004;
    // Calculate the frequency error
    float ferror = -atan(suma.im/suma.re)/SAMPLES_PER_SYMBOL;// At symbolrate divide by 2 to get error at sample rate
    // Update the stats
    stats_freq_error_update(ferror);
    // Update the error
    if(fabs(ferror) > 0.001) ferror = ferror > 0 ? 0.001 : -0.001;
    rx_update_ferror(ferror);
}
//
// Find the optimum preamble position and calculate it's time and frequency error
//
int rxb_track(FComplex *in){
    float max   = 0;
    int max_pos = 0;
	double suma  = 0;
	double sumb  = 0;
    FComplex *ms;

    FComplex *s = &in[-TRACK_RANGE/2];

    for( int i = 0; i < TRACK_RANGE; i++){
    	int j = 0;
    	int cn;
    	suma = sumb = 0;

        // Differentially decode
        for( cn = 0; cn < 25; cn++){
    	    suma   += cmultImagConj(s[j],s[j+2])*g_pream[cn];
   	        j+=2;
        }
        j = 52;
        for( cn = 25; cn < 57; cn++){
    	    sumb   += cmultImagConj(s[j],s[j+2])*g_pream[cn];
    	    j+=4;
        }
        if(fabs(suma - sumb) > fabs(suma + sumb)){
        	suma = fabs(suma - sumb);
        }else{
        	suma = fabs(suma + sumb);
        }
	    if(suma > max){
        	max     = suma;
        	max_pos = i;
            ms      = s;
        }
        s++;
	}
    stats_mag_update(max);

    // Calculate the maximum position offset
    max_pos = max_pos -(TRACK_RANGE/2);
	m_sr_error = m_sr_error + max_pos;
    rxb_frequency_error(ms);
    return max_pos;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Frame processing routines
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Standard DVB-S2 frame
//
void rxb_dvbs2_process( FComplex *s, int len ){

	int skip = 0;

	rxb_track_sob();

	for( int i = 0; i < len; i += 2 ){
    	if(m_sidx == 0 ){
    		//
    		// We should be aligned with the preamble
    		//
        	//eq_reset();
        	// look for any sync movement
    		if(skip == 0){
   		        int off = rxb_track(&s[i]);
   		        if( off != 0 ){
   		    	    if(off%2){
   		    		    if(off > 0 )
   		    			    off++;
   		    		    else
   		    			    off--;
   		    	    }
   		       	    //s = &s[off];
   		       	    //len -= off;
 //  	    		    eq_reset();
   	    		    m_sidx -= off/2;
   	    		    //printf("Adjust %d\n",off);
   	    		    if(m_sidx < 0) skip = 1;
   		        }
   		    }else{
   		    	skip = 0;
   		    }
   	    }

    	if((m_sidx >= 0)&&(m_sidx < 90)){
    		// Train on preamble
    		m_pream[m_sidx] = eq_equalize_train_known( &s[i], g_preamble[m_sidx]);
    	}

    	// test the modcod has not changed
    	if(m_sidx == (PREAM_N-1)){
            eq_de_rotate_estimate( &m_pream[58], &g_preamble[58], 32 );
            eq_de_rotate_apply_inplace( &m_pream[26], 32 );
            uint8_t modcod = pl_decode( &m_pream[26] );
            if(modcod != g_format.modcod) modcod_errors++;
            if(modcod_errors > 128){
            	receiver_los();
            	modcod_errors = 0;
            }
            //pl_new_modcod( modcod );
    	}

    	if(m_sidx >= 90){
   		    // Payload
			m_frame[g_format.fn][m_pidx] = descramble_symbol(eq_data(&s[i]), m_scdx++);
			m_pidx += NP_FRAMES;
    	}

    	m_sidx += 1;

    	if(m_sidx == (g_format.fsyms)){
    		// End of frame actions
    		rxb_end_of_frame();
    	}
    }
	rxb_track_eob();
}
//
// Standard DVB-S2 frame with pilots
//
void rxb_dvbs2_pilots_process( FComplex *s, int len ){

	int skip = 0;

	rxb_track_sob();

	for( int i = 0; i < len; i += 2 ){
    	if(m_sidx == 0 ){
    		// Start of frame actions
    		//eq_reset();
        	// look for any sync movement
    		if(skip == 0){
   		        int off = rxb_track(&s[i]);
   		        if( off != 0 ){
   		    	    if(off%2){
   		    		    if(off > 0 )
   		    			    off++;
   		    		    else
   		    			    off--;
   		    	    }
   		       	    //s = &s[off];
   		       	    //len -= off;
   	    		    eq_reset();
   	    		    m_sidx -= off/2;
   	    		    //printf("Adjust %d\n",off);
   	    		    if(m_sidx < 0) skip = 1;
   		        }
   		    }else{
   		    	skip = 0;
   		    }
    	}
    	if((m_sidx >= 0)&&(m_sidx < 90)){
    		// Train on preamble
    		m_pream[m_sidx] = eq_equalize_train_known(&s[i], g_preamble[m_sidx]);
    	}

    	// test the modcod has not changed
    	if(m_sidx == (PREAM_N-1)){
            eq_de_rotate_estimate( &m_pream[58], &g_preamble[58], 32 );
            eq_de_rotate_apply_inplace( &m_pream[26], 32 );
            uint8_t modcod = pl_decode( &m_pream[26] );
           // pl_new_modcod( modcod );
    	}

    	if( m_sidx >= 90 ){
    		if( m_pbidx < 1440 ){
    			// info symbols
    			m_frame[g_format.fn][m_pidx] = descramble_symbol(eq_data(&s[i]), m_scdx++);
    			m_pidx += NP_FRAMES;
    		}
    		if( m_pbidx >= 1440 ){
    			// Pilot symbols
    			int index = m_pbidx - 1440;
    			m_pilot_refs[index] = scramble_symbol(m_pilots, m_scdx++);
    			m_pilot_syms[index] = eq_equalize_train_known(&s[i], m_pilot_refs[index]);
    			if(index == 35 ){
    	            eq_de_rotate_estimate( m_pilot_syms, m_pilot_refs, 36 );
    			}
    		}
    		m_pbidx++;
    		if( m_pbidx >= 1476 ) m_pbidx = 0;;
    	}

    	m_sidx += 1;
    	if( m_sidx == 90 ) m_pbidx = 0;

    	if(m_sidx == g_format.fsyms){
    		// End of frame actions
    		rxb_end_of_frame();
    	}
    }
	rxb_track_eob();
}
//
// DVB-S2X frame
//
void rxb_dvbs2x_process( FComplex *s, int len ){
	rxb_dvbs2_process( s, len );

}
//
// DVB-S2X frame with pilots
//
void rxb_dvbs2x_pilots_process( FComplex *s, int len ){
	rxb_dvbs2_pilots_process( s, len );
}
//
// DVB-S2X VL_SNR frame
//
void rxb_dvbs2xvlsnr_process( FComplex *s, int len ){

}
//
// DVB-S2X VL_SNR frame with pilots
//
void rxb_dvbs2xvlsnr_pilots_process( FComplex *s, int len ){

}
//
// AOS
//
void rxb_aos(void){
	m_sidx      = 0; // Symbol index into frame
	m_pidx      = 0; // Payload index into frame
	m_scdx      = 0; // scrambler index index
	m_pbidx     = 0; // Pilot block counter
	rxb_track_sob();
	eq_reset();

}

//////////////////////////////////////////////////////////////////////////////////////
//
// Initialise, these are called from the main receiver code
//
//////////////////////////////////////////////////////////////////////////////////////

void rxb_open(void){

	d_bytes    = NULL;
	d_s        = NULL;
	d_m        = NULL;
    m_frame[0] = NULL;
    m_frame[1] = NULL;

	CHECK(cudaMalloc((void**)&d_bytes, sizeof(Bit)*FRAME_SIZE_NORMAL*NP_FRAMES/8));
	CHECK(cudaMalloc((void**)&d_s,     sizeof(FComplex)*FRAME_SIZE_NORMAL*NP_FRAMES/2));
	CHECK(cudaMalloc((void**)&d_m,     sizeof(LLR)*FRAME_SIZE_NORMAL*NP_FRAMES));
	CHECK(cudaHostAlloc((void**)&m_frame[0], sizeof(FComplex)*FRAME_SIZE_NORMAL*NP_FRAMES/2, cudaHostAllocDefault));
	CHECK(cudaHostAlloc((void**)&m_frame[1], sizeof(FComplex)*FRAME_SIZE_NORMAL*NP_FRAMES/2, cudaHostAllocDefault));

	g_format.fn = 0;

	preamble_open();
	descrambler_open();
	pl_decode_open();
	ldpc2_decode_open();
	bch_decode_open();
	bb_header_open();
	contab_open();
	eq_open();
	data_output_open();
	stats_open();
	pthread_mutex_init( &mutex_output, NULL );
/*
	if(pthread_create( &m_thread, NULL, output_thread, NULL ) != 0 )
    {
        printf("Unable to start output thread\n");
    }
*/

}
//
// Close
//
void rxb_close(void){

	if(d_bytes != NULL) CHECK(cudaFree(d_bytes));
	if(d_s     != NULL) CHECK(cudaFree(d_s));
	if(d_m     != NULL) CHECK(cudaFree(d_m));

	bch_decode_close();
	ldpc2_decode_close();
	contab_close();
	data_output_close();
	preamble_close();
	stats_close();
	if(m_frame[0] != NULL) cudaFreeHost(m_frame[0]);
	if(m_frame[1] != NULL) cudaFreeHost(m_frame[1]);
}
