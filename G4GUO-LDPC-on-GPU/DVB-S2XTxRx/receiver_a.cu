/*
 * receiver_a.cu
 *
 *  Created on: 13 Jan 2017
 *      Author: charles
 */
#include <math.h>
#include <stdio.h>
#include "dvbs2_rx.h"


extern RxFormat g_format;

static int m_pream_fine_count;
//
// Routines to track the difference in symbol rate
// between the transmitter and receiver
//
typedef struct{
	int rx;
	int ex;
	int vs;
}SymrateVotes;

#define SYM_LN 10
SymrateVotes m_symr[SYM_LN];
static int m_symrate_count;
//
// This updates the rate when we are hunting for a preamble
// It is passed the received frame length in samples and the
// frame length in samples of what we actually receive.
//
void rx_update_symrate( int rx, int ex ){
	m_symrate_count++;
//	printf("%d %d\n",rx,ex);
   // Update
   for( int i = 0; i < SYM_LN; i++){
	   if((m_symr[i].rx == rx)&&(m_symr[i].ex == ex)){
		   m_symr[i].vs++;
		   return;
	   }
   }
   // Match not found
   for( int i = 0; i < SYM_LN; i++){
	   if(m_symr[i].vs == 0){
		   m_symr[i].rx = rx;
		   m_symr[i].ex = ex;
		   m_symr[i].vs++;
		   return;
	   }
   }
   // No empty slots, find lowest and reuse that
   int l = m_symr[0].vs;
   int m = 0;

   for( int i = 1; i < SYM_LN; i++){
	   if(m_symr[i].vs < l){
		   l = m_symr[i].vs;
		   m = i;
	   }
   }
   // Found lowest votes re-use it
   m_symr[m].rx = rx;
   m_symr[m].ex = ex;
   m_symr[m].vs = 1;
}
// Reset the nuber of received symrate updates
void rx_reset_symrate(void){
    for( int i = 0; i < SYM_LN; i++){
	    m_symr[i].vs = 0;
	}
    m_symrate_count = 0;
}
//
// This applies the average correction of the samples
//
void rx_apply_symrate_adjust( void ){

	if( m_symrate_count == 0 ) return;

	// Find the one with the most votes
	int h = 0;
	int m = 0;

	for( int i = 0; i < SYM_LN; i++){
		if(m_symr[i].vs > h){
	        h = m_symr[i].vs;
			m = i;
	    }
	}
	double ex = m_symr[m].ex;
	double rx = m_symr[m].rx;

    double cor   = ex/rx;
//    double delta = g_format.req_sarate*0.05;// Allowed margin
    double delta = 3000;// Allowed margin

    g_format.act_sarate = hw_get_sarate();
    g_format.act_sarate = ceil(g_format.act_sarate*cor);

    if(fabs(g_format.act_sarate - g_format.req_sarate) < delta){
        hw_adjust_sarate( g_format.act_sarate );
        g_format.act_sarate = hw_get_sarate();
    }else{
        hw_adjust_sarate( g_format.req_sarate );
        g_format.act_sarate = hw_get_sarate();
    }
    rx_reset_symrate();
}
void rx_adjust_symrate( int rate ){

    g_format.act_sarate  = hw_get_sarate();
    g_format.act_sarate += rate*2;
    hw_adjust_sarate( g_format.act_sarate );
}
//
// Routines to track the frequency error
// between the transmitter and receiver
//
static double m_ferror_acc   = 0;
static int    m_ferror_count = 0;
//
// Update the ferror
//
void rx_update_ferror( double ferror ){
    // Throw away senseless values
	if(fabs(ferror) < 1.35){
        m_ferror_acc += ferror;
        m_ferror_count++;
	}
}
//
// This applies the average correction of the samples
//
void rx_apply_ferror_adjust( double frac ){

	if(m_ferror_count == 0 ) return;

	double val = m_ferror_acc/m_ferror_count;
    //
    // Only update by a fraction of the correction
    // This is needed because of the delay in the samples
    //
    val *= frac;

    if(g_format.lock == RX_UNLOCKED_COARSE)
        g_format.phase_delta  =  val;
    else
        g_format.phase_delta +=  val;

    m_ferror_acc   = 0;
    m_ferror_count = 0;
}
//
// Return the current measured frequency error
//
double rx_average_ferror(void){
	if(m_ferror_count == 0) return 0;
    double val = m_ferror_acc/m_ferror_count;
    return val;
}

//
// This will always be called with a multiple of 90 x 2 samples
//
// The in array pointer is offset into valid memory this will
// allow tracking of symbol rate errors into the negative region
//

void receiver_samples( FComplex *in, int len ){
	int offset;

//	if(in == NULL) return;
	switch(g_format.lock){
	case RX_UNLOCKED_COARSE:
	    if(preamble_hunt_coarse( in, len )){
	    	m_pream_fine_count = 20;
	    	g_format.lock = RX_UNLOCKED_FINE;
	    }else{
           // g_format.phase_delta  =  0;
	    }
		break;
	case RX_UNLOCKED_FINE:
		// If a lock has been achieved set appropriate state
	    if(preamble_hunt_fine( in, len, &g_format, offset)){
	    	FComplex *s = &in[offset];
	    	int nlen    = len - offset;
	    	// A good preamble has been found
	    	rxb_aos();
	    	if(g_format.s2_type == DVB_S2){
	    		if(g_format.pilots){
	    			g_format.lock = RX_LOCKED_DVB_S2P;
	    			rxb_dvbs2_pilots_process( s, nlen );
	    		}else{
	    			g_format.lock = RX_LOCKED_DVB_S2;
	    			rxb_dvbs2_process( s, nlen );
	    		}
	    	}else{
	    		if(g_format.s2_type == DVB_S2X){
		    		if(g_format.pilots){
		    			g_format.lock = RX_LOCKED_DVB_S2XP;
		    			rxb_dvbs2x_pilots_process( s, nlen );
		    		}else{
		    			g_format.lock = RX_LOCKED_DVB_S2X;
		    			rxb_dvbs2x_process( s, nlen );
		    		}
	    		}else{
	    			if(g_format.s2_type == DVB_S2XVLSNR){
			    		if(g_format.pilots){
			    			g_format.lock = RX_LOCKED_DVB_S2XVP;
			    			rxb_dvbs2xvlsnr_pilots_process( s, nlen );
			    		}else{
			    			g_format.lock = RX_LOCKED_DVB_S2XV;
			    			rxb_dvbs2xvlsnr_process( s, nlen );
			    		}
	    			}
	    		}
	    	}
	    }else{
    	    m_pream_fine_count--;
    	    if( m_pream_fine_count < 0 ){
	    	    g_format.lock        = RX_UNLOCKED_COARSE;
	    	    g_format.phase_delta = 0;
	    	    g_format.act_sarate  = g_format.req_sarate;
	    	    hw_adjust_sarate(g_format.act_sarate);
	            g_format.phase_delta  =  0;
    	    }
	    }
		break;
	case RX_LOCKED_DVB_S2:
		// Down convert in frequency
		rxb_dvbs2_process( in, len );
		break;
	case RX_LOCKED_DVB_S2P:
		rxb_dvbs2_pilots_process( in, len );
		break;
	case RX_LOCKED_DVB_S2X:
		rxb_dvbs2x_process( in, len );
		break;
	case RX_LOCKED_DVB_S2XP:
		rxb_dvbs2x_pilots_process( in, len );
		break;
	case RX_LOCKED_DVB_S2XV:
		rxb_dvbs2xvlsnr_process( in, len );
		break;
	case RX_LOCKED_DVB_S2XVP:
		rxb_dvbs2xvlsnr_pilots_process( in, len );
		break;
	default:
		break;
	}
	// Copy the end of the samples into the negative index of the array to enable
	// tracking of symbol rate errors
	//
	memcpy(&in[-RX_SAMPLE_HISTORY+1],&in[len-RX_SAMPLE_HISTORY],sizeof(FComplex)*RX_SAMPLE_HISTORY);
	// Is it time for a diagnostic
}
//
// Called when receiver looses lock
//
void receiver_los(void){
	g_format.lock = RX_UNLOCKED_FINE;
//	g_format.phase_delta = 0;
	m_pream_fine_count = 10;
	zigzag_reset();
}
bool receiver_islocked(void){
	if((g_format.lock == RX_UNLOCKED_COARSE)||(g_format.lock == RX_UNLOCKED_FINE))
		return false;
	else
		return true;
}
void receiver_open(void){
	g_format.lock = RX_UNLOCKED_COARSE;
	g_format.sample_pointer = 0;
	g_format.phase_delta = 0;
	data_output_open();
	rxb_open();
}
void receiver_close(void){
	 g_format.sample_pointer = 0;
	 rxb_close();
}

