#include "dvbs2_rx.h"
#include <stdio.h>

extern RxFormat g_format;

void hw_adjust_sarate( double srate ){
	if(g_format.sdr_type == PLUTO_SDR){
		pluto_set_rx_sr( srate );//
	}
	if(g_format.sdr_type == LIME_SDR){
		lime_set_rx_sr( srate);//
	}
}
double hw_get_sarate( void ){
	if(g_format.sdr_type == PLUTO_SDR){
		double sr = pluto_get_rx_sr();
		if(g_format.req_syrate < 260000)
			return sr/8;
		else
	        return sr;
	}
	if(g_format.sdr_type == LIME_SDR){
	    return (lime_get_rx_sr());
	}
	return 0;
}

void hw_set_rolloff(float ro){
	if(g_format.sdr_type == PLUTO_SDR){
		pluto_load_rrc_filter(ro);
	}
	if(g_format.sdr_type == LIME_SDR){
	}
}
