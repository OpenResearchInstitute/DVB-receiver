#include <memory.h>
#include <stdio.h>
#include "dvbs2_rx.h"

extern RxFormat g_format;

typedef struct{
	float mag;
	int32_t idx;
	float offset;
	FComplex ph;
}SearchResult;

// Scratch pad memory for preamble
FComplex h_preamble[360];
static FComplex *d_pream = NULL;
static SearchResult *d_search;

#define BN  192
#define SBN 180

// Coefficients used to hunt for the preamble

__constant__ float d_pream_c[57];

__global__ void pream_search_offset_old(FComplex *in, float offset, SearchResult *search )
{
    int n = (blockIdx.x * blockDim.x)+threadIdx.x;

    __shared__ float   fmag[BN];
    __shared__ FComplex   ph[SBN];
    __shared__ uint8_t findex[BN];
    __shared__ FComplex m[BN*2];

    findex[n] = n;

    m[n*2]     = in[n*2];
    m[(n*2)+1] = in[(n*2)+1];

    if( n < SBN){
	    float mag = 0;
	    FComplex suma;
	    FComplex sumb;
	    int cn = 0;
	    mag = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int i = 0; i <= 48; i+=2){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    suma.re   += (cmultImagConj(s[i],s[i+2]) + cmultImagConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    suma.im   += (cmultRealConj(s[i],s[i+2]) + cmultRealConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    cn++;
	    }
	    for( int i = 52; i <= 176; i+=4){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    sumb.re   += (cmultImagConj(s[i],s[i+2]) + cmultImagConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    sumb.im   += (cmultRealConj(s[i],s[i+2]) + cmultRealConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    cn++;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
    	search->offset = 0;
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
		    // 0 is maximum
		    search->mag = fmag[findex[0]];
		    search->ph  = ph[findex[0]];
		    search->idx = findex[0];
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		        // 1 is maximum
		        search->mag = fmag[findex[1]];
			    search->ph  = ph[findex[1]];
		        search->idx = findex[1];
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
		            // 2 is maximum
		            search->mag = fmag[findex[2]];
				    search->ph  = ph[findex[2]];
		            search->idx = findex[2];
	            }
		    }
		}
	}
    // Now sync the threads
	__syncthreads();
	// Do the -ve frequency offset
    FComplex nco;
    FComplex val;

    // Apply frequency offset to the input samples
    nco.re = cos(-(n*2)*offset);
    nco.im = sin(-(n*2)*offset);
    val.re = cmultReal(m[n*2],nco);
    val.im = cmultImag(m[n*2],nco);
    m[n*2] = val;

    nco.re = cos(-((n*2)+1)*offset);
    nco.im = sin(-((n*2)+1)*offset);
    val.re = cmultReal(m[(n*2)+1],nco);
    val.im = cmultImag(m[(n*2)+1],nco);
    m[(n*2)+1] = val;
    if( n < SBN){
	    float mag = 0;
	    FComplex suma;
	    FComplex sumb;
	    int cn = 0;
	    mag = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int i = 0; i <= 48; i+=2){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    suma.re   += (cmultImagConj(s[i],s[i+2]) + cmultImagConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    suma.im   += (cmultRealConj(s[i],s[i+2]) + cmultRealConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    cn++;
	    }
	    for( int i = 52; i <= 176; i+=4){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    sumb.re   += (cmultImagConj(s[i],s[i+2]) + cmultImagConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    sumb.im   += (cmultRealConj(s[i],s[i+2]) + cmultRealConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    cn++;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
	    	if( fmag[findex[0]] > search->mag){
		        // 0 is maximum
		        search->mag    = fmag[findex[0]];
			    search->ph     = ph[findex[0]];
		        search->idx    = findex[0];
		        search->offset = -offset;
	    	}
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		    	if( fmag[findex[1]] > search->mag){
		            // 1 is maximum
		            search->mag    = fmag[findex[1]];
				    search->ph     = ph[findex[1]];
		            search->idx    = findex[1];
			        search->offset = -offset;
		    	}
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
	    	    	if( fmag[findex[2]] > search->mag){
		                // 2 is maximum
		                search->mag    = fmag[findex[2]];
		    		    search->ph     = ph[findex[2]];
		                search->idx    = findex[2];
				        search->offset = -offset;
	    	    	}
	            }
		    }
		}
	}
    // Now sync the threads
	__syncthreads();
    // Apply frequency offset to the input samples
    nco.re = cos(2*(n*2)*offset);
    nco.im = sin(2*(n*2)*offset);
    val.re = cmultReal(m[n*2],nco);
    val.im = cmultImag(m[n*2],nco);
    m[n*2] = val;

    nco.re = cos(2*((n*2)+1)*offset);
    nco.im = sin(2*((n*2)+1)*offset);
    val.re = cmultReal(m[(n*2)+1],nco);
    val.im = cmultImag(m[(n*2)+1],nco);
    m[(n*2)+1] = val;

    if( n < SBN){
	    float mag = 0;
	    FComplex suma;
	    FComplex sumb;
	    int cn = 0;
	    mag = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int i = 0; i <= 48; i+=2){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    suma.re   += (cmultImagConj(s[i],s[i+2]) + cmultImagConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    suma.im   += (cmultRealConj(s[i],s[i+2]) + cmultRealConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    cn++;
	    }
	    for( int i = 52; i <= 176; i+=4){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    sumb.re   += (cmultImagConj(s[i],s[i+2]) + cmultImagConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    sumb.im   += (cmultRealConj(s[i],s[i+2]) + cmultRealConj(s[i+1],s[i+3]))*d_pream_c[cn];
		    cn++;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
	    	if( fmag[findex[0]] > search->mag){
		        // 0 is maximum
		        search->mag    = fmag[findex[0]];
			    search->ph     = ph[findex[0]];
		        search->idx    = findex[0];
		        search->offset = offset;
	    	}
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		    	if( fmag[findex[1]] > search->mag){
		            // 1 is maximum
		            search->mag    = fmag[findex[1]];
				    search->ph     = ph[findex[1]];
		            search->idx    = findex[1];
			        search->offset = offset;
		    	}
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
	    	    	if( fmag[findex[2]] > search->mag){
		                // 2 is maximum
		                search->mag    = fmag[findex[2]];
		    		    search->ph     = ph[findex[2]];
		                search->idx    = findex[2];
				        search->offset = offset;
	    	    	}
	            }
		    }
		}
	}
}

__global__ void pream_search_offset(FComplex *in, float offset, SearchResult *search )
{
    int n = (blockIdx.x * blockDim.x)+threadIdx.x;

    __shared__ float   fmag[BN];
    __shared__ FComplex   ph[SBN];
    __shared__ uint8_t findex[BN];
    __shared__ FComplex m[BN*2];

    findex[n] = n;

    m[n*2]     = in[n*2];
    m[(n*2)+1] = in[(n*2)+1];

    if( n < SBN){
	    float mag = 0;
	    FComplex suma;
	    FComplex sumb;
	    int i = 0;
	    mag = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int cn = 0; cn < 25; cn++){
		    mag       += cmultRealConj(s[i],s[i]);
		    suma.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    suma.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    i += 2;
	    }
	    i = 52;
	    for( int cn = 25; cn < 57; cn++){
		    mag       += cmultRealConj(s[i],s[i]);
		    sumb.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    sumb.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    i += 4;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re)/mag;
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re)/mag;
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
    	search->offset = 0;
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
		    // 0 is maximum
		    search->mag = fmag[findex[0]];
		    search->ph  = ph[findex[0]];
		    search->idx = findex[0];
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		        // 1 is maximum
		        search->mag = fmag[findex[1]];
			    search->ph  = ph[findex[1]];
		        search->idx = findex[1];
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
		            // 2 is maximum
		            search->mag = fmag[findex[2]];
				    search->ph  = ph[findex[2]];
		            search->idx = findex[2];
	            }
		    }
		}
	}
    // Now sync the threads
	__syncthreads();
	// Do the -ve frequency offset
    FComplex nco;
    FComplex val;

    // Apply frequency offset to the input samples
    nco.re = cos(-(n*2)*offset);
    nco.im = sin(-(n*2)*offset);
    val.re = cmultReal(m[n*2],nco);
    val.im = cmultImag(m[n*2],nco);
    m[n*2] = val;

    nco.re = cos(-((n*2)+1)*offset);
    nco.im = sin(-((n*2)+1)*offset);
    val.re = cmultReal(m[(n*2)+1],nco);
    val.im = cmultImag(m[(n*2)+1],nco);
    m[(n*2)+1] = val;
    if( n < SBN){
	    float mag = 0;
	    FComplex suma;
	    FComplex sumb;
	    int i = 0;
	    mag = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int cn = 0; cn < 25; cn++){
		    mag       += cmultRealConj(s[i],s[i]);
		    suma.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    suma.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    i+=2;
	    }
	    i = 52;
	    for( int cn = 25; cn < 57; cn++){
		    mag       += cmultRealConj(s[i],s[i]);
		    sumb.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    sumb.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    i += 4;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re)/mag;
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re)/mag;
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
	    	if( fmag[findex[0]] > search->mag){
		        // 0 is maximum
		        search->mag    = fmag[findex[0]];
			    search->ph     = ph[findex[0]];
		        search->idx    = findex[0];
		        search->offset = -offset;
	    	}
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		    	if( fmag[findex[1]] > search->mag){
		            // 1 is maximum
		            search->mag    = fmag[findex[1]];
				    search->ph     = ph[findex[1]];
		            search->idx    = findex[1];
			        search->offset = -offset;
		    	}
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
	    	    	if( fmag[findex[2]] > search->mag){
		                // 2 is maximum
		                search->mag    = fmag[findex[2]];
		    		    search->ph     = ph[findex[2]];
		                search->idx    = findex[2];
				        search->offset = -offset;
	    	    	}
	            }
		    }
		}
	}
    // Now sync the threads
	__syncthreads();
    // Apply frequency offset to the input samples
    nco.re = cos(2*(n*2)*offset);
    nco.im = sin(2*(n*2)*offset);
    val.re = cmultReal(m[n*2],nco);
    val.im = cmultImag(m[n*2],nco);
    m[n*2] = val;

    nco.re = cos(2*((n*2)+1)*offset);
    nco.im = sin(2*((n*2)+1)*offset);
    val.re = cmultReal(m[(n*2)+1],nco);
    val.im = cmultImag(m[(n*2)+1],nco);
    m[(n*2)+1] = val;

    if( n < SBN){
	    float mag = 0;
	    FComplex suma;
	    FComplex sumb;
	    int i = 0;
	    mag = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int cn = 0; cn < 25; cn++){
		    mag       += cmultRealConj(s[i],s[i]);
		    suma.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    suma.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    i += 2;
	    }
	    i = 52;
	    for( int cn = 25; cn < 57; cn++){
		    mag       += cmultRealConj(s[i],s[i]);
		    sumb.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    sumb.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    i += 4;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re)/mag;
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re)/mag;
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
	    	if( fmag[findex[0]] > search->mag){
		        // 0 is maximum
		        search->mag    = fmag[findex[0]];
			    search->ph     = ph[findex[0]];
		        search->idx    = findex[0];
		        search->offset = offset;
	    	}
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		    	if( fmag[findex[1]] > search->mag){
		            // 1 is maximum
		            search->mag    = fmag[findex[1]];
				    search->ph     = ph[findex[1]];
		            search->idx    = findex[1];
			        search->offset = offset;
		    	}
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
	    	    	if( fmag[findex[2]] > search->mag){
		                // 2 is maximum
		                search->mag    = fmag[findex[2]];
		    		    search->ph     = ph[findex[2]];
		                search->idx    = findex[2];
				        search->offset = offset;
	    	    	}
	            }
		    }
		}
	}
}
__global__ void pream_search_old(FComplex *in, SearchResult *search )
{
    int n = (blockIdx.x * blockDim.x)+threadIdx.x;

    __shared__ float   fmag[BN];
    __shared__ FComplex   ph[SBN];
    __shared__ uint8_t findex[BN];
    __shared__ FComplex m[BN*2];

    findex[n] = n;

    m[n*2]     = in[n*2];
    m[(n*2)+1] = in[(n*2)+1];

    if( n < SBN){
	    float mag = 0;
	    FComplex suma;
	    FComplex sumb;
	    int cn = 0;
	    mag = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int i = 0; i <= 48; i+=2){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    suma.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    suma.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    cn++;
	    }
	    for( int i = 52; i <= 176; i+=4){
		    mag    += s[i].re*s[i].re + s[i].im*s[i].im;
		    sumb.re   += cmultImagConj(s[i],s[i+2])*d_pream_c[cn];
		    sumb.im   += cmultRealConj(s[i],s[i+2])*d_pream_c[cn];
		    cn++;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re)/sqrt(mag);
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
    	search->offset = 0;
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
		    // 0 is maximum
		    search->mag = fmag[findex[0]];
		    search->ph  = ph[findex[0]];
		    search->idx = findex[0];
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		        // 1 is maximum
		        search->mag = fmag[findex[1]];
			    search->ph  = ph[findex[1]];
		        search->idx = findex[1];
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
		            // 2 is maximum
		            search->mag = fmag[findex[2]];
				    search->ph  = ph[findex[2]];
		            search->idx = findex[2];
	            }
		    }
		}
	}
}

__global__ void pream_search(FComplex *in, SearchResult *search )
{
    int n = (blockIdx.x * blockDim.x)+threadIdx.x;

    __shared__ float   fmag[BN];
    __shared__ FComplex   ph[SBN];
    __shared__ uint8_t findex[BN];
    __shared__ FComplex m[BN*2];

    findex[n] = n;

    m[n*2]     = in[n*2];
    m[(n*2)+1] = in[(n*2)+1];

    if( n < SBN){
	    FComplex suma;
	    FComplex sumb;
	    int i = 0;
	    suma.re = sumb.re = suma.im = sumb.im = 0;
	    FComplex *s = &m[n];
	    for( int cn = 0; cn < 25; cn++){
		    suma.re   += (cmultImagConj(s[i],s[i+2]))*d_pream_c[cn];
		    suma.im   += (cmultRealConj(s[i],s[i+2]))*d_pream_c[cn];
		    i+=2;
	    }
	    i = 52;
	    for( int cn = 25; cn < 57; cn++){
		    sumb.re   += (cmultImagConj(s[i],s[i+2]))*d_pream_c[cn];
		    sumb.im   += (cmultRealConj(s[i],s[i+2]))*d_pream_c[cn];
		    i+=4;
	    }
	    if(fabs(suma.re+sumb.re) > fabs(suma.re-sumb.re)){
	    	fmag[n] = fabs(suma.re+sumb.re);
	    	ph[n].re = suma.re + sumb.re;
	    	ph[n].im = suma.im + sumb.im;
	    }else{
	    	fmag[n] = fabs(suma.re-sumb.re);
	    	ph[n].re = suma.re - sumb.re;
	    	ph[n].im = suma.im - sumb.im;
	    }
    } else {
        fmag[n] = 0;
    }
    // Now sync the remaining threads
	__syncthreads();
	// Do the comparison
    if( n < (BN>>1)) if( fmag[findex[n+(BN>>1)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>1)];
	// skip unused threads
    if( n < (BN>>2) ) if( fmag[findex[n+(BN>>2)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>2)];
	// skip unused threads
    if( n < (BN>>3) ) if( fmag[findex[n+(BN>>3)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>3)];
	// skip unused threads
    if( n < (BN>>4) ) if( fmag[findex[n+(BN>>4)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>4)];
	// skip unused threads
    if( n < (BN>>5) ) if( fmag[findex[n+(BN>>5)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>5)];
	// skip unused threads
    if( n < (BN>>6) ) if( fmag[findex[n+(BN>>6)]] > fmag[findex[n]]) findex[n] = findex[n+(BN>>6)];
	// skip unused threads
    if( n == 0 ) {
    	search->offset = 0;
        // do final comparison
	    if((fmag[findex[0]] > fmag[findex[1]])&&(fmag[findex[0]] > fmag[findex[2]])){
		    // 0 is maximum
		    search->mag = fmag[findex[0]];
		    search->ph  = ph[findex[0]];
		    search->idx = findex[0];
	    }else{
	        if((fmag[findex[1]] > fmag[findex[0]])&&(fmag[findex[1]] > fmag[findex[2]])){
		        // 1 is maximum
		        search->mag = fmag[findex[1]];
			    search->ph  = ph[findex[1]];
		        search->idx = findex[1];
	        }else{
	            if((fmag[findex[2]] > fmag[findex[0]])&&(fmag[findex[2]] > fmag[findex[1]])){
		            // 2 is maximum
		            search->mag = fmag[findex[2]];
				    search->ph  = ph[findex[2]];
		            search->idx = findex[2];
	            }
		    }
		}
	}
}

float g_pream[90];
//
// Called to set the preamble search coefficients
//
void preamble_coeffs(float *in){

	memcpy(g_pream,in,sizeof(float)*57);

	cudaError_t cudaStatus = cudaMemcpyToSymbol(d_pream_c, in, sizeof(float)*57);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyToSymbol %s %d failed!\n",__FILE__,__LINE__);
	}
}
/*
//
// Calculate the frequency error
//
float preamble_freq_error_old(FComplex *in, float offset){
	FComplex suma;
	FComplex sumb;
    int cn = 0;
    FComplex m[180];
    FComplex nco;
    suma.re = suma.im = sumb.re = sumb.im = 0;

    // Translate in frequency by the offset
    for( int i = 0; i < 180; i++){
    	nco.re =  cos(i*offset);
    	nco.im =  sin(i*offset);
    	m[i].re = cmultReal(in[i],nco);
    	m[i].im = cmultImag(in[i],nco);
    }

    // Differentially decode
    for( int i = 0; i <= 48; i+=2){
	    suma.re   += (cmultImagConj(m[i],m[i+2]) + cmultImagConj(m[i+1],m[i+3]))*g_pream[cn];
	    suma.im   += (cmultRealConj(m[i],m[i+2]) + cmultRealConj(m[i+1],m[i+3]))*g_pream[cn];
	    cn++;
    }
    for( int i = 52; i <= 176; i+=4){
	    sumb.re   += (cmultImagConj(m[i],m[i+2]) + cmultImagConj(m[i+1],m[i+3]))*g_pream[cn];
	    sumb.im   += (cmultRealConj(m[i],m[i+2]) + cmultRealConj(m[i+1],m[i+3]))*g_pream[cn];
	    cn++;
    }
    if(sumb.re > 0){
    	suma.re += sumb.re;
    	suma.im += sumb.im;
    }else{
    	suma.re -= sumb.re;
    	suma.im -= sumb.im;
    }
    float delta = offset;// at sample rate
    if( suma.re != 0 ){
    	delta           += atan(suma.im/suma.re)/2;// At symbolrate divide by 2 to get sample rate
//  	    float freq_error = (delta*(g_format.req_srate*2)/(2*M_PI));// convert symbol rate to sample rate to get frequency
//    	printf("Freq offset %5.1f\tHz", freq_error);
    }
	return delta;
}
*/
float preamble_freq_error(FComplex *in, float offset){
	FComplex suma;
	FComplex sumb;
    int cn = 0;
    FComplex m[180];
    FComplex nco;
    suma.re = suma.im = sumb.re = sumb.im = 0;

    // Translate in frequency by the offset
    for( int i = 0; i < 180; i++){
    	nco.re =  cos(i*offset);
    	nco.im =  sin(i*offset);
    	m[i].re = cmultReal(in[i],nco);
    	m[i].im = cmultImag(in[i],nco);
    }

    // Differentially decode
    for( int i = 0; i <= 48; i+=2){
	    suma.re   += cmultImagConj(m[i],m[i+2])*g_pream[cn];
	    suma.im   += cmultRealConj(m[i],m[i+2])*g_pream[cn];
	    cn++;
    }
    for( int i = 52; i <= 176; i+=4){
	    sumb.re   += cmultImagConj(m[i],m[i+2])*g_pream[cn];
	    sumb.im   += cmultRealConj(m[i],m[i+2])*g_pream[cn];
	    cn++;
    }
    if((suma.re*sumb.re) > 0){
    	suma.re += sumb.re;
    	suma.im += sumb.im;
    }else{
    	suma.re -= sumb.re;
    	suma.im -= sumb.im;
    }
    float delta = offset;// at sample rate
    if( suma.re != 0 ){
    	delta           += atan(suma.im/suma.re)/2;// At symbolrate divide by 2 to get sample rate
    }
	return delta;
}
void preamble_freq_correct(FComplex *in, FComplex *out, int len, float offset){
    FComplex nco;

    // Translate in frequency by the offset
    for( int i = 0; i < len; i++){
    	nco.re =  cos(i*offset);
    	nco.im =  sin(i*offset);
    	out[i].re = cmultReal(in[i],nco);
    	out[i].im = cmultImag(in[i],nco);
    }
}
int preamble_vote(FComplex *m){
	int sof = 0;
	int pls = 0;
	int cn = 0;
	int sn;
	float var;

    // Differentially decode
    for( int i = 0; i < 25; i++){
    	sn = i*2;
	    var = cmultImagConj(m[sn],m[sn+2])*g_pream[cn];
	    if(var > 0 ) sof++;
	    cn++;
    }
    for( int i = 26; i < 89; i+=2){
    	sn = i*2;
	    var   = cmultImagConj(m[sn],m[sn+2])*g_pream[cn];
	    if(var > 0 )
	    	pls++;
	    else
	    	pls--;
	    cn++;
    }
    return( sof + abs(pls));
}
void preamble_open(void){
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&d_pream, S2_BLOCK_N*S2_SPS_N*2*sizeof(FComplex));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc %s %d failed!\n",__FILE__,__LINE__);
		return;
	}
	cudaStatus = cudaMalloc((void**)&d_search, sizeof(SearchResult));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc %s %d failed!\n",__FILE__,__LINE__);
		return;
	}
}
void preamble_close(void){
    if(d_pream  != NULL) cudaFree(d_pream);
    if(d_search != NULL) cudaFree(d_search);
}

//
// Samples are at 2x the symbol rate
// 3 correlators are used
//
#define VTHRESH 55

int preamble_hunt_coarse( FComplex *in, int len ){
    SearchResult sr;
	FComplex pls[64];
    static int last_frame_length  = 0;
    static int frame_length       = 0;
    static int exp_frame_length   = 0;
    static int frame_length_merit = 0;
    static int offset_merit       = 0;

    if(d_pream == NULL ) return 0;

    for( int i = 0; i < len; i += SBN){
    	cudaMemcpy(d_pream,       &d_pream[SBN], SBN*sizeof(FComplex), cudaMemcpyDeviceToDevice);
    	cudaMemcpy(&d_pream[SBN], &in[i],        SBN*sizeof(FComplex), cudaMemcpyHostToDevice);
    	// <<< N_BLOCKS, M_THREADS_PER_BLOCK >>>
    	pream_search_offset<<<1,BN>>>( d_pream, 0.1f, d_search );
    	// Import results into host workspace
    	cudaMemcpy(&sr, d_search, sizeof(SearchResult), cudaMemcpyDeviceToHost);
    	int start = i + sr.idx - SBN;//Start position of frame
		//printf("Mag %f\n",sr.mag);

    	if( sr.mag > 0.3){
    		//printf("Mag %f\n",sr.mag);
    	    if( preamble_vote(&in[start]) >= VTHRESH){

    		    frame_length += sr.idx;
    		    // If we have a string of frames the same length then we probably have valid frames
    		    if(abs(frame_length - last_frame_length) < 8){
       	            rx_update_symrate( frame_length, exp_frame_length );
    		        frame_length_merit += 1;
    		    }else{
    		        frame_length_merit = 1;
    		    }

    		    last_frame_length = frame_length;

    		    // We are trying to drive the frequency to the zero correlator
    		    if( sr.offset == 0 ){
    			    offset_merit += 1;
    		    }else{
    			    offset_merit = 0;
    		    }

    	        // Fine adjust for any phase error
    	        double ferror = (sr.offset - (atan(sr.ph.im/sr.ph.re)/2));
    		    // Start of new frame
    		    frame_length = SBN - sr.idx;
    	        // Calculate the new expected frame length
    		    FComplex m[200];
    		    preamble_freq_correct( &in[start-(KN/2)], m, 200, ferror );
        	    eq_course_preamble( &m[KN/2], pls);
        	    uint8_t modcod = pl_decode( pls );
        	    modcod_decode( modcod );
        	    exp_frame_length = g_format.nsams;
        	    // Update the frequency error
    	        //rx_update_ferror( ferror*0.01 );
        	    // all done
    	    }else{
    		    frame_length += SBN;
    	    }
    	}else{
		    frame_length += SBN;
    	}
    }


    if((frame_length_merit > 7)&&(offset_merit >= 2)){
//    	printf("Fine sync\n");
        rx_apply_symrate_adjust();
        frame_length_merit = 0;
        offset_merit       = 0;
    	return 1;
    }
	return 0;
}
int preamble_hunt_fine( FComplex *in, int len, RxFormat *fmt, int &offset ){
    SearchResult sr;
    SearchResult msr;
	FComplex pls[64];
    static uint8_t last_modcod;
    msr.mag = 0;

    for( int i = 0; i < len; i += SBN){
    	cudaMemcpy(d_pream,       &d_pream[SBN], SBN*sizeof(FComplex), cudaMemcpyDeviceToDevice);
    	cudaMemcpy(&d_pream[SBN], &in[i],        SBN*sizeof(FComplex), cudaMemcpyHostToDevice);
    	// <<< N_BLOCKS, M_THREADS_PER_BLOCK >>>
    	pream_search<<<1,BN>>>( d_pream, d_search );
    	cudaMemcpy(&sr, d_search, sizeof(SearchResult), cudaMemcpyDeviceToHost);
    	int start = i + sr.idx - SBN;//Start position of frame
    	if(preamble_vote(&in[start]) >= VTHRESH){
    	    // Fine adjust for any phase error
    	    double ferror = (sr.offset - (atan(sr.ph.im/sr.ph.re)/2));
    	    rx_update_ferror( ferror*0.01 );
        	// all done
    	}
    	// Find maximum preamble in this block
        if(  sr.mag     > msr.mag){
             msr.mag    = sr.mag;
        	 msr.idx    = sr.idx + i - SBN;
        	 msr.offset = sr.offset;
        }
    }
    //
    // Using the most likely preamble start position
    //
    msr.idx &= 0xFFFFFE;
    // As we are using 2 samples per symbol, use an even offset (bodge)
    // Estimate the frequency error
    float ferror = preamble_freq_error( &in[msr.idx], 0 );
    if((fabs(ferror) < 0.1 )){
        int votes = preamble_vote(&in[msr.idx]);
        if( votes >= VTHRESH){
        	// Decode the frame type being received
       		FComplex m[200];
       		preamble_freq_correct( &in[msr.idx-(KN/2)], m, 200, ferror );
           	eq_course_preamble( &m[KN/2], pls);
            uint8_t modcod = pl_decode( pls );
            if(last_modcod == modcod){
    	        // Only allow valid DVB-S2 codes to cause a lock
            	if(isvalid_modcod(modcod) == true){
//            		printf("MODCOD %d\n",modcod);
       		    	pl_new_modcod(g_format.modcod);
        	        offset = msr.idx;
        	        // Move to the first preamble in the buffer
        	        while((offset - g_format.fsyms*2) > 0 ) offset -= g_format.fsyms*2;
    	    	    return 1;
    	        }
            }
            last_modcod = modcod;
        }
    }
	return 0;
}
