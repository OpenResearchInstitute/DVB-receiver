//
// This file demaps the received symbols, calculates LLRs the de-interleaves if needed
//
#include <math.h>
#include <stdio.h>
#include "dvbs2_rx.h"

extern RxFormat g_format;

// There will always be a multiple of 90 symbols
// Interleaver row offsets

#define PSK8NC1 ((FRAME_SIZE_NORMAL*NP_FRAMES)/3)
#define PSK8NC2 (((FRAME_SIZE_NORMAL*NP_FRAMES)/3)*2)

#define PSK8SC1 (FRAME_SIZE_SHORT*NP_FRAMES/3)
#define PSK8SC2 ((FRAME_SIZE_SHORT*NP_FRAMES/3)*2)

#define APSK16NC1 (FRAME_SIZE_NORMAL*NP_FRAMES/4)
#define APSK16NC2 ((FRAME_SIZE_NORMAL*NP_FRAMES/4)*2)
#define APSK16NC3 ((FRAME_SIZE_NORMAL*NP_FRAMES/4)*3)

#define APSK16SC1 (FRAME_SIZE_SHORT*NP_FRAMES/4)
#define APSK16SC2 ((FRAME_SIZE_SHORT*NP_FRAMES/4)*2)
#define APSK16SC3 ((FRAME_SIZE_SHORT*NP_FRAMES/4)*3)

#define APSK32NC1 (FRAME_SIZE_NORMAL*NP_FRAMES/5)
#define APSK32NC2 ((FRAME_SIZE_NORMAL*NP_FRAMES/5)*2)
#define APSK32NC3 ((FRAME_SIZE_NORMAL*NP_FRAMES/5)*3)
#define APSK32NC4 ((FRAME_SIZE_NORMAL*NP_FRAMES/5)*4)

#define APSK32SC1 (FRAME_SIZE_SHORT*NP_FRAMES/5)
#define APSK32SC2 ((FRAME_SIZE_SHORT*NP_FRAMES/5)*2)
#define APSK32SC3 ((FRAME_SIZE_SHORT*NP_FRAMES/5)*3)
#define APSK32SC4 ((FRAME_SIZE_SHORT*NP_FRAMES/5)*4)

#define APSK64NC1 (FRAME_SIZE_NORMAL*NP_FRAMES/6)
#define APSK64NC2 ((FRAME_SIZE_NORMAL*NP_FRAMES/6)*2)
#define APSK64NC3 ((FRAME_SIZE_NORMAL*NP_FRAMES/6)*3)
#define APSK64NC4 ((FRAME_SIZE_NORMAL*NP_FRAMES/6)*4)
#define APSK64NC5 ((FRAME_SIZE_NORMAL*NP_FRAMES/6)*5)

#define APSK64SC1 (FRAME_SIZE_SHORT*NP_FRAMES/6)
#define APSK64SC2 ((FRAME_SIZE_SHORT*NP_FRAMES/6)*2)
#define APSK64SC3 ((FRAME_SIZE_SHORT*NP_FRAMES/6)*3)
#define APSK64SC4 ((FRAME_SIZE_SHORT*NP_FRAMES/6)*4)
#define APSK64SC5 ((FRAME_SIZE_SHORT*NP_FRAMES/6)*5)

#define APSK128NC1 (FRAME_SIZE_NORMAL*NP_FRAMES/7)
#define APSK128NC2 ((FRAME_SIZE_NORMAL*NP_FRAMES/7)*2)
#define APSK128NC3 ((FRAME_SIZE_NORMAL*NP_FRAMES/7)*3)
#define APSK128NC4 ((FRAME_SIZE_NORMAL*NP_FRAMES/7)*4)
#define APSK128NC5 ((FRAME_SIZE_NORMAL*NP_FRAMES/7)*5)
#define APSK128NC6 ((FRAME_SIZE_NORMAL*NP_FRAMES/7)*6)

#define APSK128SC1 (FRAME_SIZE_SHORT*NP_FRAMES/7)
#define APSK128SC2 ((FRAME_SIZE_SHORT*NP_FRAMES/7)*2)
#define APSK128SC3 ((FRAME_SIZE_SHORT*NP_FRAMES/7)*3)
#define APSK128SC4 ((FRAME_SIZE_SHORT*NP_FRAMES/7)*4)
#define APSK128SC5 ((FRAME_SIZE_SHORT*NP_FRAMES/7)*5)
#define APSK128SC6 ((FRAME_SIZE_SHORT*NP_FRAMES/7)*6)

#define APSK256NC1 (FRAME_SIZE_NORMAL*NP_FRAMES/8)
#define APSK256NC2 ((FRAME_SIZE_NORMAL*NP_FRAMES/8)*2)
#define APSK256NC3 ((FRAME_SIZE_NORMAL*NP_FRAMES/8)*3)
#define APSK256NC4 ((FRAME_SIZE_NORMAL*NP_FRAMES/8)*4)
#define APSK256NC5 ((FRAME_SIZE_NORMAL*NP_FRAMES/8)*5)
#define APSK256NC6 ((FRAME_SIZE_NORMAL*NP_FRAMES/8)*6)
#define APSK256NC7 ((FRAME_SIZE_NORMAL*NP_FRAMES/8)*7)

#define APSK256SC1 (FRAME_SIZE_SHORT*NP_FRAMES/8)
#define APSK256SC2 ((FRAME_SIZE_SHORT*NP_FRAMES/8)*2)
#define APSK256SC3 ((FRAME_SIZE_SHORT*NP_FRAMES/8)*3)
#define APSK256SC4 ((FRAME_SIZE_SHORT*NP_FRAMES/8)*4)
#define APSK256SC5 ((FRAME_SIZE_SHORT*NP_FRAMES/8)*5)
#define APSK256SC6 ((FRAME_SIZE_SHORT*NP_FRAMES/8)*6)
#define APSK256SC7 ((FRAME_SIZE_SHORT*NP_FRAMES/8)*7)

typedef struct{
	float one;
	float zero;
}Em;

// Device memory constellations

__constant__ FComplex d_const[256];

void (*demapin)(FComplex *d_s, LLR *d_m, float sn, uint8_t inter, size_t len );
__global__ void d_dvbs2_qpsk3(FComplex *s, LLR *m, float sa, float sb){
    uint32_t n  = (blockIdx.x * blockDim.x     ) + threadIdx.x;
    uint32_t n0 = (blockIdx.x * blockDim.x * 2 ) + threadIdx.x;
    uint32_t n1 = (blockIdx.x * blockDim.x * 2 ) + blockDim.x + threadIdx.x;
    float e, min;
    float ma[2];
    int min_pos = 0;

    min = CERROR(s[n],d_const[0]);

    for( int i = 1; i < 4; i++){
    	e = CERROR(s[n],d_const[i]);
    	if( e < min ){
    		min     = e;
    		min_pos = i;
    	}
    }
 //   min = -1.0/__logf(min);
    (min_pos&0b01) ? ma[0] = 0.5/min : ma[0] = -0.5/min;
    (min_pos&0b10) ? ma[1] = 0.5/min : ma[1] = -0.5/min;

    m[n0] = UCLAMP(ma[1]);
    m[n1] = UCLAMP(ma[0]);
    if( n0 == 0 ) printf("METRIC %d %d %f\n",m[n0],m[n1], min);
}

__global__ void d_dvbs2_qpsk2(FComplex *s, LLR *m, float sa, float sb){
    uint32_t n  = (blockIdx.x * blockDim.x     ) + threadIdx.x;
    uint32_t n0 = (blockIdx.x * blockDim.x * 2 ) + threadIdx.x;
    uint32_t n1 = (blockIdx.x * blockDim.x * 2 ) + blockDim.x + threadIdx.x;
    Em em[2];
    float e;

    em[0].one = em[0].zero = 10;
    em[1].one = em[1].zero = 10;

    for( int i = 0; i < 4; i++){
    	e = CERROR(s[n],d_const[i]);
     	if(i&0b01){
    		if(e < em[0].one) em[0].one = e;
    	}else{
    		if(e < em[0].zero) em[0].zero = e;
    	}
    	if(i&0b10){
    		if(e < em[1].one) em[1].one = e;
    	}else{
    		if(e < em[1].zero) em[1].zero = e;
    	}
    }
    m[n0] = UCLAMP((em[0].zero - em[0].one)*sb);
    m[n1] = UCLAMP((em[1].zero - em[1].one)*sb);
    if(m[n0] == 0) printf("METRIC %d %d\n",m[n0],m[n1]);
}
__global__ void d_dvbs2_qpsk(FComplex *s, LLR *m, float sa, float sb){
    uint32_t n  = (blockIdx.x * blockDim.x     ) + threadIdx.x;
    uint32_t n0 = (blockIdx.x * blockDim.x * 2 ) + threadIdx.x;
    uint32_t n1 = (blockIdx.x * blockDim.x * 2 ) + blockDim.x + threadIdx.x;
    Em em[2];
    float e;
    em[0].one = em[0].zero = 0;
    em[1].one = em[1].zero = 0;

    for( int i = 0; i < 4; i++){
    	e = CERROR(s[n],d_const[i]);
    	e = sa*(__expf(-e*sb));
    	(i&0b01) ? em[0].one += e : em[0].zero += e;
    	(i&0b10) ? em[1].one += e : em[1].zero += e;
    }
    m[n0] = UCLAMP(__logf(em[1].one/em[1].zero));
    m[n1] = UCLAMP(__logf(em[0].one/em[0].zero));
//    if(m[n0] == 0) printf("METRIC %d %d\n",m[n0],m[n1]);
//    m[(n*2)]   = UCLAMP(__logf(em[1].one/em[1].zero));
//    m[(n*2)+1] = UCLAMP(__logf(em[0].one/em[0].zero));
}
__global__ void d_dvbs2_8psk(FComplex *s, LLR *m, float sa, float sb, uint8_t inter){
    int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    Em em[3];
    float e;
    em[0].one = em[0].zero = 0;
    em[1].one = em[1].zero = 0;
    em[2].one = em[2].zero = 0;

    for( int i = 0; i < 8; i++){
    	e = CERROR(s[n],d_const[i]);
    	e = sa*__expf(-e*sb);
    	(i&0b001) ? em[0].one += e : em[0].zero += e;
    	(i&0b010) ? em[1].one += e : em[1].zero += e;
    	(i&0b100) ? em[2].one += e : em[2].zero += e;
    }

    // All threads will take the same path
    switch(inter){
    case I_012_N:
    	m[n+PSK8NC2] = UCLAMP(__logf(em[0].one/em[0].zero));
    	m[n+PSK8NC1] = UCLAMP(__logf(em[1].one/em[1].zero));
    	m[n]         = UCLAMP(__logf(em[2].one/em[2].zero));
    	break;
    case I_012_S:
    	m[n+PSK8SC2] = UCLAMP(__logf(em[0].one/em[0].zero));
    	m[n+PSK8SC1] = UCLAMP(__logf(em[1].one/em[1].zero));
    	m[n]         = UCLAMP(__logf(em[2].one/em[2].zero));
    	break;
    case I_102_N:
    	m[n+PSK8NC2] = UCLAMP(__logf(em[0].one/em[0].zero));
    	m[n]         = UCLAMP(__logf(em[1].one/em[1].zero));
    	m[n+PSK8NC1] = UCLAMP(__logf(em[2].one/em[2].zero));
    	break;
    case I_102_S:
    	m[n+PSK8SC2] = UCLAMP(__logf(em[0].one/em[0].zero));
    	m[n]         = UCLAMP(__logf(em[1].one/em[1].zero));
    	m[n+PSK8SC1] = UCLAMP(__logf(em[2].one/em[2].zero));
    	break;
   case I_210_N:
    	m[n]         = UCLAMP(__logf(em[0].one/em[0].zero));
    	m[n+PSK8NC1] = UCLAMP(__logf(em[1].one/em[1].zero));
    	m[n+PSK8NC2] = UCLAMP(__logf(em[2].one/em[2].zero));
    	break;
    case I_210_S:
    	m[n] =         UCLAMP(__logf(em[0].one/em[0].zero));
    	m[n+PSK8SC1] = UCLAMP(__logf(em[1].one/em[1].zero));
    	m[n+PSK8SC2] = UCLAMP(__logf(em[2].one/em[2].zero));
    	break;
    }
}
__global__ void d_dvbs2_8apsk(FComplex *s, LLR *m, float sb, uint8_t inter ){

	int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    Em em[3];
    float e;
    em[0].one = em[0].zero = 10;
    em[1].one = em[1].zero = 10;
    em[2].one = em[2].zero = 10;

    for( uint8_t i = 0; i < 8; i++){
    	e = CERROR(s[n],d_const[i]);
    	if(i&0b001){
    		if(e < em[0].one) em[0].one = e;
    	}else{
    		if(e < em[0].zero) em[0].zero = e;
    	}
    	if(i&0b010){
    		if(e < em[1].one) em[1].one = e;
    	}else{
    		if(e < em[1].zero) em[1].zero = e;
    	}
    	if(i&0b100){
    		if(e < em[2].one) em[2].one = e;
    	}else{
    		if(e < em[2].zero) em[2].zero = e;
    	}
    }
    switch(inter){
    case I_012_N:
    	m[n+PSK8NC2] = UCLAMP((em[0].zero - em[0].one)*sb);
    	m[n+PSK8NC1] = UCLAMP((em[1].zero - em[1].one)*sb);
    	m[n]         = UCLAMP((em[2].zero - em[2].one)*sb);
    	break;
    case I_012_S:
    	m[n+PSK8SC2] = UCLAMP((em[0].zero - em[0].one)*sb);
    	m[n+PSK8SC1] = UCLAMP((em[1].zero - em[1].one)*sb);
    	m[n]         = UCLAMP((em[2].zero - em[2].one)*sb);
    	break;
    }
}

__global__ void d_dvbs2_16apsk(FComplex *s, LLR *m, float sb, uint8_t inter ){

	int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    Em em[4];
    float e;
    em[0].one = em[0].zero = 10;
    em[1].one = em[1].zero = 10;
    em[2].one = em[2].zero = 10;
    em[3].one = em[3].zero = 10;

    for( uint8_t i = 0; i < 16; i++){
    	e = CERROR(s[n],d_const[i]);
    	if(i&0b0001){
    		if(e < em[0].one) em[0].one = e;
    	}else{
    		if(e < em[0].zero) em[0].zero = e;
    	}
    	if(i&0b0010){
    		if(e < em[1].one) em[1].one = e;
    	}else{
    		if(e < em[1].zero) em[1].zero = e;
    	}
    	if(i&0b0100){
    		if(e < em[2].one) em[2].one = e;
    	}else{
    		if(e < em[2].zero) em[2].zero = e;
    	}
    	if(i&0b1000){
    		if(e < em[3].one) em[3].one = e;
    	}else{
    		if(e < em[3].zero) em[3].zero = e;
    	}
    }
    switch(inter){
    case I_0123_N:
    	m[n+APSK16NC3] = UCLAMP((em[0].zero - em[0].one)*sb);
    	m[n+APSK16NC2] = UCLAMP((em[1].zero - em[1].one)*sb);
    	m[n+APSK16NC1] = UCLAMP((em[2].zero - em[2].one)*sb);
    	m[n]           = UCLAMP((em[3].zero - em[3].one)*sb);
    	break;
    case I_0123_S:
    	m[n+APSK16SC3] = UCLAMP((em[0].zero - em[0].one)*sb);
    	m[n+APSK16SC2] = UCLAMP((em[1].zero - em[1].one)*sb);
    	m[n+APSK16SC1] = UCLAMP((em[2].zero - em[2].one)*sb);
    	m[n]           = UCLAMP((em[3].zero - em[3].one)*sb);
    	break;
    }
}
__global__ void d_dvbs2_32apsk(FComplex *s, LLR *m, float sb, uint8_t inter  ){

	int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    Em em[5];
    float e;
    em[0].one = em[0].zero = 10;
    em[1].one = em[1].zero = 10;
    em[2].one = em[2].zero = 10;
    em[3].one = em[3].zero = 10;
    em[4].one = em[4].zero = 10;

    for( uint8_t i = 0; i < 32; i++){
    	e = CERROR(s[n],d_const[i]);
    	if(i&0b00001){
    		if(e < em[0].one) em[0].one = e;
    	}else{
    		if(e < em[0].zero) em[0].zero = e;
    	}
    	if(i&0b00010){
    		if(e < em[1].one) em[1].one = e;
    	}else{
    		if(e < em[1].zero) em[1].zero = e;
    	}
    	if(i&0b00100){
    		if(e < em[2].one) em[2].one = e;
    	}else{
    		if(e < em[2].zero) em[2].zero = e;
    	}
    	if(i&0b01000){
    		if(e < em[3].one) em[3].one = e;
    	}else{
    		if(e < em[3].zero) em[3].zero = e;
    	}
    	if(i&0b10000){
    		if(e < em[4].one) em[4].one = e;
    	}else{
    		if(e < em[4].zero) em[4].zero = e;
    	}
    }
    switch(inter){
    case I_01234_N:
    	m[n+APSK32NC4] = UCLAMP((em[0].zero - em[0].one)*sb);
    	m[n+APSK32NC3] = UCLAMP((em[1].zero - em[1].one)*sb);
    	m[n+APSK32NC2] = UCLAMP((em[2].zero - em[2].one)*sb);
    	m[n+APSK32NC1] = UCLAMP((em[3].zero - em[3].one)*sb);
    	m[n]           = UCLAMP((em[4].zero - em[4].one)*sb);
    	break;
    case I_01234_S:
    	m[n+APSK32SC4] = UCLAMP((em[0].zero - em[0].one)*sb);
    	m[n+APSK32SC3] = UCLAMP((em[1].zero - em[1].one)*sb);
    	m[n+APSK32SC2] = UCLAMP((em[2].zero - em[2].one)*sb);
    	m[n+APSK32SC1] = UCLAMP((em[3].zero - em[3].one)*sb);
    	m[n]           = UCLAMP((em[4].zero - em[4].one)*sb);
    	break;
    }
}
__global__ void d_dvbs2_64apsk(FComplex *s, LLR *m, float sb, uint8_t inter ){

	int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    Em em[6];
    float e;
    em[0].one = em[0].zero = 10;
    em[1].one = em[1].zero = 10;
    em[2].one = em[2].zero = 10;
    em[3].one = em[3].zero = 10;
    em[4].one = em[4].zero = 10;
    em[5].one = em[5].zero = 10;

    for( uint8_t i = 0; i < 64; i++){
    	e = CERROR(s[n],d_const[i]);
    	if(i&0b000001){
    		if(e < em[0].one) em[0].one = e;
    	}else{
    		if(e < em[0].zero) em[0].zero = e;
    	}
    	if(i&0b000010){
    		if(e < em[1].one) em[1].one = e;
    	}else{
    		if(e < em[1].zero) em[1].zero = e;
    	}
    	if(i&0b000100){
    		if(e < em[2].one) em[2].one = e;
    	}else{
    		if(e < em[2].zero) em[2].zero = e;
    	}
    	if(i&0b001000){
    		if(e < em[3].one) em[3].one = e;
    	}else{
    		if(e < em[3].zero) em[3].zero = e;
    	}
    	if(i&0b010000){
    		if(e < em[4].one) em[4].one = e;
    	}else{
    		if(e < em[4].zero) em[4].zero = e;
    	}
    	if(i&0b100000){
    		if(e < em[5].one) em[5].one = e;
    	}else{
    		if(e < em[5].zero) em[5].zero = e;
    	}
    }

	m[n+APSK64NC5] = UCLAMP((em[0].zero - em[0].one)*sb);
	m[n+APSK64NC4] = UCLAMP((em[1].zero - em[1].one)*sb);
	m[n+APSK64NC3] = UCLAMP((em[2].zero - em[2].one)*sb);
	m[n+APSK64NC2] = UCLAMP((em[3].zero - em[3].one)*sb);
	m[n+APSK64NC1] = UCLAMP((em[4].zero - em[4].one)*sb);
	m[n]           = UCLAMP((em[5].zero - em[5].one)*sb);
}
__global__ void d_dvbs2_128apsk(FComplex *s, LLR *m, float sb, uint8_t inter ){

	int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    Em em[7];
    float e;
    em[0].one = em[0].zero = 10;
    em[1].one = em[1].zero = 10;
    em[2].one = em[2].zero = 10;
    em[3].one = em[3].zero = 10;
    em[4].one = em[4].zero = 10;
    em[5].one = em[5].zero = 10;
    em[6].one = em[6].zero = 10;

    for( uint8_t i = 0; i < 128; i++){
    	e = CERROR(s[n],d_const[i]);
    	if(i&0b0000001){
    		if(e < em[0].one) em[0].one = e;
    	}else{
    		if(e < em[0].zero) em[0].zero = e;
    	}
    	if(i&0b0000010){
    		if(e < em[1].one) em[1].one = e;
    	}else{
    		if(e < em[1].zero) em[1].zero = e;
    	}
    	if(i&0b0000100){
    		if(e < em[2].one) em[2].one = e;
    	}else{
    		if(e < em[2].zero) em[2].zero = e;
    	}
    	if(i&0b0001000){
    		if(e < em[3].one) em[3].one = e;
    	}else{
    		if(e < em[3].zero) em[3].zero = e;
    	}
    	if(i&0b0010000){
    		if(e < em[4].one) em[4].one = e;
    	}else{
    		if(e < em[4].zero) em[4].zero = e;
    	}
    	if(i&0b0100000){
    		if(e < em[5].one) em[5].one = e;
    	}else{
    		if(e < em[5].zero) em[5].zero = e;
    	}
    	if(i&0b1000000){
    		if(e < em[6].one) em[6].one = e;
    	}else{
    		if(e < em[6].zero) em[6].zero = e;
    	}
    }

	m[n+APSK128NC6] = UCLAMP((em[0].zero - em[0].one)*sb);
	m[n+APSK128NC5] = UCLAMP((em[1].zero - em[1].one)*sb);
	m[n+APSK128NC4] = UCLAMP((em[2].zero - em[2].one)*sb);
	m[n+APSK128NC3] = UCLAMP((em[3].zero - em[3].one)*sb);
	m[n+APSK128NC2] = UCLAMP((em[4].zero - em[4].one)*sb);
	m[n+APSK128NC1] = UCLAMP((em[5].zero - em[5].one)*sb);
	m[n]            = UCLAMP((em[6].zero - em[6].one)*sb);
}
__global__ void d_dvbs2_256apsk(FComplex *s, LLR *m, float sb, uint8_t inter ){

	int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    Em em[8];
    float e;
    em[0].one = em[0].zero = 10;
    em[1].one = em[1].zero = 10;
    em[2].one = em[2].zero = 10;
    em[3].one = em[3].zero = 10;
    em[4].one = em[4].zero = 10;
    em[5].one = em[5].zero = 10;
    em[6].one = em[6].zero = 10;
    em[7].one = em[7].zero = 10;

    for( uint8_t i = 0; i < 256; i++){
    	e = CERROR(s[n],d_const[i]);
    	if(i&0b00000001){
    		if(e < em[0].one) em[0].one = e;
    	}else{
    		if(e < em[0].zero) em[0].zero = e;
    	}
    	if(i&0b00000010){
    		if(e < em[1].one) em[1].one = e;
    	}else{
    		if(e < em[1].zero) em[1].zero = e;
    	}
    	if(i&0b00000100){
    		if(e < em[2].one) em[2].one = e;
    	}else{
    		if(e < em[2].zero) em[2].zero = e;
    	}
    	if(i&0b00001000){
    		if(e < em[3].one) em[3].one = e;
    	}else{
    		if(e < em[3].zero) em[3].zero = e;
    	}
    	if(i&0b00010000){
    		if(e < em[4].one) em[4].one = e;
    	}else{
    		if(e < em[4].zero) em[4].zero = e;
    	}
    	if(i&0b00100000){
    		if(e < em[5].one) em[5].one = e;
    	}else{
    		if(e < em[5].zero) em[5].zero = e;
    	}
    	if(i&0b01000000){
    		if(e < em[6].one) em[6].one = e;
    	}else{
    		if(e < em[6].zero) em[6].zero = e;
    	}
    	if(i&0b10000000){
    		if(e < em[7].one) em[7].one = e;
    	}else{
    		if(e < em[7].zero) em[7].zero = e;
    	}
    }

	m[n+APSK256NC6] = UCLAMP((em[0].zero - em[0].one)*sb);
	m[n+APSK256NC6] = UCLAMP((em[1].zero - em[1].one)*sb);
	m[n+APSK256NC5] = UCLAMP((em[2].zero - em[2].one)*sb);
	m[n+APSK256NC4] = UCLAMP((em[3].zero - em[3].one)*sb);
	m[n+APSK256NC3] = UCLAMP((em[4].zero - em[4].one)*sb);
	m[n+APSK256NC2] = UCLAMP((em[5].zero - em[5].one)*sb);
	m[n+APSK256NC1] = UCLAMP((em[6].zero - em[6].one)*sb);
	m[n]            = UCLAMP((em[7].zero - em[7].one)*sb);
}
//
// va == noise variance
//
void dvbs2_qpsk(FComplex *ds,LLR *dm, float va, uint8_t inter, size_t len){
	float sa = 1.0/sqrtf(2.0*M_PI*va);
    float sb = 1.0/(2.0*va);
	d_dvbs2_qpsk<<<len,NP_FRAMES>>>( ds, dm, sa, sb );
}
//
// 8 PSK variants
//
void dvbs2_8psk(FComplex *ds, LLR *dm, float sn, uint8_t inter, size_t len ){
	float sa = 1.0/sqrtf(2.0f*M_PI*sn);
    float sb = 1.0/(2.0f*sn);

	d_dvbs2_8psk<<<len,NP_FRAMES>>>( ds, dm, sa, sb, inter );

}
void dvbs2_8apsk(FComplex *ds, LLR *dm, float sn, uint8_t inter, size_t len ){
	float sa = 1.0/sqrtf(2.0f*M_PI*sn);
    float sb = 1.0/(2.0f*sn);

	d_dvbs2_8apsk<<<len,NP_FRAMES>>>( ds, dm, sb, inter );

}

void dvbs2_16apsk(FComplex *ds, LLR *dm, float sn, uint8_t inter, size_t len){
	float sa = 1.0/sqrtf(2.0f*M_PI*sn);
    float sb = 1.0/(2.0f*sn);

	d_dvbs2_16apsk<<<len,NP_FRAMES>>>( ds, dm, sb, inter );

}
void dvbs2_32apsk(FComplex *ds, LLR *dm, float sn, uint8_t inter, size_t len){
	float sa = 1.0/sqrtf(2.0f*M_PI*sn);
    float sb = 1.0/(2.0f*sn);

	d_dvbs2_32apsk<<<len,NP_FRAMES>>>( ds, dm, sb, inter );

}
void dvbs2_64apsk(FComplex *ds, LLR *dm, float sn, uint8_t inter, size_t len){
	float sa = 1.0/sqrtf(2.0f*M_PI*sn);
    float sb = 1.0/(2.0f*sn);

	d_dvbs2_64apsk<<<len,NP_FRAMES>>>( ds, dm, sb, inter );

}
void dvbs2_128apsk(FComplex *ds, LLR *dm, float sn, uint8_t inter, size_t len){
	float sa = 1.0/sqrtf(2.0f*M_PI*sn);
    float sb = 1.0/(2.0f*sn);

	d_dvbs2_128apsk<<<len,NP_FRAMES>>>( ds, dm, sb, inter  );

}
void dvbs2_256apsk(FComplex *ds, LLR *dm, float sn, uint8_t inter, size_t len){
	float sa = 1.0/sqrtf(2.0f*M_PI*sn);
    float sb = 1.0/(2.0f*sn);

	d_dvbs2_256apsk<<<len,NP_FRAMES>>>( ds, dm, sb, inter  );

}

void demap_update_constellation(FComplex *c, int len){
	CHECK(cudaMemcpyToSymbol(d_const, c, sizeof(FComplex)*len));
}
void demapin_set(void){

	switch(g_format.mod_class){
	case m_QPSK:
		demapin = &dvbs2_qpsk;
		break;
	case m_8PSK:
		demapin = &dvbs2_8psk;
		break;
	case m_8APSK:
		demapin = &dvbs2_8apsk;
		break;
	case m_16APSK:
		demapin = &dvbs2_16apsk;
		break;
	case m_32APSK:
		demapin = &dvbs2_32apsk;
		break;
	case m_64APSK:
		demapin = &dvbs2_64apsk;
		break;
	case m_128APSK:
		demapin = &dvbs2_128apsk;
		break;
	case m_256APSK:
		demapin = &dvbs2_256apsk;
		break;
	default:
		printf("Error Unknown demapping format\n");
		demapin = &dvbs2_qpsk;// default
		break;
	}
}
