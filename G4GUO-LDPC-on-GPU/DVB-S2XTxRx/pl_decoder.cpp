#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "dvbs2_rx.h"

//extern RxFormat g_format;
FComplex g_preamble[90];

#define MSIZE 32
//
// Tables used to generate and decode header
//
const float ph_scram_ftab[64] =
{
	-1.0,  1.0, 1.0,  1.0, -1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0, 
	 1.0,  1.0, 1.0, -1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
	 1.0,  1.0, 1.0,  1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0, 
	 1.0, -1.0, 1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  1.0, -1.0, -1.0, 
	-1.0, -1.0, 1.0, -1.0, -1.0, -1.0,  1.0, -1.0,  1.0,  1.0, -1.0, 
	 1.0,  1.0, 1.0,  1.0,  1.0,  1.0, -1.0,  1.0, -1.0
};
const unsigned long g[7] =
{
	0x90AC2DDD, 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF, 0xFFFFFFFF
};

const int ph_scram_tab[64] =
{
	0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1,
	0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0
};
const int ph_sync_seq[26] =
{
	0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0
};

float m_h[MSIZE][MSIZE];
FComplex m_bpsk0[2][2];
FComplex m_bpsk1[2][2];

uint8_t m_last_modcod;

void b_64_8_code(unsigned char in, int *out)
{
	unsigned long temp, bit;

	temp = 0;

	if (in & 0x80) temp ^= g[0];
	if (in & 0x40) temp ^= g[1];
	if (in & 0x20) temp ^= g[2];
	if (in & 0x10) temp ^= g[3];
	if (in & 0x08) temp ^= g[4];
	if (in & 0x04) temp ^= g[5];
	if (in & 0x02) temp ^= g[6];

	bit = 0x80000000;
	for (int m = 0; m < 32; m++)
	{
		out[(m * 2)] = (temp&bit) ? 1 : 0;
		out[(m * 2) + 1] = out[m * 2] ^ (in & 0x01);
		bit >>= 1;
	}
	// Randomise it
	for (int m = 0; m < 64; m++)
	{
		out[m] = out[m] ^ ph_scram_tab[m];
	}
}
void b_64_7_code( unsigned char in, int *out )
{
    unsigned long temp,bit;

    temp = 0;

    if(in&0x40) temp ^= g[1];
    if(in&0x20) temp ^= g[2];
    if(in&0x10) temp ^= g[3];
    if(in&0x08) temp ^= g[4];
    if(in&0x04) temp ^= g[5];
    if(in&0x02) temp ^= g[6];

    bit = 0x80000000;
    for( int m = 0; m < 32; m++ )
    {
        out[(m*2)]   = (temp&bit)?1:0;
        out[(m*2)+1] = out[m*2]^(in&0x01);
        bit >>= 1;
    }
    // Randomise it
    for( int m = 0; m < 64; m++ )
    {
        out[m] = out[m] ^ ph_scram_tab[m];
    }
}

//
// Encode a DVB-S2/X preamble
//
void pl_encode_preamble(int code, FComplex *pream){
	int b[90];

	// Add the sync sequence SOF
	for (int i = 0; i < 26; i++) b[i] = ph_sync_seq[i];

	// Add the mode and code
	b_64_8_code(code, &b[26]);

	// BPSK modulate and add the header

	for (int i = 0; i < 26; i++)
	{
		pream[i] = m_bpsk0[i & 1][b[i]];
	}
	if(code&0x80){
		// S2X
		for (int i = 26; i < 90; i++)
		{
			pream[i] = m_bpsk1[i & 1][b[i]];
		}
	}else{
		// S2
		for (int i = 26; i < 90; i++)
		{
			pream[i] = m_bpsk0[i & 1][b[i]];
		}
	}
}
void pl_differential_preamble(void){
	float di[57];
	float dr[57];
	int idx = 0;
	for( int i = 0; i < 25; i++){
		di[idx] = cmultImagConj(g_preamble[i],g_preamble[i+1]);
//		dr[idx] = cmultRealConj(g_preamble[i],g_preamble[i+1]);
		idx++;
	}

	for( int i = 26; i < 89; i+=2){
		di[idx] = cmultImagConj(g_preamble[i],g_preamble[i+1]);
//		dr[idx] = cmultRealConj(g_preamble[i],g_preamble[i+1]);
		idx++;
	}
	preamble_coeffs(di);
/*
	for( int i = 26; i < 57; i++ ){
		printf("%d \t%f \t%f\n",i, dr[i],di[i]);
	}
*/
}

//
// Builds a Hadamard Matrix to be used in the decode
//
void build_hmatrix(void){
	int m = 1;
	int M = 32;
	m_h[0][0] = 1.0;

	int i, j;

	while ( m < M) {
		for (i = 0; i < m; i++) {
			for (j = 0; j < m; j++) {
				m_h[i + m][j]     =  m_h[i][j];
				m_h[i][j + m]     =  m_h[i][j];
				m_h[i + m][j + m] = -m_h[i][j];
			}
		}
		m *= 2;
	}
}
void build_bpsk_table(void){

	float sym = sqrt(0.5);
	//S2
	m_bpsk0[0][0].re =  sym;
	m_bpsk0[0][0].im =  sym;
	m_bpsk0[0][1].re = -sym;
	m_bpsk0[0][1].im = -sym;
	m_bpsk0[1][0].re = -sym;
	m_bpsk0[1][0].im =  sym;
	m_bpsk0[1][1].re =  sym;
	m_bpsk0[1][1].im = -sym;

    // S2X
	m_bpsk1[0][0].re = -sym;
	m_bpsk1[0][0].im =  sym;
	m_bpsk1[0][1].re =  sym;
	m_bpsk1[0][1].im = -sym;
	m_bpsk1[1][0].re = -sym;
	m_bpsk1[1][0].im = -sym;
	m_bpsk1[1][1].re =  sym;
	m_bpsk1[1][1].im =  sym;

}
void do_hadamard(float *in, float *out){
	for (int i = 0; i < 32; i++){
		out[i] = 0;
		for (int j = 0; j < 32; j++){
			out[i] += in[j] * m_h[i][j];
		}
	}
}

int pl_s2_decode( float *m ){

	float res[32];
	
	do_hadamard( m, res);

	float a;
	float max  = 0;
	int   code = 0;
	int   pos  = 0;

	// Find the maximum
	for (int i = 0; i < 32; i++){
		a = fabs(res[i]);
		if (a > max){
			max = a;
			pos  = i;
		}
	}

	if (res[pos] > 0) code |= 0x02;
	if (pos & 0x10) code   |= 0x04;
	if (pos & 0x08) code   |= 0x08;
	if (pos & 0x04) code   |= 0x10;
	if (pos & 0x02) code   |= 0x20;
	if (pos & 0x01) code   |= 0x40;

	return code;
}
int pl_s2x_decode( float *m ){
	float res[32];
	float max = 0;
	float a;
	int   code = 0;
	int   pos = 0;
	//
	// remove the S2X msb encoded word and test again
	//
	uint32_t bit = 0x80000000;
	for( int i = 0; i < 32; i++){
		if((bit & g[0])) m[i] *= -1;
		bit >>= 1;
	}

	do_hadamard(m, res);

	// Find the maximum
	for (int i = 0; i < 32; i++){
		a = fabs(res[i]);
		if (a > max){
			max = a;
			pos = i;
		}
	}

	code = 0x80;
	if (res[pos] > 0) code |= 0x02;
	if (pos & 0x10)   code |= 0x04;
	if (pos & 0x08)   code |= 0x08;
	if (pos & 0x04)   code |= 0x10;
	if (pos & 0x02)   code |= 0x20;
	if (pos & 0x01)   code |= 0x40;

	return code;
}
//
// Pass pointer to start of MODCOD field
//
// The S2X PLS code is encoded differently so this needs looking at
// as it was only designed for S2 but seems to work for S2X
//

uint8_t pl_decode(FComplex *in){
	float p[2][64];
	uint8_t code;

	// Remove the offset modulation

	for (int i = 0; i < 64; i++){
		p[0][i] = cmultRealConj(in[i], m_bpsk0[i&1][1]);
		p[1][i] = cmultRealConj(in[i], m_bpsk1[i&1][1]);
	}
	// Descramble
	for (int i = 1; i < 64; i++){
		float v = ph_scram_ftab[i] == 1 ? -1.0 : 1.0;
		p[0][i] = p[0][i] * v;
		p[1][i] = p[1][i] * v;
	}
	// Work out lsb
	float sum0 = 0;
	float sum1 = 0;

	for (int i = 0; i < 64 - 1; i+=2){
		sum0 += p[0][i] * p[0][i+1];
		sum1 += p[1][i] * p[1][i+1];
	}
//	printf("sum %f %f\n",sum0,sum1);

	if(1){
//	if(fabs(sum0) > fabs(sum1)){
		//printf("DVB-S2 %f %f \n",sum0,sum1);
		if (sum0 < 0)
			code = 1;
		else
			code = 0;
		// Use this bit to remove differential encoding and accumulate bits
		int n = 0;
		if (code){
			for (int i = 0; i < 64; i += 2){
				p[0][n++] = p[0][i] - p[0][i + 1];
			}
		}
		else{
			for (int i = 0; i < 64; i += 2){
				p[0][n++] = p[0][i] + p[0][i + 1];
			}
		}
		code = pl_s2_decode(p[0]) | code;
	}else{
		//printf("DVB-S2X %f %f\n",sum0,sum1);
		if (sum1 < 0)
			code = 1;
		else
			code = 0;
		// Use this bit to remove differential encoding and accumulate bits
		int n = 0;
		if (code){
			for (int i = 0; i < 64; i += 2){
				p[1][n++] = p[1][i] - p[1][i + 1];
			}
		}
		else{
			for (int i = 0; i < 64; i += 2){
				p[1][n++] = p[1][i] + p[1][i + 1];
			}
		}
		code = pl_s2x_decode(p[1]) | code;
	}
	return code;
}
ModcodStatus pl_new_modcod(uint8_t modcod ){
	ModcodStatus status;
	if(modcod != m_last_modcod){
	    // Tell the receiver what to expect
	    status = modcod_decode(modcod);
	    if (status == MODCOD_OK){
	    	printf("MODCOD changed %d\n",modcod);
		    // Update the constellation tables
		    contab_set();
		    // Set the demmapping and de-interleaving
		    demapin_set();
		    // Set the type of BCH decoder
		    bch_select_device_lookup_table();
		    // Generate a new reference preamble
		    pl_encode_preamble(modcod, g_preamble);
		    ldpc2_decode_set_fec();
		    // Set the constellation to use in the DFE
		    eq_set_modulation();
		    // Set The BCH code to use
		    bch_set_decoder();
		    // Save the new code format
		    m_last_modcod = modcod;
	    }
	}
	return status;
}
void pl_decode_open(void){
	build_hmatrix();
	build_bpsk_table();
	// Build a dummy preamble
	pl_encode_preamble(16, g_preamble);
	pl_differential_preamble();
	m_last_modcod = 255;
}
void pl_decode_close(void){
}
