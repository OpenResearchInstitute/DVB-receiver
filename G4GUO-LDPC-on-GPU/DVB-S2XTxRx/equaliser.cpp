//
// RxEqualizer.
//
#include <stdio.h>
#include <math.h>
#include "dvbs2_rx.h"

extern RxFormat g_format;
extern FComplex g_preamble[90];
/*
*
* Diagnostic routine
*
*
*/


// External tables
extern FComplex g_qpsk[4];
extern FComplex g_8psk[8];
extern FComplex g_16apsk[16];
extern FComplex g_32apsk[32];
extern FComplex g_64apsk[64];
extern FComplex g_128apsk[128];
extern FComplex g_256apsk[256];

typedef struct{
    FComplex c[KN];
    FComplex g[KN];
    FComplex u[KN][KN];
    FComplex fbr;
    FComplex de_rotate;
    float d[KN];
    float q;
    float E;
    float y;

    double error_sum;
    double symbol_sum;
    uint32_t symbol_cnt;
    FComplex error;
    FComplex sum;
    uint32_t error_cnt;
}SEqual;

SEqual m_e;

void eq_k_reset_coffs(void)
{
	int i;

	for (i = 0; i < KN; i++)
	{
		m_e.c[i].re = 0.0;
		m_e.c[i].im = 0.0;
	}
}
void eq_k_reset_ud(void)
{
	int i, j;

	for (j = 0; j < KN; j++)
	{
		for (i = 0; i < j; i++)

		{
			m_e.u[i][j].re = 0.0;
			m_e.u[i][j].im = 0.0;
		}
		m_e.d[j] = 0.1f;
	}
}
/*
*
*
* Modified Root Kalman gain Vector estimator
*
*
*/
void eq_k_calculate(FComplex *x)
{
	int     i, j;
	FComplex  B0;
	float   hq;
	float   B;
	float   ht;
	FComplex  f[KN];
	FComplex  h[KN];
	float     a[KN];

	f[0].re =  x[0].re;               // 6.2
	f[0].im = -x[0].im;

	for (j = 1; j < KN; j++)              // 6.3
	{
		f[j].re = cmultRealConj(m_e.u[0][j], x[0]) + x[j].re;
		f[j].im = cmultImagConj(m_e.u[0][j], x[0]) - x[j].im;
		for (i = 1; i < j; i++)
		{
			f[j].re += cmultRealConj(m_e.u[i][j], x[i]);
			f[j].im += cmultImagConj(m_e.u[i][j], x[i]);
		}
	}

	for (j = 0; j < KN; j++)                // 6.4
	{
		m_e.g[j].re = m_e.d[j] * f[j].re;
		m_e.g[j].im = m_e.d[j] * f[j].im;
	}

	a[0] = m_e.E + cmultRealConj(m_e.g[0], f[0]); 	 // 6.5

	for (j = 1; j < KN; j++) // 6.6
	{
		a[j] = a[j - 1] + cmultRealConj(m_e.g[j], f[j]);
	}
	hq = 1 + m_e.q;                              // 6.7
	ht = a[KN - 1] * m_e.q;

	m_e.y = (float)1.0 / (a[0] + ht);                       // 6.19

	m_e.d[0] = m_e.d[0] * hq * (m_e.E + ht) * m_e.y;       // 6.20

	// 6.10 - 6.16 (Calculate recursively)

	for (j = 1; j < KN; j++)
	{
		B = a[j - 1] + ht;                 // 6.21

		h[j].re = -f[j].re*m_e.y;        // 6.11
		h[j].im = -f[j].im*m_e.y;

		m_e.y = (float)1.0 / (a[j] + ht);               // 6.22

		m_e.d[j] = m_e.d[j] * hq*B*m_e.y;              // 6.13

		for (i = 0; i < j; i++)
		{
			B0 = m_e.u[i][j];
			m_e.u[i][j].re = B0.re + cmultRealConj(h[j], m_e.g[i]); // 6.15
			m_e.u[i][j].im = B0.im + cmultImagConj(h[j], m_e.g[i]);

			m_e.g[i].re += cmultRealConj(m_e.g[j], B0);               // 6.16
			m_e.g[i].im += cmultImagConj(m_e.g[j], B0);
		}
	}
}
/*
*
* Update the filter coefficients using the Kalman gain vector and
* the error
*
*/
void eq_k_update(FComplex *s, FComplex error, FComplex symbol )
{
	int i;
	//
	// Calculate the new Kalman gain vector
	//
	eq_k_calculate(s);
	//
	// Update the variance and sum error
	m_e.error_sum  += (error.re*error.re)   + (error.im*error.im);
	m_e.symbol_sum += (symbol.re*symbol.re) + (symbol.im*symbol.im);
	m_e.symbol_cnt++;
	m_e.error.re += error.re;
	m_e.error.im += error.im;
	m_e.error_cnt++;

	//
	// Update the filter coefficients using the gain vector
	// and the error.
	//
	error.re *= m_e.y;
	error.im *= m_e.y;

	for (i = 0; i < KN; i++)
	{
		m_e.c[i].re += cmultReal(error, m_e.g[i]);
		m_e.c[i].im += cmultImag(error, m_e.g[i]);
	}
}
//
// Update the coefficients using the LMS algorithm
//
void eq_lms_update(FComplex *s, FComplex error, FComplex symbol)
{
	static int count;

	count++;

	m_e.sum.re += s->re;
	m_e.sum.im += s->im;

	if((count&0x0) == 0 )
	{
	    eq_k_update(s, error, symbol);
	}else{
		//
		// Update the filter coefficients, the samples and the error.
		//
		error.re *=  0.001; // ue*
		error.im *= -0.001;

		for (int i = 0; i < KN; i++)
		{
			m_e.c[i].re += cmultReal(error,s[i]);
			m_e.c[i].im += cmultImag(error,s[i]);
		}
		// Update the variance and sum error
		m_e.error.re += error.re;
		m_e.error.im += error.im;
		m_e.error_sum  += (error.re*error.re) + (error.im*error.im);
		m_e.symbol_sum += (symbol.re*symbol.re) + (symbol.im*symbol.im);
		m_e.symbol_cnt++;
		m_e.error_cnt++;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
// Equalizer routines
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////////////////

FComplex eq_equalize(FComplex *s)
{
	FComplex symbol;

	/* Calculate the symbol */

	symbol.re = cmultReal(s[0],m_e.c[0]);
	symbol.im = cmultImag(s[0],m_e.c[0]);


	for (int i = 1; i < KN; i++)
	{
		symbol.re += cmultReal(s[i],m_e.c[i]);
		symbol.im += cmultImag(s[i],m_e.c[i]);
	}

	return symbol;
}
void eq_equalize_reset(void)
{
	eq_reset();
}
void eq_equalize_restart(void)
{
	eq_k_reset_ud();
}
/*
*
* Train the equalizer using known symbols
*
*/
FComplex eq_equalize_train_known(FComplex *in, FComplex train)
{
	FComplex error;
	FComplex symbol;

	// Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

	symbol = eq_equalize(s);

	/* Calculate error */

	error.re = train.re - symbol.re;
	error.im = train.im - symbol.im;

	/* Update the coefficients */
	eq_k_update( s, error, symbol );

	/* Update the FB data */

	m_e.fbr = train;

	return(symbol);
}
/*
*
* Equalise the data and train on the hard decision
*
*/
FComplex eq_qpsk_preamble(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

    // Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

    // Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_qpsk[0]);

    for( int i = 1; i < 4; i++){
        mmin = CERROR(symbol,g_qpsk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_qpsk[imin];
    m_e.fbr = decision;
	/* Update the coefficients */
	eq_k_update( s, error, symbol );

	return symbol;
}

FComplex eq_data_qpsk(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

    // Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

    // Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_qpsk[0]);

    for( int i = 1; i < 4; i++){
        mmin = CERROR(symbol,g_qpsk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_qpsk[imin];
	// Update the FB data
    m_e.fbr = decision;
	// Update the coefficients 
	error.re = decision.re - symbol.re;
	error.im = decision.im - symbol.im;

	eq_lms_update( s, error, symbol);

	FComplex sym;
	sym.re = cmultRealConj(symbol,m_e.de_rotate);
	sym.im = cmultImagConj(symbol,m_e.de_rotate);

	return sym;
}
FComplex eq_data_8psk(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

    // Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

    // Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_8psk[0]);

    for( int i = 1; i < 8; i++){
        mmin = CERROR(symbol,g_8psk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_8psk[imin];
	// Update the FB data
    m_e.fbr = decision;
	// Update the coefficients
	error.re = decision.re - symbol.re;
	error.im = decision.im - symbol.im;

	eq_lms_update( s, error, symbol);

	FComplex sym;
	sym.re = cmultRealConj(symbol,m_e.de_rotate);
	sym.im = cmultImagConj(symbol,m_e.de_rotate);

	return sym;
}
FComplex eq_data_16apsk(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

    // Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

    // Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_16apsk[0]);

    for( int i = 1; i < 16; i++){
        mmin = CERROR(symbol,g_16apsk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_16apsk[imin];
	// Update the FB data
    m_e.fbr = decision;
	// Update the coefficients
	error.re = decision.re - symbol.re;
	error.im = decision.im - symbol.im;

	eq_lms_update( s, error, symbol);

	FComplex sym;
	sym.re = cmultRealConj(symbol,m_e.de_rotate);
	sym.im = cmultImagConj(symbol,m_e.de_rotate);

	return sym;
}
FComplex eq_data_32apsk(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

    // Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

    // Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_32apsk[0]);

    for( int i = 1; i < 32; i++){
        mmin = CERROR(symbol,g_32apsk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_32apsk[imin];
	// Update the FB data
    m_e.fbr = decision;
	// Update the coefficients
	error.re = decision.re - symbol.re;
	error.im = decision.im - symbol.im;

	eq_lms_update( s, error, symbol);

	FComplex sym;
	sym.re = cmultRealConj(symbol,m_e.de_rotate);
	sym.im = cmultImagConj(symbol,m_e.de_rotate);

	return sym;
}
FComplex eq_data_64apsk(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

	// Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

	// Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_64apsk[0]);

    for( int i = 1; i < 64; i++){
        mmin = CERROR(symbol,g_64apsk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_64apsk[imin];
	// Update the FB data
    m_e.fbr = decision;
	// Update the coefficients
	error.re = decision.re - symbol.re;
	error.im = decision.im - symbol.im;

	eq_lms_update( s, error, symbol);

	FComplex sym;
	sym.re = cmultRealConj(symbol,m_e.de_rotate);
	sym.im = cmultImagConj(symbol,m_e.de_rotate);

	return sym;
}
FComplex eq_data_128apsk(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

	// Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

    // Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_128apsk[0]);

    for( int i = 1; i < 128; i++){
        mmin = CERROR(symbol,g_128apsk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_128apsk[imin];
	// Update the FB data
    m_e.fbr = decision;
	// Update the coefficients
	error.re = decision.re - symbol.re;
	error.im = decision.im - symbol.im;

	eq_lms_update( s, error, symbol);

	FComplex sym;
	sym.re = cmultRealConj(symbol,m_e.de_rotate);
	sym.im = cmultImagConj(symbol,m_e.de_rotate);

	return sym;
}
FComplex eq_data_256apsk(FComplex *in)
{
	FComplex    symbol;
	FComplex    error;
	FComplex    decision;
    float fmin;
    float mmin;
    int   imin;

    // Align so symbols is 1/2 way through filter
	FComplex *s =   &in[-KN/2];

    // Equalise using the current coefficients
    symbol = eq_equalize(s);

	// Find the constellation point with the minimum error
    // and use it as hard decision.

    imin = 0;
    fmin = CERROR(symbol,g_256apsk[0]);

    for( int i = 1; i < 256; i++){
        mmin = CERROR(symbol,g_256apsk[i]);
        if(mmin < fmin){
        	imin = i;
        	fmin = mmin;
        }
    }
    decision = g_256apsk[imin];
	// Update the FB data
    m_e.fbr = decision;
	// Update the coefficients
	error.re = decision.re - symbol.re;
	error.im = decision.im - symbol.im;

	eq_lms_update( s, error, symbol);

	FComplex sym;
	sym.re = cmultRealConj(symbol,m_e.de_rotate);
	sym.im = cmultImagConj(symbol,m_e.de_rotate);

	return sym;
}

/*
*
*
* Reset Kalman variables, to ensure stability
*
*
*/
void eq_reset(void)
{
	eq_k_reset_ud();
	eq_k_reset_coffs();
	m_e.error_sum = 0;
	m_e.symbol_sum = 0;
	m_e.symbol_cnt = 0;
}

/*
*
*
* Initialise this module
*
*
*/
void eq_open(void)
{
	m_e.q = (float)0.08;
	m_e.E = (float)0.01;
	eq_data = eq_data_qpsk;

	eq_k_reset_ud();
	eq_k_reset_coffs();
}
/*
 *
 * Set the constellation
 *
 */
FComplex (*eq_data)(FComplex *in);

void eq_set_modulation(void){

	switch(g_format.mod_class){
	case m_QPSK:
		eq_data = eq_data_qpsk;
		break;
	case m_8PSK:
		eq_data = eq_data_8psk;
		break;
	case m_16APSK:
		eq_data = eq_data_16apsk;
		break;
	case m_32APSK:
		eq_data = eq_data_32apsk;
		break;
	case m_64APSK:
		eq_data = eq_data_64apsk;
		break;
	case m_128APSK:
		eq_data = eq_data_128apsk;
		break;
	case m_256APSK:
		eq_data = eq_data_256apsk;
		break;
	default:
		printf("Error Unknown Equaliser demodulation format\n");
		eq_data = eq_data_qpsk;
		break;
	}
}

void eq_stats( float &variance, float &mer){
	if(m_e.symbol_cnt > 0 ){
	    variance = (float)(m_e.error_sum/m_e.symbol_cnt);
	    mer      = (float)10.0*log10(m_e.symbol_sum/m_e.error_sum);
//	    printf("Sum %f %f \n",m_sum.re,m_sum.im);
	    m_e.sum.re = m_e.sum.im = 0;
	    m_e.error_sum  = 0;
	    m_e.symbol_sum = 0;
	    m_e.symbol_cnt = 0;
	}else{
		variance = 0;
		mer      = 0;
	}
}
//
// Do a course equalisation, used during preamble hunt
// Check alignment of samples
//
void eq_course_preamble( FComplex *in, FComplex *out){
	int cn = 0;
	int i  = 0;

	// We know nothing
	eq_reset();

	for( i = 0; i < 52; i += 2){
	    eq_equalize_train_known(&in[i], g_preamble[cn++]);
	}
	// The PLSCODE uses QPSK
	cn = 0;
	for( ; i < 180; i += 2 ){
		out[cn++] = eq_qpsk_preamble(&in[i]);
	}
}
//
// Calculate the Magnitude and phase error after
// the equalisation of known symbols and apply the correction
//
void eq_de_rotate_estimate(FComplex *s, FComplex *r, int nr ){
	FComplex sum;

	sum.re = cmultRealConj(s[0],r[0]);
	sum.im = cmultImagConj(s[0],r[0]);

	for( int i = 1; i < nr; i++){
		sum.re += cmultRealConj(s[i],r[i]);
		sum.im += cmultImagConj(s[i],r[i]);
	}
    sum.re = sum.re/nr;
    sum.im = sum.im/nr;

	float mag = 1.0/sqrt(sum.re*sum.re + sum.im*sum.im);
//	mag = 1.0/sum.re;
//printf("Mag %f\n",1.0/mag);
	m_e.de_rotate.re = sum.re * mag;
	m_e.de_rotate.im = sum.im * mag;
//	m_e.de_rotate.re = 1.0;
//	m_e.de_rotate.im = 0;
}
void eq_de_rotate_apply_inplace(FComplex *s, int nr ){
	FComplex sum;
	for( int i = 0; i < nr; i++){
		sum.re = cmultRealConj(s[i],m_e.de_rotate);
		sum.im = cmultImagConj(s[i],m_e.de_rotate);
	    s[i] = sum;
	}
}
