#include "dvbs2_rx.h"

static int m_cscram[FRAME_SIZE_NORMAL];
static uint8_t m_bb_randomise[FRAME_SIZE_NORMAL/8];

void init_bb_randomiser(void)
{
    int sr = 0x4A80;
    for( int i = 0; i < FRAME_SIZE_NORMAL/8; i++ )
    {
		m_bb_randomise[i] = 0;

		for(int j = 0; j < 8; j++){
    		int b = ((sr)^(sr>>1))&1;
    		m_bb_randomise[i] <<= 1;
    		m_bb_randomise[i] |= b;
    		sr >>= 1;
    		if( b ) sr |= 0x4000;
    	}
    }
}
//
// Randomise the data bits
//
void bb_derandomise(uint8_t *frame, int len)
{
    for( int i = 0; i < len; i++ )
   {
        frame[i] ^= m_bb_randomise[i];
    }
}

int parity_chk(long a, long b)
{
	int c = 0;
	a = a & b;
	for (int i = 0; i < 18; i++)
	{
		if (a&(1L << i)) c++;
	}
	return c & 1;
}
//
// This is not time sensitive and is only run at start up
//
void build_symbol_scrambler_table(void)
{
	long x, y;
	int xa, xb, xc, ya, yb, yc;
	int rn, zna, znb;

	// Initialisation
	x = 0x00001;
	y = 0x3FFFF;

	for (int i = 0; i < 64800; i++)
	{
		xa = parity_chk(x, 0x8050);
		xb = parity_chk(x, 0x0081);
		xc = x & 1;

		x >>= 1;
		if (xb) x |= 0x20000;

		ya = parity_chk(y, 0x04A1);
		yb = parity_chk(y, 0xFF60);
		yc = y & 1;

		y >>= 1;
		if (ya) y |= 0x20000;

		zna = xc ^ yc;
		znb = xa ^ yb;
		rn = (znb << 1) + zna;
		m_cscram[i] = rn;
	}
}
int m_scram_index;

FComplex scramble_symbol(FComplex x, int n)
{
	FComplex y;
	// Start at the end of the PL Header.

	switch (m_cscram[n])
	{
	case 0:
		// Do nothing
		y = x;
		break;
	case 1:
		y.re = -x.im;
		y.im =  x.re;
		break;
	case 2:
		y.re = -x.re;
		y.im = -x.im;
		break;
	case 03:
		y.re =  x.im;
		y.im = -x.re;
		break;
	}
	return y;
}
FComplex descramble_symbol(FComplex y, int n)
{
	FComplex x;
	// Start at the end of the PL Header.

	switch (m_cscram[n])
	{
	case 0:
		// Do nothing
		x = y;
		break;
	case 1:
		x.im = -y.re;
		x.re =  y.im;
		break;
	case 2:
		x.re = -y.re;
		x.im = -y.im;
		break;
	case 03:
		x.re = -y.im;
		x.im =  y.re;
		break;
	}
	return x;
}

void descrambler_open(void){
	build_symbol_scrambler_table();
	init_bb_randomiser();
}
void descrambler_close(void){
}
