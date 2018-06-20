#include <stdint.h>
#include <memory.h>
#include <stdio.h>
#include "dvbs2_rx.h"

/////////////////////////////////////////////////////////////////////////////
//
// Lookup routines for accelerated BCH checking
//
/////////////////////////////////////////////////////////////////////////////

extern RxFormat g_format;

static BCHLookup m_bch_n8_lookup[256];
static BCHLookup m_bch_n10_lookup[256];
static BCHLookup m_bch_n12_lookup[256];
static BCHLookup m_bch_s12_lookup[256];
static BCHLookup m_bch_m12_lookup[256];

static uint32_t m_32_poly_n_8[4];
static uint32_t m_32_poly_n_10[5];
static uint32_t m_32_poly_n_12[6];
static uint32_t m_32_poly_m_12[6];
static uint32_t m_32_poly_s_12[6];

void cb_sr_32_to_64( uint32_t *in, BCHLookup *out, int n)
{
	int idx = 0;
	for( int i = 0; i < n; i++){
		out->bch_r[i] = in[idx++];
		out->bch_r[i] <<= 32;
		out->bch_r[i] |= in[idx++];
	}
}
//
// Polynomial calculation routines
//
// multiply polynomials
//
int poly_mult( const int *ina, int lena, const int *inb, int lenb, int *out )
{
    memset( out, 0, sizeof(int)*(lena+lenb));

    for( int i = 0; i < lena; i++ )
    {
        for( int j = 0; j < lenb; j++ )
        {
            if( ina[i]*inb[j] > 0 ) out[i+j]++;// count number of terms for this pwr of x
        }
    }
    int max=0;
    for( int i = 0; i < lena+lenb; i++ )
    {
        out[i] = out[i]&1;// If even ignore the term
        if(out[i]) max = i;
    }
    // return the size of array to house the result.
    return max + 1;

}
//
// Pack the polynomial into a 32 bit array
//

void poly_32_pack( const int *pin, uint32_t* pout, int len )
{
    int lw = len/32;
    int ptr = 0;
    uint32_t temp;
    if( len % 32 ) lw++;

    for( int i = 0; i < lw; i++ )
    {
        temp    = 0x80000000;
        pout[i] = 0;
        for( int j = 0; j < 32; j++ )
        {
            if( pin[ptr++] ) pout[i] |= temp;
            temp >>= 1;
        }
    }
}

void poly_reverse( int *pin, int *pout, int len )
{
    int c;
    c = len-1;

    for( int i = 0; i < len; i++ )
    {
        pout[c--] = pin[i];
    }
}


////////////////////////////////////////////////////////////////////////
//
// Build the 64 bit based lookup tables
// This is only called at start up so speed is not important
//
////////////////////////////////////////////////////////////////////////

//
// Shift a 128 bit register
//
void  inline reg_32_4_shift( uint32_t *sr )
{
    sr[0] = (sr[0]<<1) | (sr[1]>>31);
    sr[1] = (sr[1]<<1) | (sr[2]>>31);
    sr[2] = (sr[2]<<1) | (sr[3]>>31);
    sr[3] = (sr[3]<<1);
}
//
// Shift 160 bits
//
void  inline reg_32_5_shift( uint32_t *sr )
{
    sr[0] = (sr[0]<<1) | (sr[1]>>31);
    sr[1] = (sr[1]<<1) | (sr[2]>>31);
    sr[2] = (sr[2]<<1) | (sr[3]>>31);
    sr[3] = (sr[3]<<1) | (sr[4]>>31);
    sr[4] = (sr[4]<<1);
}
//
// Shift 192 bits
//
void  inline reg_32_6_shift( uint32_t *sr )
{
    sr[0] = (sr[0]<<1) | (sr[1]>>31);
    sr[1] = (sr[1]<<1) | (sr[2]>>31);
    sr[2] = (sr[2]<<1) | (sr[3]>>31);
    sr[3] = (sr[3]<<1) | (sr[4]>>31);
    sr[4] = (sr[4]<<1) | (sr[5]>>31);
    sr[5] = (sr[5]<<1);
}

void bch_n_8_parity_check_lookup_table( void )
{
    uint32_t b;
    uint32_t shift[4];

    for( uint32_t n = 0; n <= 255; n++){
        //Zero the shift register
        memset( shift,0,sizeof(uint32_t)*4);
        shift[0] = (n<<24);
        for( int i = 0; i < 8; i++ )
        {
            b =  (shift[0]&0x80000000);
            reg_32_4_shift( shift );
            if( b )
            {
                shift[0] ^= m_32_poly_n_8[0];
                shift[1] ^= m_32_poly_n_8[1];
                shift[2] ^= m_32_poly_n_8[2];
                shift[3] ^= m_32_poly_n_8[3];
            }
        }
        cb_sr_32_to_64( shift, &m_bch_n8_lookup[n], 2);
    }
}

void bch_n_10_parity_check_lookup_table( void )
{
    uint32_t b;
    uint32_t shift[5];

    for( uint32_t n = 0; n <= 255; n++){
        //Zero the shift register
        memset( shift,0,sizeof(uint32_t)*5);
        shift[0] = (n<<24);
        for( int i = 0; i < 8; i++ )
        {
            b = (shift[0]&0x80000000);
            reg_32_5_shift( shift );
            if(b)
            {
                shift[0] ^= m_32_poly_n_10[0];
                shift[1] ^= m_32_poly_n_10[1];
                shift[2] ^= m_32_poly_n_10[2];
                shift[3] ^= m_32_poly_n_10[3];
                shift[4] ^= m_32_poly_n_10[4];
            }
        }
        cb_sr_32_to_64( shift, &m_bch_n10_lookup[n], 3);
        m_bch_n10_lookup[n].bch_r[2] &= 0xFFFFFFFF00000000;
    }
}

void bch_n_12_parity_check_lookup_table( void )
{
    uint32_t b;
    uint32_t shift[6];
    for( uint32_t n = 0; n <= 255; n++){
        //Zero the shift register
        memset( shift,0,sizeof(uint32_t)*6);
        shift[0] = (n<<24);
        for( int i = 0; i < 8; i++ )
        {
            b =  (shift[0]&0x80000000);
            reg_32_6_shift( shift );
            if(b)
            {
                shift[0] ^= m_32_poly_n_12[0];
                shift[1] ^= m_32_poly_n_12[1];
                shift[2] ^= m_32_poly_n_12[2];
                shift[3] ^= m_32_poly_n_12[3];
                shift[4] ^= m_32_poly_n_12[4];
                shift[5] ^= m_32_poly_n_12[5];
            }
        }
        cb_sr_32_to_64( shift, &m_bch_n12_lookup[n], 3);
    }
}

void bch_s_12_parity_check_lookup_table( void )
{
    uint32_t b;
    uint32_t shift[6];

    for( uint32_t n = 0; n <= 255; n++){
        //Zero the shift register
        memset( shift,0,sizeof(uint32_t)*6);
        shift[0] = (n<<24);
        for( int i = 0; i < 8; i++ )
        {
            b = (shift[0]&0x80000000);
            reg_32_6_shift( shift );
            if(b)
            {
                shift[0] ^= m_32_poly_s_12[0];
                shift[1] ^= m_32_poly_s_12[1];
                shift[2] ^= m_32_poly_s_12[2];
                shift[3] ^= m_32_poly_s_12[3];
                shift[4] ^= m_32_poly_s_12[4];
                shift[5] ^= m_32_poly_s_12[5];
            }
        }
        cb_sr_32_to_64( shift, &m_bch_s12_lookup[n], 3);
        m_bch_s12_lookup[n].bch_r[2] &= 0xFFFFFFFFFF000000;
    }
}

void bch_m_12_parity_check_lookup_table( void )
{
    uint32_t b;
    uint32_t shift[6];

    for( int n = 0; n <= 255; n++){
        //Zero the shift register
        memset( shift,0,sizeof(uint32_t)*6);
        shift[0] = (n<<24);
        for( int i = 0; i < 8; i++ )
        {
            b = (shift[0]&0x80000000);
            reg_32_6_shift( shift );
            if(b)
            {
                shift[0] ^= m_32_poly_m_12[0];
                shift[1] ^= m_32_poly_m_12[1];
                shift[2] ^= m_32_poly_m_12[2];
                shift[3] ^= m_32_poly_m_12[3];
                shift[4] ^= m_32_poly_m_12[4];
                shift[5] ^= m_32_poly_m_12[5];
            }
        }
        cb_sr_32_to_64( shift, &m_bch_m12_lookup[n], 3);
        m_bch_n10_lookup[n].bch_r[2] &= 0xFFFFFFFFFFFFF000;
    }
}

//////////////////////////////////////////////////////////////////////
//
// The actual parity check
//
//////////////////////////////////////////////////////////////////////
//
// 64 bit shifts
//
void  inline reg_64_2_shift( uint64_t *sr )
{
    sr[0] = (sr[0]<<8) | (sr[1]>>56);
    sr[1] = (sr[1]<<8);
}
void  inline reg_64_3_shift( uint64_t *sr )
{
    sr[0] = (sr[0]<<8) | (sr[1]>>56);
    sr[1] = (sr[1]<<8) | (sr[2]>>56);
    sr[2] = (sr[2]<<8);
}

bool bch_h_byte_n_8_parity_check( uint8_t *in, int len ){
    uint64_t shift[2];
    uint8_t b;
    //Zero the shift register 128 bits

    shift[0] = 0;
    shift[1] = 0;

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^in[i];
    	reg_64_2_shift( shift );
        shift[0] ^= m_bch_n8_lookup[b].bch_r[0];
        shift[1] ^= m_bch_n8_lookup[b].bch_r[1];
    }
    if(shift[0] != 0 ) return true;
    if(shift[1] != 0 ) return true;
    return false;
}

bool bch_h_byte_n_10_parity_check( uint8_t *in, int len ){
    uint64_t shift[3];
    uint8_t b;
    //Zero the shift register 160 bits
    memset( shift,0,sizeof(uint64_t)*3);

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^in[i];
    	reg_64_3_shift( shift );
        shift[0] ^= m_bch_n10_lookup[b].bch_r[0];
        shift[1] ^= m_bch_n10_lookup[b].bch_r[1];
        shift[2] ^= m_bch_n10_lookup[b].bch_r[2];
    }
//    printf("N10 %lx %lx %lx\n",shift[0],shift[1],shift[2]);
    if(shift[0] != 0 ) return true;
    if(shift[1] != 0 ) return true;
    if(shift[2] != 0 ) return true;
    return false;
}

bool bch_h_byte_n_12_parity_check( uint8_t *in, int len ){
    uint64_t shift[3];
    uint8_t b;
    //Zero the shift register 192 bits
    memset( shift,0,sizeof(uint64_t)*3);

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^in[i];
    	reg_64_3_shift( shift );
        shift[0] ^= m_bch_n12_lookup[b].bch_r[0];
        shift[1] ^= m_bch_n12_lookup[b].bch_r[1];
        shift[2] ^= m_bch_n12_lookup[b].bch_r[2];
    }
    if(shift[0] != 0 ) return true;
    if(shift[1] != 0 ) return true;
    if(shift[2] != 0 ) return true;
    return false;
}

bool bch_h_byte_s_12_parity_check( uint8_t *in, int len ){
    uint64_t shift[3];
    uint8_t b;
    //Zero the shift register 168 bits
    memset( shift,0,sizeof(uint64_t)*3);

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^in[i];
    	reg_64_3_shift( shift );
        shift[0] ^= m_bch_s12_lookup[b].bch_r[0];
        shift[1] ^= m_bch_s12_lookup[b].bch_r[1];
        shift[2] ^= m_bch_s12_lookup[b].bch_r[2];
    }
    if(shift[0] != 0 ) return true;
    if(shift[1] != 0 ) return true;
    if(shift[2] != 0 ) return true;
    return false;
}

bool bch_h_byte_m_12_parity_check( uint8_t *in, int len ){
    uint64_t shift[3];
    uint8_t b;
    //Zero the shift register 180 bits
    memset( shift,0,sizeof(uint64_t)*3);

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^in[i];
    	reg_64_3_shift( shift );
        shift[0] ^= m_bch_m12_lookup[b].bch_r[0];
        shift[1] ^= m_bch_m12_lookup[b].bch_r[1];
        shift[2] ^= m_bch_m12_lookup[b].bch_r[2];
    }
    if(shift[0] != 0 ) return true;
    if(shift[1] != 0 ) return true;
    if(shift[2] != 0 ) return true;
    return false;
}
//
// Load the constant table required by the CUDA BCH Check code
//
void bch_select_device_lookup_table(void){
	BCHLookup *lu = NULL;
	switch (g_format.pbch)
	{
	case 128:
		lu = m_bch_n8_lookup;
		break;
	case 160:
		lu = m_bch_n10_lookup;
		break;
	case 192:
		lu = m_bch_n12_lookup;
		break;
	case 168:
		lu = m_bch_s12_lookup;
		break;
	case 180:
		// Use same as N
		lu = m_bch_m12_lookup;
		break;
	default:
		// Unknown
		lu = m_bch_n12_lookup;
		break;
	}
	bch_d_copy_lookup(lu);
}
///////////////////////////////////////////////////////////////
//
// Initialisation
//
///////////////////////////////////////////////////////////////

void bch_open_b( void )
{
    // Normal polynomials
    const int polyn01[]={1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1};
    const int polyn02[]={1,1,0,0,1,1,1,0,1,0,0,0,0,0,0,0,1};
    const int polyn03[]={1,0,1,1,1,1,0,1,1,1,1,1,0,0,0,0,1};
    const int polyn04[]={1,0,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1};
    const int polyn05[]={1,1,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1};
    const int polyn06[]={1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1};
    const int polyn07[]={1,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,1};
    const int polyn08[]={1,1,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1};
    const int polyn09[]={1,0,0,0,0,1,0,1,0,1,1,1,0,0,0,0,1};
    const int polyn10[]={1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,1};
    const int polyn11[]={1,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,1};
    const int polyn12[]={1,1,0,0,0,1,1,1,0,1,0,1,1,0,0,0,1};

    // Medium polynomials
    const int polym01[]={1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,1};
    const int polym02[]={1,1,0,0,1,0,0,1,0,0,1,1,0,0,0,1};
    const int polym03[]={1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1};
    const int polym04[]={1,0,1,1,0,1,1,0,1,0,1,1,0,0,0,1};
    const int polym05[]={1,1,1,0,1,0,1,1,0,0,1,0,1,0,0,1};
    const int polym06[]={1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,1};
    const int polym07[]={1,0,1,0,1,1,0,1,0,0,0,1,1,0,1,1};
    const int polym08[]={1,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1};
    const int polym09[]={1,1,1,0,1,1,0,1,0,1,0,1,1,1,0,1};
    const int polym10[]={1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,1};
    const int polym11[]={1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1};
    const int polym12[]={1,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1};

    // Short polynomials
    const int polys01[]={1,1,0,1,0,1,0,0,0,0,0,0,0,0,1};
    const int polys02[]={1,0,0,0,0,0,1,0,1,0,0,1,0,0,1};
    const int polys03[]={1,1,1,0,0,0,1,0,0,1,1,0,0,0,1};
    const int polys04[]={1,0,0,0,1,0,0,1,1,0,1,0,1,0,1};
    const int polys05[]={1,0,1,0,1,0,1,0,1,1,0,1,0,1,1};
    const int polys06[]={1,0,0,1,0,0,0,1,1,1,0,0,0,1,1};
    const int polys07[]={1,0,1,0,0,1,1,1,0,0,1,1,0,1,1};
    const int polys08[]={1,0,0,0,0,1,0,0,1,1,1,1,0,0,1};
    const int polys09[]={1,1,1,1,0,0,0,0,0,1,1,0,0,0,1};
    const int polys10[]={1,0,0,1,0,0,1,0,0,1,0,1,1,0,1};
    const int polys11[]={1,0,0,0,1,0,0,0,0,0,0,1,1,0,1};
    const int polys12[]={1,1,1,1,0,1,1,1,1,0,1,0,0,1,1};

    int len;
    int polyout[2][200];

    // Normal
    memset(polyout[0],0,sizeof(int)*200);
    memset(polyout[1],0,sizeof(int)*200);
    len = poly_mult( polyn01, 17, polyn02,    17,  polyout[0] );
    len = poly_mult( polyn03, 17, polyout[0], len, polyout[1] );
    len = poly_mult( polyn04, 17, polyout[1], len, polyout[0] );
    len = poly_mult( polyn05, 17, polyout[0], len, polyout[1] );
    len = poly_mult( polyn06, 17, polyout[1], len, polyout[0] );
    len = poly_mult( polyn07, 17, polyout[0], len, polyout[1] );
    len = poly_mult( polyn08, 17, polyout[1], len, polyout[0] );
    poly_reverse( polyout[0], polyout[1], 128 );
    poly_32_pack( polyout[1], m_32_poly_n_8, 128 );

    len = poly_mult( polyn09, 17, polyout[0], len, polyout[1] );
    len = poly_mult( polyn10, 17, polyout[1], len, polyout[0] );
    poly_reverse( polyout[0], polyout[1], 160 );
    poly_32_pack( polyout[1], m_32_poly_n_10, 160 );

    len = poly_mult( polyn11, 17, polyout[0], len, polyout[1] );
    len = poly_mult( polyn12, 17, polyout[1], len, polyout[0] );
    poly_reverse( polyout[0], polyout[1], 192 );
    poly_32_pack( polyout[1], m_32_poly_n_12, 192 );

    // Medium
    memset(polyout[0],0,sizeof(int)*200);
    memset(polyout[1],0,sizeof(int)*200);
    len = poly_mult( polym01, 16, polym02,    16,  polyout[0] );
    len = poly_mult( polym03, 16, polyout[0], len, polyout[1] );
    len = poly_mult( polym04, 16, polyout[1], len, polyout[0] );
    len = poly_mult( polym05, 16, polyout[0], len, polyout[1] );
    len = poly_mult( polym06, 16, polyout[1], len, polyout[0] );
    len = poly_mult( polym07, 16, polyout[0], len, polyout[1] );
    len = poly_mult( polym08, 16, polyout[1], len, polyout[0] );
    len = poly_mult( polym09, 16, polyout[0], len, polyout[1] );
    len = poly_mult( polym10, 16, polyout[1], len, polyout[0] );
    len = poly_mult( polym11, 16, polyout[0], len, polyout[1] );
    len = poly_mult( polym12, 16, polyout[1], len, polyout[0] );
    poly_reverse( polyout[0], polyout[1], 180 );
    poly_32_pack( polyout[1], m_32_poly_m_12, 180 );


    // Short
    memset(polyout[0],0,sizeof(int)*200);
    memset(polyout[1],0,sizeof(int)*200);
    len = poly_mult( polys01, 15, polys02,    15,  polyout[0] );
    len = poly_mult( polys03, 15, polyout[0], len, polyout[1] );
    len = poly_mult( polys04, 15, polyout[1], len, polyout[0] );
    len = poly_mult( polys05, 15, polyout[0], len, polyout[1] );
    len = poly_mult( polys06, 15, polyout[1], len, polyout[0] );
    len = poly_mult( polys07, 15, polyout[0], len, polyout[1] );
    len = poly_mult( polys08, 15, polyout[1], len, polyout[0] );
    len = poly_mult( polys09, 15, polyout[0], len, polyout[1] );
    len = poly_mult( polys10, 15, polyout[1], len, polyout[0] );
    len = poly_mult( polys11, 15, polyout[0], len, polyout[1] );
    len = poly_mult( polys12, 15, polyout[1], len, polyout[0] );
    poly_reverse( polyout[0], polyout[1], 168 );
    poly_32_pack( polyout[1], m_32_poly_s_12, 168 );

    // Build the lookup tables for the fast encoder
    bch_n_8_parity_check_lookup_table();
    bch_n_10_parity_check_lookup_table();
    bch_n_12_parity_check_lookup_table();
    bch_s_12_parity_check_lookup_table();
    bch_m_12_parity_check_lookup_table();
}
