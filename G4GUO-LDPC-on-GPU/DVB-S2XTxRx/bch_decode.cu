#include "cuda_runtime.h"
#include "dvbs2_rx.h"

#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <stdint.h>

extern RxFormat g_format;

void build_hmatrix(void);

#define u15 unsigned short
#define u14 unsigned short
#define u32 unsigned int

#define G_P0_SIZE_N 0x10000
#define G_P0_SIZE_M 0x8000
#define G_P0_SIZE_S 0x4000
//
// Inversion table lookups
//
//#define BUILD_BCH_TABLES
#ifdef BUILD_BCH_TABLES
uint16_t m_p0invn[G_P0_SIZE_N];
uint16_t m_p0invm[G_P0_SIZE_M];
uint16_t m_p0invs[G_P0_SIZE_S];
#else
extern uint16_t m_p0invn[G_P0_SIZE_N];
extern uint16_t m_p0invm[G_P0_SIZE_M];
extern uint16_t m_p0invs[G_P0_SIZE_S];
#endif

// Polynomial Log tables
uint16_t m_logtabn[0x10000];
uint16_t m_alogtabn[0x10020];
uint16_t m_logtabs[0x4000];
uint16_t m_alogtabs[0x4020];
uint16_t m_logtabm[0x8000];
uint16_t m_alogtabm[0x8020];

uint16_t *ds;
uint16_t *dz;
uint16_t *d_logn;
uint16_t *d_alogn;
uint16_t *d_logs;
uint16_t *d_alogs;
uint16_t *d_logm;
uint16_t *d_alogm;

int m_nsize;

//
// Calculate the power of a Short polynomial
//
__device__ uint16_t cgmpwrs(u14 a, u14 pwr)
{
	u32 out;
	out = a;
	for (int i = 0; i < pwr; i++){
		out <<=1;
		if (out & 0x4000) out ^= 0x002B;
	}
	return out & 0x3FFF;
}
//
// Calculate the syndromes
//
__global__ void bchSSyndrome(Bit *in, int len, u14 *s){
	int i = threadIdx.x + 1;
	//Zero the shift register
	uint16_t sr = 0;

	for (int n = 0; n < len; n++) sr = cgmpwrs(sr, i) ^ in[n];
	s[i] = sr;
}

//
// Calculate the power of a Normal polynomial
//
inline __device__ uint16_t cgmpwrn( uint16_t a, uint16_t pwr)
{
	for (int i = 0; i < pwr; i++){
		a = (a & 0x8000) ? (a<<1) ^ 0x002D : a<<1;
	}
	return a;
}
//
// Calculate the syndromes
//
__global__ void bchNSyndrome( Bit *in, int len, uint16_t *s ){
	int i = threadIdx.x + 1;
	//Zero the shift register
	uint16_t sr = in[0];

	for (int n = 1; n < len; n++) sr = cgmpwrn(sr, i) ^ in[n];

	s[i] = sr;
}


__global__ void bchNSyndrome1( Bit *in, int len, uint16_t *s ){
	int i = threadIdx.x + 1;
	//Zero the shift register
	uint32_t sr = in[0];

	for (int n = 1; n < len; n++){
		for (int j = 0; j < i; j++){
			sr = (sr<<1);
			if (sr & 0x10000) sr ^= 0x002D;
		}
		sr ^= in[n];
	}
	s[i] = sr&0xFFFF;
}
//
// Calculate the odd syndromes of the 128 Frames, return them in consecutive array entries.
//
__global__ void bchNSyndrome2( Bit *in, int len, uint16_t *s, const uint16_t  * lg, const uint16_t * alg){
	int i = (threadIdx.x*2) + 1;
	int m = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t k = blockIdx.x * len;
	//Zero the shift register
    uint16_t sr = 0;

	for (uint32_t n = k; n < k+len; n++){
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>7)&1) : ((in[n]>>7)&1);
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>6)&1) : ((in[n]>>6)&1);
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>5)&1) : ((in[n]>>5)&1);
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>4)&1) : ((in[n]>>4)&1);
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>3)&1) : ((in[n]>>3)&1);
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>2)&1) : ((in[n]>>2)&1);
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>1)&1) : ((in[n]>>1)&1);
		sr = sr ? alg[lg[sr]+i] ^ ((in[n]>>0)&1) : ((in[n]>>0)&1);
	}
	s[m] = sr;
}

__global__ void bchNSyndrome8( Bit *in, int len, uint16_t *s ){
	int i = (threadIdx.x*2) + 1;
	int m = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t k = blockIdx.x * len;

	//Zero the shift register

	uint16_t sr = 0;

	for (uint32_t n = k; n < k+len; n++){
		sr = cgmpwrn(sr, i) ^ ((in[n]>>7)&1);
		sr = cgmpwrn(sr, i) ^ ((in[n]>>6)&1);
		sr = cgmpwrn(sr, i) ^ ((in[n]>>5)&1);
		sr = cgmpwrn(sr, i) ^ ((in[n]>>4)&1);
		sr = cgmpwrn(sr, i) ^ ((in[n]>>3)&1);
		sr = cgmpwrn(sr, i) ^ ((in[n]>>2)&1);
		sr = cgmpwrn(sr, i) ^ ((in[n]>>1)&1);
		sr = cgmpwrn(sr, i) ^ ((in[n])&1);
	}
	s[m] = sr;
}

//
// Fix errors, assume packed into 8 bits
//
__global__ void bchFixErrors8(Bit *in, int len, uint16_t *z){
	int i = threadIdx.x;
	int index = len - (z[i] + 1);
	in[index/8] ^= (0x80>>(index%8));
}
//
// Pack the bit array into a byte array return inplace
//
__global__ void compactto_8( Bit *bio ){
    int n = (blockIdx.x * blockDim.x)+threadIdx.x;
    int m = n<<3;
    uint8_t b;
    b  = (bio[m]<<7)|(bio[m+1]<<6)|(bio[m+2]<<5)|(bio[m+3]<<4)|(bio[m+4]<<3)|(bio[m+5]<<2)|(bio[m+6]<<1)|(bio[m+7]);
    bio[n] = b;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Below this line is standard host C code
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////

uint16_t (*gmult)(uint16_t,uint16_t);
uint16_t *ginvtab;

#define ginv(a)    ginvtab[a]
#define gadd(a,b) (a^b)
#define gpwrn(a,b) (bch_alpha_powern(a,b))
//
// Normal block
//
uint16_t bch_poly_mult0_n(uint16_t a, uint16_t b){

	u32 sr = 0;
	for (int i = 0; i < 16; i++){
		sr <<= 1;
		if (a & 0x8000)   sr ^= b;
		if (sr & 0x10000) sr ^= 0x002D;//a^16 = (a + 1)
		a <<= 1;
	}
	return (uint16_t)(sr & 0xFFFF);
}
//
// Medium block
//
uint16_t bch_poly_mult0_m(uint16_t a, uint16_t b){

	u32 sr = 0;
	for (int i = 0; i < 15; i++){
		sr <<= 1;
		if (a & 0x4000)  sr ^= b;
		if (sr & 0x8000) sr ^= 0x002D;//a^15 = (a + 1)
		a <<= 1;
	}
	return (uint16_t)(sr & 0x7FFF);
}
//
// Short block
//
uint16_t bch_poly_mult0_s(uint16_t a, uint16_t b){

	uint16_t sr = 0;
	for (int i = 0; i < 14; i++){
		sr <<= 1;
		if (a & 0x2000)   sr ^= b;
		if (sr & 0x4000) sr ^= 0x002B;//a^14 = (a + 1)
		a <<= 1;
	}
	return (uint16_t)(sr & 0x3FFF);
}
//
// Determines whether it is a Normal, Medium or Short block
//
void bch_set_decoder( void ){

	switch(g_format.bch){

	case BCH_N8:
	case BCH_N10:
	case BCH_N12:
		gmult   = bch_poly_mult0_n;
		ginvtab = m_p0invn;
		m_nsize = G_P0_SIZE_N;
		break;
	case BCH_S12:
		gmult   = bch_poly_mult0_s;
		ginvtab = m_p0invs;
		m_nsize = G_P0_SIZE_S;
		break;
	case BCH_M12:
		gmult   = bch_poly_mult0_m;
		ginvtab = m_p0invm;
		m_nsize = G_P0_SIZE_M;
		break;
	}
}

uint16_t bch_alpha_powern(uint16_t s, int pwr){
	uint32_t out;
	out = s;
	for (int i = 0; i < pwr; i++){
		out <<= 1;
		if (out & 0x10000) out ^= 0x002D;
	}
	return (out & 0xFFFF);
}

uint16_t bch_alpha_powerm(uint16_t s, int pwr){
	unsigned int out;
	out = s;
	for (int i = 0; i < pwr; i++){
		if (out & 0x4000)
			out = (out << 1) ^ 0x002B;// needs looking at
		else
			out = (out << 1);
	}
	return out & 0x7FFF;
}

uint16_t bch_alpha_powers(uint16_t s, int pwr){
	unsigned int out;
	out = s;
	for (int i = 0; i < pwr; i++){
		if (out & 0x2000)
			out = (out << 1) ^ 0x002B;
		else
			out = (out << 1);
	}
	return out & 0x3FFF;
}

void bch_poly_copy(uint16_t *out, uint16_t *in, int l)
{
	for (int i = 0; i < l; i++) out[i] = in[i];
}
void bch_poly_add(uint16_t *out, uint16_t *ina, uint16_t *inb, int l)
{
	for (int i = 0; i < l; i++) out[i] = ina[i] ^ inb[i];
}
void bch_polyn_mult(uint16_t *out, uint16_t *ina, uint16_t v, int px, int l)
{
	for (int i = 0; i < (l - px); i++) out[i + px] = gmult(ina[i], v);
}
//
// Build the log and anti log tables for the polynomials
//
void bch_alog_log_build_table(void){

	// First do the Normal frame polynomial
	m_alogtabn[0]     = 0;
	m_alogtabn[1]     = 1;
	m_logtabn[0]      = 0;
	m_logtabn[1]      = 1;
	for( unsigned int i = 2; i < (0x10020); i++){
		m_alogtabn[i] = bch_alpha_powern(m_alogtabn[i-1], 1);
	}
	for( unsigned int i = 0; i < 0x10000; i++){
		m_logtabn[m_alogtabn[i]] = i;
	}

	CHECK(cudaMemcpy(d_logn,   m_logtabn,  sizeof(uint16_t) * (0x10000), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_alogn,  m_alogtabn, sizeof(uint16_t) * (0x10020), cudaMemcpyHostToDevice));

	// Now do the Short frame polynomial
	m_alogtabs[0] = 0;
	m_alogtabs[1] = 1;
	m_logtabs[0]  = 0;
	m_logtabs[1]  = 1;
	for( int i = 2; i < 0x4020; i++){
		m_alogtabs[i] = bch_alpha_powers(m_alogtabs[i-1], 1);
	}
	for( int i = 2; i < 0x4000; i++){
		m_logtabs[m_alogtabs[i]] = i;
	}

	CHECK(cudaMemcpy(d_logs,   m_logtabs,  sizeof(uint16_t) * 0x4000, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_alogs,  m_alogtabs, sizeof(uint16_t) * 0x4020, cudaMemcpyHostToDevice));

	// Now do the Medium frame polynomial
	m_alogtabm[0] = 0;
	m_alogtabm[1] = 1;
	m_logtabm[0]  = 0;
	m_logtabm[1]  = 1;
	for( int i = 2; i < 0x8020; i++){
		m_alogtabm[i] = bch_alpha_powerm(m_alogtabm[i-1], 1);
	}
	for( unsigned int i = 0; i < 0x8000; i++){
		m_logtabm[m_alogtabm[i]] = i;
	}
	CHECK(cudaMemcpy(d_logm,   m_logtabm,  sizeof(uint16_t) * 0x8000, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_alogm,  m_alogtabm, sizeof(uint16_t) * 0x8020, cudaMemcpyHostToDevice));

}
#ifdef BUILD_BCH_TABLES

void bch_build_poly0n_inv_tables(void){
	gmult = bch_poly_mult0_n;
	ginvtab = m_p0invn;
	m_nsize = G_P0_SIZE_N;

	memset(m_p0invn, 0, sizeof(uint16_t)*G_P0_SIZE_N);

	for (int i = 1; i < G_P0_SIZE_N; i++){
		if (m_p0invn[i] == 0){
			for (int j = i; j < G_P0_SIZE_N; j++){
				if (m_p0invn[j] == 0){
					if (gmult(i, j) == 0x0001){
						m_p0invn[i] = j;
						m_p0invn[j] = i;
						break;
					}
				}
			}
		}
	}
}
void bch_build_poly0s_inv_tables(void){

	gmult = bch_poly_mult0_s;
	ginvtab = m_p0invs;
	m_nsize = G_P0_SIZE_S;

	memset(m_p0invs, 0, sizeof(uint16_t)*G_P0_SIZE_S);

	for (int i = 1; i < G_P0_SIZE_S; i++){
		if (m_p0invs[i] == 0){
			for (int j = i; j < G_P0_SIZE_S; j++){
				if (m_p0invs[j] == 0){
					if (gmult(i, j) == 0x0001){
						m_p0invs[i] = j;
						m_p0invs[j] = i;
						break;
					}
				}
			}
		}
	}
}
void bch_build_poly0m_inv_tables(void){

	gmult = bch_poly_mult0_m;
	ginvtab = m_p0invm;
	m_nsize = G_P0_SIZE_M;

	memset(m_p0invm, 0, sizeof(uint16_t)*G_P0_SIZE_M);

	for (int i = 1; i < G_P0_SIZE_M; i++){
		if (m_p0invm[i] == 0){
			for (int j = i; j < G_P0_SIZE_M; j++){
				if (m_p0invm[j] == 0){
					if (gmult(i, j) == 0x0001){
						m_p0invm[i] = j;
						m_p0invm[j] = i;
						break;
					}
				}
			}
		}
	}
}
//
// Used to create inversion tables, they are large but save time
//
void bch_save_poly_inversion_tables(void){
	FILE *fp;
	int index;
	if ((fp = fopen("poly_inv_tab.cpp","w")) != NULL){
		fprintf(fp, "//\n// Computer generated files do not edit\n//\n");
		fprintf(fp, "#include <stdint.h>\n");
		fprintf(fp, "uint16_t m_p0invn[%d]={\n", G_P0_SIZE_N);
		index = 0;
		for (int i = 0; i < G_P0_SIZE_N/8; i++){
			fprintf(fp, "\t");
			for (int j = 0; j < 8; j++){
				fprintf(fp,"0x%.4x",m_p0invn[index]);
				if (index != (G_P0_SIZE_N-1))
					fprintf(fp, ", ");
				else{
					fprintf(fp, "  ");
					break;
				}
				index++;
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "};\n");
		//
		fprintf(fp, "//\n//\n//\n");
		fprintf(fp, "uint16_t m_p0invm[%d]={\n", G_P0_SIZE_M);
		index = 0;
		for (int i = 0; i < G_P0_SIZE_M / 8; i++){
			fprintf(fp, "\t");
			for (int j = 0; j < 8; j++){
				fprintf(fp, "0x%.4x", m_p0invm[index]);
				if (index != (G_P0_SIZE_M - 1))
					fprintf(fp, ", ");
				else{
					fprintf(fp, "  ");
					break;
				}
				index++;
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "};\n");
		//
		fprintf(fp, "//\n//\n//\n");
		fprintf(fp, "uint16_t m_p0invs[%d]={\n", G_P0_SIZE_S);
		index = 0;
		for (int i = 0; i < G_P0_SIZE_S / 8; i++){
			fprintf(fp, "\t");
			for (int j = 0; j < 8; j++){
				fprintf(fp, "0x%.4x", m_p0invs[index]);
				if (index != (G_P0_SIZE_S - 1))
					fprintf(fp, ", ");
				else{
					fprintf(fp, "  ");
					break;
				}
				index++;
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "};\n");

		fclose(fp);
	}
}
#endif

//
// After the errors positions have been found fix them
//
void host_bch_fix_errors(Bit *in, int len, uint16_t *z, int r){
	for( int i = 0; i < r; i++){
		int pos = len - (z[i] + 1);
		if((pos >= 0)&&(pos < len)) in[pos] ^= 1;
	}
}
void host_bch_fix_errors_8(Bit *in, int len, uint16_t *z, int r){
	for( int i = 0; i < r; i++){
		int pos =  (len*8) - (z[i] + 1);
		if((pos >= 0)&&(pos < len)) in[pos/8] ^= (0x80>>(pos%8));
	}
}
//
// Pack the bit array into a byte array return inplace
//
int host_bch_pack_bits(uint8_t *out, Bit *in,  int len){
	uint8_t b;
	int j = 0;
	for( int i = 0; i < len; i+= 8){
		b =  (in[i]   << 7 ) | (in[i+1] << 6) | (in[i+2] << 5) | \
		     (in[i+3] << 4 ) | (in[i+4] << 3) | (in[i+5] << 2) | \
		     (in[i+6] << 1 ) | (in[i+7] << 0);
		out[j++] = b;
	}
	return j;
}

//////////////////////////////////////////////////////////////////////////////////////
//
// Routines to correct the errors from the syndromes
//
//////////////////////////////////////////////////////////////////////////////////////

static void  bch_mas(uint16_t *s, uint16_t *out, int t_size)
{
	typedef struct{
		uint16_t ar[30];
	}tau;

	tau au_array[30];
	uint16_t du_array[30];
	uint16_t hu_array[30];

	int p = 0;
	int pmax = -2;
	int pv = 0;
	int pwr = 0;

	t_size = (2 * t_size) + 1;

	// Zero all the arrays
	memset(au_array, 0, sizeof(tau) * 30);
	memset(du_array, 0, sizeof(uint16_t) * 30);
	memset(hu_array, 0, sizeof(uint16_t) * 30);

	// C arrays start at 0
	tau *a = &au_array[1];
	uint16_t *d = &du_array[1];
	uint16_t *h = &hu_array[1];
	uint16_t *sd = &s[0];

	// Initial conditions
	int u = 0;

	a[-1].ar[0] = 1;
	d[-1] = 1;
	h[-1] = 0;

	a[0].ar[0] = 1;
	d[0] = sd[1];
	h[0] = 0;

	for (int row = 1; row < t_size; row++)
	{
		if (d[u] == 0)
		{
			bch_poly_copy(a[u + 1].ar, a[u].ar, t_size);
			h[u + 1] = h[u];
		}
		else
		{
			// Find the largest row
			for (int r = -1; r < u; r++)
			{
				pv = r - h[r];
				if ((pv > pmax) && (d[r] != 0))
				{
					p = r;
					pmax = pv;
				}
			}
			// Calculate new au value
			uint16_t v = gmult(d[u], ginv(d[p]));
			pwr = u - p;
			bch_polyn_mult(a[u + 1].ar, a[p].ar, v, pwr, t_size);
			bch_poly_add(a[u + 1].ar, a[u + 1].ar, a[u].ar, t_size);
			h[u + 1] = (h[u] > (h[p] + u - p)) ? h[u] : (h[p] + u - p);
		}
		int an = 1;
		d[u + 1] = sd[u + 2];
		for (int sn = u + 1; sn >= (u + 2 - h[u + 1]); sn--)
		{
			d[u + 1] = gadd(d[u + 1], gmult(sd[sn], a[u + 1].ar[an++]));
		}
		u++;
	}
	memcpy(out, a[t_size - 1].ar, sizeof(uint16_t)*(t_size));
}

static int bch_chien_search(uint16_t *a, uint16_t *z, int tsize)
{
	// Calculate the reciprocal of the coefficients
	int flag, order, nr_coffs;
	uint16_t ord[30];
	uint16_t t[30];
	order = 0;
	nr_coffs = 0;
	flag = 0;

	tsize = (2 * tsize) + 1;

	for (int i = tsize; i >= 0; i--)
	{
		if (a[i] != 0)
		{
			if (flag)
			{
				ord[nr_coffs] = order;
				t[nr_coffs] = a[i];
				nr_coffs++;
			}
			else
			{
				order = 0;
				t[0] = a[i];
				ord[nr_coffs] = order;
				flag = 1;
				nr_coffs = 1;
			}
		}
		order++;
	}
	//	printf("Chien nr %d Order %d\n",nr_coffs,ord[nr_coffs-1]);
	//	for( int i = 0; i < nr_coffs; i++ ) printf("Order %d Coeff %.2X\n",ord[i],t[i]);

	int eq_order = ord[nr_coffs - 1];

	//
	// We now have the inverse of sigma, we can do an exhaustive search
	// to find the roots
	//
	int roots = 0;
	uint16_t y[200];
	uint16_t sum;

	// Sigma ^ 0
	sum = 0;

	for (int m = 0; m < nr_coffs; m++)
	{
		sum = sum ^ t[m];
		y[m] = t[m];
	}

	if (sum == 0)
	{
		z[roots] = 0;
		roots++;

		if (roots == eq_order) return roots;
	}

	// sigma ^ > 0

	for (int n = 1; n < m_nsize; n++)
	{
		sum = y[0];
		for (int m = 1; m < nr_coffs; m++)
		{
			y[m] = gmult(y[m], gpwrn(1, ord[m]));
			sum  = gadd(sum, y[m]);
		}
		if (sum == 0)
		{
			z[roots] = n;
			roots++;
			if (roots == eq_order) return roots;
		}
	}

	return roots;
}


//
//
// extra required for other sizes
//
void bch_n_8_syndrome(uint8_t *in, uint16_t *s, int len){
	uint8_t b;
	memset(s, 0, sizeof(uint16_t) * 17);

	for (int i = 0; i < len; i++) {
		for( int k = 7; k >= 0; k--){
			b = (in[i]>>k)&1;
		    s[1]  = s[1]  ? m_alogtabn[(m_logtabn[s[1]]+1)]   ^ b : b;
		    s[3]  = s[3]  ? m_alogtabn[(m_logtabn[s[3]]+3)]   ^ b : b;
		    s[5]  = s[5]  ? m_alogtabn[(m_logtabn[s[5]]+5)]   ^ b : b;
		    s[7]  = s[7]  ? m_alogtabn[(m_logtabn[s[7]]+7)]   ^ b : b;
		    s[9]  = s[9]  ? m_alogtabn[(m_logtabn[s[9]]+9)]   ^ b : b;
		    s[11] = s[11] ? m_alogtabn[(m_logtabn[s[11]]+11)] ^ b : b;
		    s[13] = s[13] ? m_alogtabn[(m_logtabn[s[13]]+13)] ^ b : b;
		    s[15] = s[15] ? m_alogtabn[(m_logtabn[s[15]]+15)] ^ b : b;
		}
	}

	s[2]  = gmult(s[1], s[1]);
	s[4]  = gmult(s[2], s[2]);
	s[6]  = gmult(s[3], s[3]);
	s[8]  = gmult(s[4], s[4]);
	s[10] = gmult(s[5], s[5]);
	s[12] = gmult(s[6], s[6]);
	s[14] = gmult(s[7], s[7]);
	s[16] = gmult(s[8], s[8]);
}

void bch_n_10_syndrome(uint8_t *in, uint16_t *s, int len){
	uint8_t b;
	memset(s, 0, sizeof(uint16_t) * 21);

	for (int i = 0; i < len; i++) {
		for( int k = 7; k >= 0; k--){
			b = (in[i]>>k)&1;
		    s[1]  = s[1]  ? m_alogtabn[(m_logtabn[s[1]]+1)]   ^ b : b;
		    s[3]  = s[3]  ? m_alogtabn[(m_logtabn[s[3]]+3)]   ^ b : b;
		    s[5]  = s[5]  ? m_alogtabn[(m_logtabn[s[5]]+5)]   ^ b : b;
		    s[7]  = s[7]  ? m_alogtabn[(m_logtabn[s[7]]+7)]   ^ b : b;
		    s[9]  = s[9]  ? m_alogtabn[(m_logtabn[s[9]]+9)]   ^ b : b;
		    s[11] = s[11] ? m_alogtabn[(m_logtabn[s[11]]+11)] ^ b : b;
		    s[13] = s[13] ? m_alogtabn[(m_logtabn[s[13]]+13)] ^ b : b;
		    s[15] = s[15] ? m_alogtabn[(m_logtabn[s[15]]+15)] ^ b : b;
		    s[17] = s[17] ? m_alogtabn[(m_logtabn[s[17]]+17)] ^ b : b;
		    s[19] = s[19] ? m_alogtabn[(m_logtabn[s[19]]+19)] ^ b : b;
		}
	}

	s[2]  = gmult(s[1],  s[1]);
	s[4]  = gmult(s[2],  s[2]);
	s[6]  = gmult(s[3],  s[3]);
	s[8]  = gmult(s[4],  s[4]);
	s[10] = gmult(s[5],  s[5]);
	s[12] = gmult(s[6],  s[6]);
	s[14] = gmult(s[7],  s[7]);
	s[16] = gmult(s[8],  s[8]);
	s[18] = gmult(s[9],  s[9]);
	s[20] = gmult(s[10], s[10]);
}

void bch_n_12_syndrome( uint8_t *in, uint16_t *s, int len){
	uint8_t b;
	memset(s, 0, sizeof(uint16_t) * 25);

	for (int i = 0; i < len; i++) {
		for( int k = 7; k >= 0; k--){
			b = (in[i]>>k)&1;
		    s[1]  = s[1]  ? m_alogtabn[(m_logtabn[s[1]]+1)]   ^ b : b;
		    s[3]  = s[3]  ? m_alogtabn[(m_logtabn[s[3]]+3)]   ^ b : b;
		    s[5]  = s[5]  ? m_alogtabn[(m_logtabn[s[5]]+5)]   ^ b : b;
		    s[7]  = s[7]  ? m_alogtabn[(m_logtabn[s[7]]+7)]   ^ b : b;
		    s[9]  = s[9]  ? m_alogtabn[(m_logtabn[s[9]]+9)]   ^ b : b;
		    s[11] = s[11] ? m_alogtabn[(m_logtabn[s[11]]+11)] ^ b : b;
		    s[13] = s[13] ? m_alogtabn[(m_logtabn[s[13]]+13)] ^ b : b;
		    s[15] = s[15] ? m_alogtabn[(m_logtabn[s[15]]+15)] ^ b : b;
		    s[17] = s[17] ? m_alogtabn[(m_logtabn[s[17]]+17)] ^ b : b;
		    s[19] = s[19] ? m_alogtabn[(m_logtabn[s[19]]+19)] ^ b : b;
		    s[21] = s[21] ? m_alogtabn[(m_logtabn[s[21]]+21)] ^ b : b;
		    s[23] = s[23] ? m_alogtabn[(m_logtabn[s[23]]+23)] ^ b : b;
		}
	}

	s[2]  = gmult(s[1],  s[1]);
	s[4]  = gmult(s[2],  s[2]);
	s[6]  = gmult(s[3],  s[3]);
	s[8]  = gmult(s[4],  s[4]);
	s[10] = gmult(s[5],  s[5]);
	s[12] = gmult(s[6],  s[6]);
	s[14] = gmult(s[7],  s[7]);
	s[16] = gmult(s[8],  s[8]);
	s[18] = gmult(s[9],  s[9]);
	s[20] = gmult(s[10], s[10]);
	s[22] = gmult(s[11], s[11]);
	s[24] = gmult(s[12], s[12]);
}

void bch_s_12_syndrome(uint8_t *in, uint16_t *s, int len){
	uint8_t b;
	memset(s, 0, sizeof(uint16_t) * 25);

	for (int i = 0; i < len; i++) {
		for( int k = 7; k >= 0; k--){
			b = (in[i]>>k)&1;
		    s[1]  = s[1]  ? m_alogtabs[(m_logtabs[s[1]]+1)]   ^ b : b;
		    s[3]  = s[3]  ? m_alogtabs[(m_logtabs[s[3]]+3)]   ^ b : b;
		    s[5]  = s[5]  ? m_alogtabs[(m_logtabs[s[5]]+5)]   ^ b : b;
		    s[7]  = s[7]  ? m_alogtabs[(m_logtabs[s[7]]+7)]   ^ b : b;
		    s[9]  = s[9]  ? m_alogtabs[(m_logtabs[s[9]]+9)]   ^ b : b;
		    s[11] = s[11] ? m_alogtabs[(m_logtabs[s[11]]+11)] ^ b : b;
		    s[13] = s[13] ? m_alogtabs[(m_logtabs[s[13]]+13)] ^ b : b;
		    s[15] = s[15] ? m_alogtabs[(m_logtabs[s[15]]+15)] ^ b : b;
		    s[17] = s[17] ? m_alogtabs[(m_logtabs[s[17]]+17)] ^ b : b;
		    s[19] = s[19] ? m_alogtabs[(m_logtabs[s[19]]+19)] ^ b : b;
		    s[21] = s[21] ? m_alogtabs[(m_logtabs[s[21]]+21)] ^ b : b;
		    s[23] = s[23] ? m_alogtabs[(m_logtabs[s[23]]+23)] ^ b : b;
		}
	}

	s[2]  = gmult(s[1],  s[1]);
	s[4]  = gmult(s[2],  s[2]);
	s[6]  = gmult(s[3],  s[3]);
	s[8]  = gmult(s[4],  s[4]);
	s[10] = gmult(s[5],  s[5]);
	s[12] = gmult(s[6],  s[6]);
	s[14] = gmult(s[7],  s[7]);
	s[16] = gmult(s[8],  s[8]);
	s[18] = gmult(s[9],  s[9]);
	s[20] = gmult(s[10], s[10]);
	s[22] = gmult(s[11], s[11]);
	s[24] = gmult(s[12], s[12]);
}

///////////////////////////////////////////////////////////////////////
//
// Host decoder
// This is only called if there are errors in the frame
//
///////////////////////////////////////////////////////////////////////

int bch_host_byte_decode( uint8_t *in  )
{
	uint16_t s[30];
	uint16_t t[30];
	uint16_t z[30];

	int r = 0;

	int len = g_format.nbch/8;
	memset(t, 0, 30 * sizeof(uint16_t));
	switch (g_format.bch)
	{
	case BCH_N8:
		bch_n_8_syndrome( in, s, len );
		// There are errors
		bch_mas( s, t, 8 );
		r = bch_chien_search( t, z, 8 );
		if(r != 0)
			host_bch_fix_errors_8( in, len, z, r );
		else
			r = 8;// Too many errors to correct
		break;
	case BCH_N10:
		bch_n_10_syndrome( in, s, len );
		// There are errors
		bch_mas( s, t, 10 );
		r = bch_chien_search( t, z, 10 );
		if(r != 0)
			host_bch_fix_errors_8( in, len, z, r );
		else
			r = 10;
		break;
	case BCH_N12:
		bch_n_12_syndrome( in, s, len );
		// There are errors
		bch_mas( s, t, 12 );
		r = bch_chien_search( t, z, 12 );
		if(r != 0)
		    host_bch_fix_errors_8( in, len, z, r );
		else
			r = 12;
		break;
	case BCH_S12:
		bch_s_12_syndrome( in, s, len);
		// There are errors
		bch_mas( s, t, 12 );
		r = bch_chien_search( t, z, 12 );
		if(r != 0)
		    host_bch_fix_errors_8( in, len, z, r);
		else
			r = 12;
		break;
	case BCH_M12:
		// Use same as N
	    bch_n_12_syndrome( in, s, len);
		// There are errors
		bch_mas( s, t, 12 );
		r = bch_chien_search( t, z, 12 );
		if(r != 0)
		    host_bch_fix_errors_8( in, len, z, r );
		else
			r = 12;
		break;
	default:
		printf("Unknown BCH decoder requested %d\n",g_format.bch);
		return 0;
	}
	return r;
}
////////////////////////////////////////////////////////////////////////////
//
// Device kernels
//
////////////////////////////////////////////////////////////////////////////

static __constant__ BCHLookup m_bch_lookup[256];// BCH Lookup table

void bch_d_copy_lookup(BCHLookup *lu){
	cudaMemcpyToSymbol(m_bch_lookup, lu, sizeof(BCHLookup)*256);
}

__global__ void bch_d_byte_n_8_parity_check( uint8_t *in, int len, uint8_t *out ){

	int p = blockIdx.x;

    uint64_t shift[2];
    uint8_t b;
    uint8_t *ip = &in[p*len];// Point to start of frame for this thread
    //Zero the shift register 128 bits

    shift[0] = 0;
    shift[1] = 0;

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^ip[i];
    	// reg_64_2_shift( shift );
        shift[0] = (shift[0]<<8) | (shift[1]>>56);
        shift[1] = (shift[1]<<8);

        shift[0] ^= m_bch_lookup[b].bch_r[0];
        shift[1] ^= m_bch_lookup[b].bch_r[1];
    }
    if((shift[0] == 0)&&(shift[1] == 0 ))
    	out[p] = 0;
    else
    	out[p] = 1;
}

__global__ void bch_h_byte_n_10_parity_check( uint8_t *in, int len, uint8_t *out ){
	int p = blockIdx.x;

	uint64_t shift[3];
    uint8_t b;
    uint8_t *ip = &in[p*len];// Point to start of frame for this thread
    //Zero the shift register 160 bits
    shift[0] = shift[1] = shift[2] = 0;

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^ip[i];
    	//reg_64_3_shift( shift );
        shift[0] = (shift[0]<<8) | (shift[1]>>56);
        shift[1] = (shift[1]<<8) | (shift[2]>>56);
        shift[2] = (shift[2]<<8);

    	shift[0] ^= m_bch_lookup[b].bch_r[0];
        shift[1] ^= m_bch_lookup[b].bch_r[1];
        shift[2] ^= m_bch_lookup[b].bch_r[2];
    }
//    printf("N10 %lx %lx %lx\n",shift[0],shift[1],shift[2]);
    if((shift[0] == 0 )&&(shift[1] == 0 )&&(shift[2] == 0 ))
    	out[p] = 0;
    else
    	out[p] = 1;
}

__global__ void bch_h_byte_n_12_parity_check( uint8_t *in, int len, uint8_t *out){
	int p = blockIdx.x;

	uint64_t shift[3];
    uint8_t b;
    uint8_t *ip = &in[p*len];// Point to start of frame for this thread
    //Zero the shift register 192 bits
    shift[0] = shift[1] = shift[2] = 0;

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^ip[i];
    	//reg_64_3_shift( shift );
        shift[0] = (shift[0]<<8) | (shift[1]>>56);
        shift[1] = (shift[1]<<8) | (shift[2]>>56);
        shift[2] = (shift[2]<<8);

        shift[0] ^= m_bch_lookup[b].bch_r[0];
        shift[1] ^= m_bch_lookup[b].bch_r[1];
        shift[2] ^= m_bch_lookup[b].bch_r[2];
    }
    if((shift[0] == 0 )&&(shift[1] == 0 )&&(shift[2] == 0 ))
    	out[p] = 0;
    else
    	out[p] = 1;
}

__global__ void bch_h_byte_s_12_parity_check( uint8_t *in, int len, uint8_t *out ){
	int p = blockIdx.x;

	uint64_t shift[3];
    uint8_t b;
    uint8_t *ip = &in[p*len];// Point to start of frame for this thread
    //Zero the shift register 168 bits
    shift[0] = shift[1] = shift[2] = 0;

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^ip[i];
    	//reg_64_3_shift( shift );
        shift[0] = (shift[0]<<8) | (shift[1]>>56);
        shift[1] = (shift[1]<<8) | (shift[2]>>56);
        shift[2] = (shift[2]<<8);

        shift[0] ^= m_bch_lookup[b].bch_r[0];
        shift[1] ^= m_bch_lookup[b].bch_r[1];
        shift[2] ^= m_bch_lookup[b].bch_r[2];
    }
//    printf("S12 %lx %lx %lx\n",shift[0],shift[1],shift[2]);
    if((shift[0] == 0 )&&(shift[1] == 0 )&&(shift[2] == 0 ))
    	out[p] = 0;
    else
    	out[p] = 1;
}

__global__ void bch_h_byte_m_12_parity_check( uint8_t *in, int len,  uint8_t *out ){
	int p = blockIdx.x;

	uint64_t shift[3];
    uint8_t b;
    uint8_t *ip = &in[p*len];// Point to start of frame for this thread
    //Zero the shift register 180 bits
    shift[0] = shift[1] = shift[2] = 0;

    for( int i = 0; i < len; i++){
    	b = ((uint8_t)(shift[0]>>56))^ip[i];
    	//reg_64_3_shift( shift );
        shift[0] = (shift[0]<<8) | (shift[1]>>56);
        shift[1] = (shift[1]<<8) | (shift[2]>>56);
        shift[2] = (shift[2]<<8);

        shift[0] ^= m_bch_lookup[b].bch_r[0];
        shift[1] ^= m_bch_lookup[b].bch_r[1];
        shift[2] ^= m_bch_lookup[b].bch_r[2];
    }
    if((shift[0] == 0 )&&(shift[1] == 0 )&&(shift[2] == 0 ))
    	out[p] = 0;
    else
    	out[p] = 1;
}
//
// Called to error check 128 BCH frames
//
void bch_device_nframe_check( uint8_t *din, uint8_t *dout ){
	switch (g_format.pbch)
	{
	case 128:
		bch_d_byte_n_8_parity_check<<<NP_FRAMES,1>>>(  din, g_format.nbch/8, dout );
		break;
	case 160:
		bch_h_byte_n_10_parity_check<<<NP_FRAMES,1>>>( din, g_format.nbch/8, dout );
		break;
	case 192:
		bch_h_byte_n_12_parity_check<<<NP_FRAMES,1>>>( din, g_format.nbch/8, dout );
		break;
	case 168:
		bch_h_byte_s_12_parity_check<<<NP_FRAMES,1>>>( din, g_format.nbch/8, dout );
		break;
	case 180:
		// Use same as N
		bch_h_byte_m_12_parity_check<<<NP_FRAMES,1>>>( din, g_format.nbch/8, dout );
		break;
	default:
		// Unknown
		bch_h_byte_n_12_parity_check<<<NP_FRAMES,1>>>( din, g_format.nbch/8, dout );
		break;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////
//
// device decoder
//
////////////////////////////////////////////////////////////////////////////////////////////////

//
// Assumes device data input is packed into 1 bit.
// Outputs corrected frames in 8 bit format
// returns number of total BCH errors
//
int (*bch_device_decode)(Bit *din, uint8_t *out);

int bch_device_decode128(Bit *din, uint8_t *out)
{
	uint16_t hmem[10*NP_FRAMES];
	uint16_t ht[30];
	uint16_t hz[30];

	int r = 0;
    int esum = 0;
	memset(ht, 0, 30 * sizeof(uint16_t));
	// Calculate the required lengths and start positions
	int k_octets = g_format.kbch/8;
	int n_octets = g_format.nbch/8;

	compactto_8<<<g_format.nbch*8,16>>>( din );
	bchNSyndrome2<<<NP_FRAMES,8>>>( din, n_octets, ds, d_logn, d_alogn );
	CHECK(cudaMemcpy( hmem, ds, sizeof(uint16_t)*8*NP_FRAMES, cudaMemcpyDeviceToHost));
	// Process each of the 128 frames
    for( int i = 0; i < NP_FRAMES; i++){
		int d_start = i*g_format.nbch/8;
	    int h_start = i*g_format.kbch/8;
    	bool errors = false;
    	int m = i*8;
    	// See if there are errors in the frame
    	for( int k = 0; k < 8; k++ ){
    		if(hmem[m+k] != 0 ){
    			errors = true;
    			break;
    		}
    	}

        if(errors == true){
		    // There are errors
    		// Calculate all the required syndromes
    		uint16_t s[17];
    		s[0]  = 0;
    		s[1]  = hmem[m];
    		s[3]  = hmem[m+1];
    		s[5]  = hmem[m+2];
    		s[7]  = hmem[m+3];
    		s[9]  = hmem[m+4];
    		s[11] = hmem[m+5];
    		s[13] = hmem[m+6];
    		s[15] = hmem[m+7];
    		s[2]  = gmult(s[1],  s[1]);
    		s[4]  = gmult(s[2],  s[2]);
    		s[6]  = gmult(s[3],  s[3]);
    		s[8]  = gmult(s[4],  s[4]);
    		s[10] = gmult(s[5],  s[5]);
    		s[12] = gmult(s[6],  s[6]);
    		s[14] = gmult(s[7],  s[7]);
    		s[16] = gmult(s[8],  s[8]);

		    bch_mas(s, ht, 8);
		    r = bch_chien_search( ht, hz, 8 );

		    // Now correct the errors
		    CHECK(cudaMemcpy( dz, hz, r * sizeof(uint16_t), cudaMemcpyHostToDevice));
		    if( r != 0)
			    bchFixErrors8<<<1,r>>>(&din[d_start], n_octets*8, dz);
		    else
			    r = 12;

		    esum += r;
    	}
    	// Copy out the results
    	CHECK(cudaMemcpy( &out[h_start], &din[d_start], sizeof(uint8_t)*k_octets, cudaMemcpyDeviceToHost));
    }
	return esum;
}

int bch_device_decode160(Bit *din, uint8_t *out)
{
	uint16_t hmem[10*NP_FRAMES];
	uint16_t ht[30];
	uint16_t hz[30];

	int r = 0;
    int esum = 0;
	memset(ht, 0, 30 * sizeof(uint16_t));

	// Calculate the required lengths and start positions
	int k_octets = g_format.kbch/8;
	int n_octets = g_format.nbch/8;

	compactto_8<<<g_format.nbch*8,16>>>( din );
	bchNSyndrome2<<<NP_FRAMES,10>>>( din, n_octets, ds, d_logn, d_alogn );
	CHECK(cudaMemcpy( hmem, ds, sizeof(uint16_t)*10*NP_FRAMES, cudaMemcpyDeviceToHost));
	// Process each of the 128 frames
    for( int i = 0; i < NP_FRAMES; i++){
    	int d_start = i*n_octets;
        int h_start = i*k_octets;
    	bool errors = false;
    	int m = i*10;
    	// See if there are errors in the frame
    	for( int k = 0; k < 10; k++ ){
    		if(hmem[m+k] != 0 ){
    			errors = true;
    			break;
    		}
    	}

        if(errors == true){
		    // There are errors
    		// Calculate all the required syndromes
    		uint16_t s[21];
    		s[0]  = 0;
    		s[1]  = hmem[m];
    		s[3]  = hmem[m+1];
    		s[5]  = hmem[m+2];
    		s[7]  = hmem[m+3];
    		s[9]  = hmem[m+4];
    		s[11] = hmem[m+5];
    		s[13] = hmem[m+6];
    		s[15] = hmem[m+7];
    		s[17] = hmem[m+8];
    		s[19] = hmem[m+9];
    		s[2]  = gmult(s[1],  s[1]);
    		s[4]  = gmult(s[2],  s[2]);
    		s[6]  = gmult(s[3],  s[3]);
    		s[8]  = gmult(s[4],  s[4]);
    		s[10] = gmult(s[5],  s[5]);
    		s[12] = gmult(s[6],  s[6]);
    		s[14] = gmult(s[7],  s[7]);
    		s[16] = gmult(s[8],  s[8]);
    		s[18] = gmult(s[9],  s[9]);
    		s[20] = gmult(s[10], s[10]);

		    bch_mas( s, ht, 10);
		    r = bch_chien_search( ht, hz, 10 );
		    // Now correct the errors
		    CHECK(cudaMemcpy( dz, hz, r * sizeof(uint16_t), cudaMemcpyHostToDevice));
		    if( r != 0)
			    bchFixErrors8<<<1,r>>>( &din[d_start], n_octets*8, dz);
		    else
			    r = 12;

		    esum += r;
    	}
    	// Copy out the results
    	CHECK(cudaMemcpy(&out[h_start], &din[d_start], sizeof(uint8_t)*k_octets, cudaMemcpyDeviceToHost));
    }
	return esum;
}
int bch_device_decode192_180(Bit *din, uint8_t *out)
{
	uint16_t hmem[12*NP_FRAMES];
	uint16_t ht[30];
	uint16_t hz[30];

	int r = 0;
    int esum = 0;
	memset(ht, 0, 30 * sizeof(uint16_t));

	// Calculate the required lengths and start positions
	int k_octets = g_format.kbch/8;
	int n_octets = g_format.nbch/8;

	compactto_8<<<g_format.nbch*8,16>>>( din );
	bchNSyndrome2<<<NP_FRAMES,12>>>( din, n_octets, ds, d_logn, d_alogn );
	CHECK(cudaMemcpy( hmem, ds, sizeof(uint16_t)*12*NP_FRAMES, cudaMemcpyDeviceToHost));

	// Process each of the 128 frames
    for( int i = 0; i < NP_FRAMES; i++){
    	int d_start = i*g_format.nbch/8;
        int h_start = i*g_format.kbch/8;
    	bool errors = false;
    	int m = i*12;
    	// See if there are errors in the frame
    	for( int k = 0; k < 12; k++ ){
    		if(hmem[m+k] != 0 ){
    			errors = true;
    			break;
    		}
    	}

        if(errors == true){
		    // There are errors
    		// Calculate all the required syndromes
    		uint16_t s[25];
    		s[0]  = 0;
    		s[1]  = hmem[m];
    		s[3]  = hmem[m+1];
    		s[5]  = hmem[m+2];
    		s[7]  = hmem[m+3];
    		s[9]  = hmem[m+4];
    		s[11] = hmem[m+5];
    		s[13] = hmem[m+6];
    		s[15] = hmem[m+7];
    		s[17] = hmem[m+8];
    		s[19] = hmem[m+9];
    		s[21] = hmem[m+10];
    		s[23] = hmem[m+11];
    		s[2]  = gmult(s[1],  s[1]);
    		s[4]  = gmult(s[2],  s[2]);
    		s[6]  = gmult(s[3],  s[3]);
    		s[8]  = gmult(s[4],  s[4]);
    		s[10] = gmult(s[5],  s[5]);
    		s[12] = gmult(s[6],  s[6]);
    		s[14] = gmult(s[7],  s[7]);
    		s[16] = gmult(s[8],  s[8]);
    		s[18] = gmult(s[9],  s[9]);
    		s[20] = gmult(s[10], s[10]);
    		s[22] = gmult(s[11], s[11]);
    		s[24] = gmult(s[12], s[12]);

		    bch_mas(s, ht, 12);
		    r = bch_chien_search( ht, hz, 12 );
		    // Now correct the errors
		    CHECK(cudaMemcpy( dz, hz, r * sizeof(uint16_t), cudaMemcpyHostToDevice));
		    if( r != 0)
			    bchFixErrors8<<<1,r>>>(&din[d_start], n_octets*8, dz);
		    else
			    r = 12;

		    esum += r;
    	}
    	// Copy out the results
    	CHECK(cudaMemcpy(&out[h_start], &din[d_start], sizeof(uint8_t)*k_octets, cudaMemcpyDeviceToHost));
    }
	return esum;
}
//
// Short frame
//
int bch_device_decode168(Bit *din, uint8_t *out)
{
	uint16_t hmem[12*NP_FRAMES];
	uint16_t ht[30];
	uint16_t hz[30];

	int r = 0;
    int esum = 0;
	memset(ht, 0, 30 * sizeof(uint16_t));

	// Calculate the required lengths and start positions
	int k_octets = g_format.kbch/8;
	int n_octets = g_format.nbch/8;

	compactto_8<<<g_format.nbch*8,16>>>( din );
	bchNSyndrome2<<<NP_FRAMES,12>>>( din, n_octets, ds, d_logs, d_alogs );
	CHECK(cudaMemcpy( hmem, ds, sizeof(uint16_t)*12*NP_FRAMES, cudaMemcpyDeviceToHost));

	// Process each of the 128 frames
    for( int i = 0; i < NP_FRAMES; i++){
    	int d_start = i*g_format.nbch/8;
        int h_start = i*g_format.kbch/8;
    	bool errors = false;
    	int m = i*12;
    	// See if there are errors in the frame
    	for( int k = 0; k < 12; k++ ){
    		if(hmem[m+k] != 0 ){
    			errors = true;
    			break;
    		}
    	}

        if(errors == true){
		    // There are errors
    		// Calculate all the required syndromes
    		uint16_t s[25];
    		s[0]  = 0;
    		s[1]  = hmem[m];
    		s[3]  = hmem[m+1];
    		s[5]  = hmem[m+2];
    		s[7]  = hmem[m+3];
    		s[9]  = hmem[m+4];
    		s[11] = hmem[m+5];
    		s[13] = hmem[m+6];
    		s[15] = hmem[m+7];
    		s[17] = hmem[m+8];
    		s[19] = hmem[m+9];
    		s[21] = hmem[m+10];
    		s[23] = hmem[m+11];
    		s[2]  = gmult(s[1],  s[1]);
    		s[4]  = gmult(s[2],  s[2]);
    		s[6]  = gmult(s[3],  s[3]);
    		s[8]  = gmult(s[4],  s[4]);
    		s[10] = gmult(s[5],  s[5]);
    		s[12] = gmult(s[6],  s[6]);
    		s[14] = gmult(s[7],  s[7]);
    		s[16] = gmult(s[8],  s[8]);
    		s[18] = gmult(s[9],  s[9]);
    		s[20] = gmult(s[10], s[10]);
    		s[22] = gmult(s[11], s[11]);
    		s[24] = gmult(s[12], s[12]);

		    bch_mas(s, ht, 12);
		    r = bch_chien_search( ht, hz, 12 );
		    // Now correct the errors
		    CHECK(cudaMemcpy( dz, hz, r * sizeof(uint16_t), cudaMemcpyHostToDevice));
		    if( r != 0)
			    bchFixErrors8<<<1,r>>>(&din[d_start], n_octets*8, dz);
		    else
			    r = 12;

		    esum += r;
    	}
    	// Copy out the results
    	CHECK(cudaMemcpy(&out[h_start], &din[d_start], sizeof(uint8_t)*k_octets, cudaMemcpyDeviceToHost));
    }
	return esum;
}
//
// Sets the current BCH decoder for the frames being received
//
void bch_set_device_decode(void){

	switch (g_format.pbch)
	{
		case 128:
			bch_device_decode = bch_device_decode128;
			break;
		case 160:
			bch_device_decode = bch_device_decode160;
			break;
		case 192:
			bch_device_decode = bch_device_decode192_180;
			break;
		case 168:
			bch_device_decode = bch_device_decode168;
			break;
		case 180:
			bch_device_decode = bch_device_decode192_180;
			break;
		default:
			bch_device_decode = bch_device_decode192_180;
			break;
	}
}

//
// BCH decoder open and close routines
//

void bch_decode_open(void){

	build_hmatrix();

#ifdef BUILD_BCH_TABLES

	bch_build_poly0n_inv_tables();
	bch_build_poly0s_inv_tables();
	bch_build_poly0m_inv_tables();
	bch_save_poly_inversion_tables();

#endif

	ds      = NULL;
	dz      = NULL;
	d_logn  = NULL;
	d_alogn = NULL;
	d_logs  = NULL;
	d_alogs = NULL;
	d_logm  = NULL;
	d_alogm = NULL;

	// Memory used for syndromes and zeros
	CHECK(cudaMalloc((void**)&ds, 12 * sizeof(uint16_t) * NP_FRAMES));
	CHECK(cudaMalloc((void**)&dz, 30 * sizeof(uint16_t)));
	CHECK(cudaMalloc((void**)&d_logs,  0x4000 * sizeof(uint16_t)));
	CHECK(cudaMalloc((void**)&d_alogs, 0x4020 * sizeof(uint16_t)));
	CHECK(cudaMalloc((void**)&d_logm,  0x8000 * sizeof(uint16_t)));
	CHECK(cudaMalloc((void**)&d_alogm, 0x8020 * sizeof(uint16_t)));
	CHECK(cudaMalloc((void**)&d_logn,  0x10000 * sizeof(uint16_t)));
	CHECK(cudaMalloc((void**)&d_alogn, 0x10020 * sizeof(uint16_t)));
	// Build the log and anti log tables
	bch_alog_log_build_table();

	// build the fast BCH decoder check tables
	bch_open_b();
}

void bch_decode_close(void){
	if(dz != NULL) cudaFree(dz);
	if(dz != NULL) cudaFree(ds);
	if(d_logn != NULL) cudaFree(d_logn);
	if(d_alogn != NULL) cudaFree(d_alogn);
	if(d_logs != NULL) cudaFree(d_logs);
	if(d_alogs != NULL) cudaFree(d_alogs);
	if(d_logm != NULL) cudaFree(d_logm);
	if(d_alogm != NULL) cudaFree(d_alogm);
}
