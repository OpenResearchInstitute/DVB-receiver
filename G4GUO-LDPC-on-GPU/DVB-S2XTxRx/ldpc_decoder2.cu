#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <semaphore.h>
#include <pthread.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dvbs2_rx.h"

extern RxFormat g_format;

typedef struct{
	size_t   n_edges;
	uint16_t tbits;
	uint32_t *map;
	uint32_t *hvn;
	uint32_t *hvn_starts;
	uint16_t hvn_nstarts;
	uint32_t *hcn;
	uint32_t *hcn_starts;
	uint16_t hcn_nstarts;
	uint16_t frame;
	uint16_t code;
	int8_t   *msg;
}TableDB;

static TableDB m_device_tab;
static Bits    *m_d_bits   = NULL;
static uint8_t *m_d_bytes  = NULL;
static uint8_t *m_d_checks = NULL;
static uint8_t *m_h_bytes  = NULL;
static uint8_t *m_h_checks = NULL;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GPU Kernels start here
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Each kernel processes 4 nodes at a time and there are 32 threads in a block (128 in total)
//
__global__ void initial_update( const uint32_t *llr, const uint32_t *start, const uint32_t *hvn, uint32_t *msg ){
	uint32_t c  = blockIdx.x * blockDim.x;
	uint32_t ki = threadIdx.x;
    uint32_t m;
    uint32_t n;
    uint32_t i = start[blockIdx.x];
    n = i;
    // Sum the channel metric with the metrics of all the connected checknodes
    m  = llr[c+ki];
    do{
    	msg[(n*blockDim.x)+ki] = m;
    }while((n = hvn[n]) != i);
//    __syncthreads();
}

__global__ void bitnode_update( const uint32_t * __restrict__ llr, const uint32_t * __restrict__ start, const uint32_t * __restrict__ hvn, uint32_t *msg ){
	uint32_t c  = blockIdx.x * blockDim.x;
	uint32_t ki = threadIdx.x;
    uint32_t m;
    uint32_t n;
    uint32_t i = start[blockIdx.x];
    n = i;
    // Sum the channel metric with the metrics of all the connected checknodes
    m  = llr[c+ki];
    do{
    	m =  __vaddss4(m, msg[(n*blockDim.x)+ki]);// m += msg[n];
    }while((n = hvn[n]) != i);

    // subtract the msg from that connected node from the accumulated metric
    // Save the result.

    n = i;
    do{
    	msg[(n*blockDim.x)+ki] = __vsubss4 (m,msg[(n*blockDim.x)+ki]);// msg[n] = UCLAMP(m - msg[n]);

    }while((n=hvn[n]) != i);
    // All done
//    __syncthreads();
}
__global__ void checknode_update( const uint32_t * __restrict__ start, const uint32_t * __restrict__ hcn, uint32_t *msg ){
	uint32_t tm,nmin,min,absm;
	uint32_t sign;
	uint32_t j,i,n;
    uint8_t  cnt = 0;
    uint8_t  mi[4];

    // We need byte pointers when we have no suitable SIMDs
    uint8_t *pabsm = (uint8_t*)&absm;
    uint8_t *pmin  = (uint8_t*)&min;
    uint8_t *pnmin = (uint8_t*)&nmin;
    uint8_t *psign = (uint8_t*)&sign;

    uint32_t r  = blockIdx.x;
	uint32_t ki = threadIdx.x;

	i = n = start[r];
    // Find the minimum

    min  = 0x7F7F7F7F;// highest value (127)
    nmin = 0x7F7F7F7F;
    sign = 0; // sign positive

    // Find the 2 minimums
    j = 0;
    do{
    	tm    = msg[(n*blockDim.x)+ki];// load 4 messages
    	absm  = __vabs4 (tm);// find their absolute values;
    	for( uint8_t x = 0; x < 4; x++){ // process byte by byte
     	    if(pabsm[x] < pmin[x]){
     	    	    pnmin[x] = pmin[x];
    		        pmin[x] = pabsm[x];
    		        mi[x]   = j;
    	    }else{
    		    if(pabsm[x] < pnmin[x]) pnmin[x] = pabsm[x];
    	    }
        }
     	sign ^= tm;// sign*msg, use msb to determine final sign
    	j = j+1;
    	cnt++;
    }while((n = hcn[n]) != i);

    if(cnt&1) sign ^= 0x80808080;// If odd change the sign
    //
    // Update, the metrics
    //
    n = i;
    j = 0;

    do{
    	int8_t  *pmsg = (int8_t*)&msg[(n*blockDim.x)+ki];// process byte by byte

    	for(uint8_t x = 0; x < 4; x++){
            if( mi[x] != j){
    	        pmsg[x] = ((psign[x]^pmsg[x])&0x80) ? -pmin[x]  : pmin[x];
            }
            else{
    	        pmsg[x] = ((psign[x]^pmsg[x])&0x80) ? -pnmin[x] : pnmin[x];
            }
    	}
    	j = j+1;
    }while((n = hcn[n]) != i);
//    __syncthreads();
}
__global__ void final_update( const uint32_t * __restrict__ llr, const uint32_t * __restrict__ start, const uint32_t * __restrict__ hvn, const uint32_t * __restrict__ msg, Bits *b ){

	uint32_t n0 = (((threadIdx.x*4)+0)*gridDim.x) + blockIdx.x;
	uint32_t n1 = (((threadIdx.x*4)+1)*gridDim.x) + blockIdx.x;
	uint32_t n2 = (((threadIdx.x*4)+2)*gridDim.x) + blockIdx.x;
	uint32_t n3 = (((threadIdx.x*4)+3)*gridDim.x) + blockIdx.x;

    uint32_t m;
	uint32_t c  = blockIdx.x * blockDim.x;
    uint32_t i  = start[blockIdx.x];
    uint32_t n  = i;
	uint32_t ki  = threadIdx.x;

    // Sum the channel metric with the metrics of all the connected checknodes
    m  = llr[c+ki];
    do{
    	m =  __vaddss4(m, msg[(n*blockDim.x)+ki]);// m += msg[n];
    }while((n=hvn[n]) != i);

    // Output the bits to the correct position

    int8_t *bm = (int8_t*)&m;

    b[n0] = bm[0] > 0 ? 1 : 0 ;
    b[n1] = bm[1] > 0 ? 1 : 0 ;
    b[n2] = bm[2] > 0 ? 1 : 0 ;
    b[n3] = bm[3] > 0 ? 1 : 0 ;
    // All done
//    __syncthreads();
}

//
// Pack the bit array into a byte array return inplace
//
__global__ void compactto8( const Bit * __restrict__ bi, uint8_t *bo, uint32_t nbits ){
    uint8_t b;
    uint32_t m = blockIdx.x * nbits;
    uint32_t n = blockIdx.x * nbits/8;
    uint32_t end = nbits/8;

    for( int i = 0; i < end; i++ ){
        b  = (bi[m]<<7)|(bi[m+1]<<6)|(bi[m+2]<<5)|(bi[m+3]<<4)|(bi[m+4]<<3)|(bi[m+5]<<2)|(bi[m+6]<<1)|(bi[m+7]);
        bo[n] = b;
        m    += 8;
        n++;
   }
    __syncthreads();
}

//////////////////////////////////////////////////////////////////////////////////////////////
//
// Macros used to build the tables
//
//////////////////////////////////////////////////////////////////////////////////////////////

#define LDPC_RT( TABLE_NAME, ROWS ) \
	bn = 0; \
    for( row = 0; row < ROWS; row++ ){ \
	    for( n = 0; n < 360; n++ ){ \
		    for( col = 1; col <= TABLE_NAME[row][0]; col++ ){ \
			    cn = ((TABLE_NAME[row][col] + (n*q)) % pbits); \
			    (*H)[cn][bn] = 1; \
		    } \
		    bn++; \
	    } \
    }


//
// Typedefs for the tables being used
//
typedef struct{
	uint16_t col;
	uint16_t row;
	uint16_t m;
	uint16_t n;
}TEntry;
typedef struct{
	uint32_t next;
	uint16_t m;
	uint16_t n;
}TLEntry;

typedef uint8_t tabH[64800][64800];

// Mutex
static pthread_mutex_t mutex_tab;

#define MAX_TABLES 55
static int m_table_index;
static TableDB m_tab_db[MAX_TABLES];

void save_table( TableDB *entry){

	pthread_mutex_lock( &mutex_tab );
    if(m_table_index < MAX_TABLES ){
    	m_tab_db[m_table_index].n_edges     = entry->n_edges;
    	m_tab_db[m_table_index].tbits       = entry->tbits;
    	m_tab_db[m_table_index].map         = entry->map;

    	m_tab_db[m_table_index].hvn         = entry->hvn;
    	m_tab_db[m_table_index].hvn_starts  = entry->hvn_starts;
    	m_tab_db[m_table_index].hvn_nstarts = entry->hvn_nstarts;

    	m_tab_db[m_table_index].hcn         = entry->hcn;
    	m_tab_db[m_table_index].hcn_starts  = entry->hcn_starts;
    	m_tab_db[m_table_index].hcn_nstarts = entry->hcn_nstarts;

    	m_tab_db[m_table_index].frame       = entry->frame;
    	m_tab_db[m_table_index].code        = entry->code;
	    m_table_index++;
    }
    pthread_mutex_unlock( &mutex_tab );
}
void load_table(uint16_t frame, uint16_t code){

	for( int i = 0; i < MAX_TABLES; i++){
		if((m_tab_db[i].frame == frame)&&(m_tab_db[i].code == code)){

			// Matching entry found
			m_device_tab.n_edges     = m_tab_db[i].n_edges;
			m_device_tab.hvn_nstarts = m_tab_db[i].hvn_nstarts;
			m_device_tab.hcn_nstarts = m_tab_db[i].hcn_nstarts;
			m_device_tab.frame       = m_tab_db[i].frame;
			m_device_tab.code        = m_tab_db[i].code;

			if(m_device_tab.hvn != NULL) cudaFree(m_device_tab.hvn);
			cudaMalloc((void**)&m_device_tab.hvn,sizeof(uint32_t)*m_tab_db[i].n_edges);
			cudaMemcpy(m_device_tab.hvn, m_tab_db[i].hvn, sizeof(uint32_t)*m_tab_db[i].n_edges, cudaMemcpyHostToDevice);

			if(m_device_tab.hvn_starts != NULL) cudaFree(m_device_tab.hvn_starts);
			cudaMalloc((void**)&m_device_tab.hvn_starts,sizeof(uint32_t)*m_tab_db[i].hvn_nstarts);
			cudaMemcpy(m_device_tab.hvn_starts, m_tab_db[i].hvn_starts, sizeof(uint32_t)*m_tab_db[i].hvn_nstarts, cudaMemcpyHostToDevice);

			if(m_device_tab.hcn != NULL) cudaFree(m_device_tab.hcn);
			cudaMalloc((void**)&m_device_tab.hcn,sizeof(uint32_t)*m_tab_db[i].n_edges);
			cudaMemcpy(m_device_tab.hcn, m_tab_db[i].hcn, sizeof(uint32_t)*m_tab_db[i].n_edges, cudaMemcpyHostToDevice);

			if(m_device_tab.hcn_starts != NULL) cudaFree(m_device_tab.hcn_starts);
			cudaMalloc((void**)&m_device_tab.hcn_starts,sizeof(uint32_t)*m_tab_db[i].hcn_nstarts);
			cudaMemcpy(m_device_tab.hcn_starts, m_tab_db[i].hcn_starts, sizeof(uint32_t)*m_tab_db[i].hcn_nstarts, cudaMemcpyHostToDevice);

			if(m_device_tab.map != NULL) cudaFree(m_device_tab.map);
			cudaMalloc((void**)&m_device_tab.map,sizeof(uint32_t)*m_tab_db[i].n_edges);
			cudaMemcpy(m_device_tab.map, m_tab_db[i].map, sizeof(uint32_t)*m_tab_db[i].n_edges, cudaMemcpyHostToDevice);

			if(m_device_tab.msg != NULL) cudaFree(m_device_tab.msg);
			cudaMalloc((void**)&m_device_tab.msg,sizeof(int8_t)*m_tab_db[i].n_edges*NP_FRAMES);
			return;
		}
	}
}
size_t save_table_to_disk( const char *name, TableDB *table){
	FILE *fp;
    size_t bytes = 0;
	if((fp=fopen(name,"wb"))!=NULL){
		bytes += fwrite(&table->n_edges,sizeof(size_t),1,fp)*sizeof(size_t);
		bytes += fwrite(&table->tbits,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fwrite(&table->hvn_nstarts,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fwrite(&table->hcn_nstarts,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fwrite(&table->frame,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fwrite(&table->code,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fwrite(table->hvn,sizeof(uint32_t),table->n_edges,fp)*sizeof(uint32_t);
		bytes += fwrite(table->hcn,sizeof(uint32_t),table->n_edges,fp)*sizeof(uint32_t);
		bytes += fwrite(table->hvn_starts,sizeof(uint32_t),table->hvn_nstarts,fp)*sizeof(uint32_t);
		bytes += fwrite(table->hcn_starts,sizeof(uint32_t),table->hcn_nstarts,fp)*sizeof(uint32_t);
		bytes += fwrite(table->map,sizeof(uint32_t),table->n_edges,fp)*sizeof(uint32_t);
		fclose(fp);
		return bytes;
	}
	return 0;
}

size_t load_table_from_disk( const char *name, TableDB *table){
	FILE *fp;
    size_t bytes = 0;
	if((fp=fopen(name,"rb"))!=NULL){
		bytes += fread(&table->n_edges,sizeof(size_t),1,fp)*sizeof(size_t);
		bytes += fread(&table->tbits,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fread(&table->hvn_nstarts,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fread(&table->hcn_nstarts,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fread(&table->frame,sizeof(uint16_t),1,fp)*sizeof(uint16_t);
		bytes += fread(&table->code,sizeof(uint16_t),1,fp)*sizeof(uint16_t);

		table->hvn        = (uint32_t*)calloc(table->n_edges,sizeof(uint32_t));
		table->hcn        = (uint32_t*)calloc(table->n_edges,sizeof(uint32_t));
		table->hvn_starts = (uint32_t*)calloc(table->hvn_nstarts,sizeof(uint32_t));
		table->hcn_starts = (uint32_t*)calloc(table->hcn_nstarts,sizeof(uint32_t));
		table->map        = (uint32_t*)calloc(table->n_edges,sizeof(uint32_t));

		bytes += fread(table->hvn,sizeof(uint32_t),table->n_edges,fp)*sizeof(uint32_t);
		bytes += fread(table->hcn,sizeof(uint32_t),table->n_edges,fp)*sizeof(uint32_t);
		bytes += fread(table->hvn_starts,sizeof(uint32_t),table->hvn_nstarts,fp)*sizeof(uint32_t);
		bytes += fread(table->hcn_starts,sizeof(uint32_t),table->hcn_nstarts,fp)*sizeof(uint32_t);
		bytes += fread(table->map,sizeof(uint32_t),table->n_edges,fp)*sizeof(uint32_t);
		fclose(fp);
		return bytes;
	}
	return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Global memory equivalent
//
/////////////////////////////////////////////////////////////////////

TLEntry *find_in_hcn_return_hvn_linear( int m, int n, TLEntry *hcn, TLEntry *hbn, size_t edges, uint32_t &next ){
    for( size_t i = 0; i < edges; i++){
    	if((hcn[i].m == m)&&(hcn[i].n == n)){
    		next = i;
    		return &hbn[i];
    	}
    }
    printf("not found 1\n");
    return NULL;
}

void find_in_hcn_linear( int m, int n, TLEntry *hcn, size_t edges, uint32_t &next){
    for( size_t i = 0; i < edges; i++){
    	if((hcn[i].m == m)&&(hcn[i].n == n)){
    			next = i;
    			return;
    	}
    }
    printf("not found 2\n");
}
//
// Same as above but write to linear memory rather than square array
//
void build_hcn_from_h_linear( tabH *h, int M, int N, TLEntry *hcn, uint32_t *starts, uint16_t &nstarts ){
	TLEntry *ce;
	int en = 0;
	int ef;
    nstarts = 0;

	for( int m = 0; m < M; m++){
		ef = en;// Save id of first entry in this row
		starts[nstarts++] = en;
		for( int n = 0; n < N; n++){
			if((*h)[m][n] == 1){
				// Found non zero entry
				ce      = &hcn[en];
				// Point to the Row and Column in the H matrix that this entry refers to.
				ce->m   = m;
				ce->n   = n;
				// Just in case this is the last
				// Found first non zero entry
				// Find the following entries in the row
				for( int j = 1; j < N+n; j++){
					if((*h)[m][(j+n)%N] == 1 ){
						// There is a following entry in the row
						// Set the current entry to point to the next
						en++;
						ce->next = en;
						// Break out of the loop
						break;
					}
				}
			}
		}
		ce->next = ef;
	}
}

void build_hvn_from_hcn_and_h_linear( tabH *h, int M, int N, TLEntry *hcn, TLEntry *hvn, size_t edges, uint32_t *starts, uint16_t &nstarts ){
	TLEntry *ce;
	TLEntry ne;
	uint32_t next;
	int ecnt;
    nstarts = 0;

	for( int n = 0; n < N; n++){
		ecnt = 0;// count the number of entries
		for( int m = 0; m < M; m++){
			if((*h)[m][n] == 1){
				// Found non zero entry
				// Point to the Row and Column in the H matrix that this entry refers to.
				ce = find_in_hcn_return_hvn_linear( m, n, hcn, hvn, edges, next );
				if(ecnt == 0 ) starts[nstarts++] = next;

				ce->next = next;
				ce->m = m;
                ce->n = n;
				ecnt++;
				// Just in case this is the last
				// Found first non zero entry
				// Find the following entries in the row
				for( int i = 1; i < M+m; i++){
					if((*h)[((i+m)%M)][n] == 1 ){
						// There is a following entry in the column
						ne.n = n;
						ne.m = ((i+m)%M);
						find_in_hcn_linear( ne.m, ne.n, hcn, edges, next);
						// Set the current entry to point to the row/col of the new entry
						ce->next = next;
						// Break out of the loop
						break;
					}
				}
			}
		}
	}
}
//
// Compress into a format more suitable for the decoder
// The arrays are split to make memory caching work better
//
void compress_hvn_linear( uint32_t *hvn, TLEntry *hvn_temp, size_t edges ){
	for( size_t i = 0; i < edges; i++){
		hvn[i] = hvn_temp[i].next;
	}
}
void compress_hcn_linear( uint32_t *hcn, TLEntry *hcn_temp, size_t edges ){
	for( size_t i = 0; i < edges; i++){
		hcn[i] = hcn_temp[i].next;
	}
}
void compress_map_linear( uint32_t *map, TLEntry *hcn_temp, size_t edges ){
	for( size_t i = 0; i < edges; i++){
		map[i] = hcn_temp[i].n;
	}
}

//
// Calculate the number of intersections between check nodes and bitnodes
// in the sparse matrix
//
void calculate_d(tabH *h, size_t M, size_t N, size_t &D, size_t &edges ){

	edges = 0;
	for(int m = 0; m < M; m++ ){
		for(int n = 0; n < N; n++ ){
		    if((*h)[m][n]) edges++;
		}
	}
    double dd = sqrt(edges);
    D = (size_t)ceil(dd);
}
//
// Add the connections between the parity bits and their checknode
//
void ldpc_cn_to_bn(tabH *H, uint64_t dbits, uint64_t pbits){
	uint64_t cn;
	uint64_t bn;
    //
	// Add the parity bits to their checknode
	//
	bn = dbits;

	for (cn = 0; cn < pbits; cn++){
		(*H)[cn][bn] = 1;
   	    bn++;
	}
	//
	// Add the parity bits to their previous checknode due to final XOR
	//
	bn = dbits;
	for ( cn = 1; cn < pbits; cn++){
		(*H)[cn][bn] = 1;
   	    bn++;
	}

}
//
// Mother thread used to build the Edge tables
//
void *ldpc_generate_tables(  void *arg )
{
	uint64_t bn;
	uint64_t cn;
	uint64_t q;
	uint64_t row;
	uint64_t col;
	uint64_t n;

	uint16_t pbits;
    uint16_t dbits;
    uint16_t tbits;
    size_t   D,edges;

    tabH *H = NULL;

    uint32_t *array = (uint32_t*)arg;

    uint16_t frame = (uint16_t)array[0];
    uint16_t code  = (uint16_t)array[1];
    const char *name = NULL;
    H = (tabH*)calloc(64800,64800);

	if (frame == frame_NORMAL)
	{
	    tbits = FRAME_SIZE_NORMAL;// Length in bits of a Normal frame
		// DVB-S2 Formats
		if (code == cR_1_4 ){
		    q     = 135;
		    dbits = 16200;
		    pbits = tbits - dbits;
		    name = "n14.tab";
		    // Working memory
			LDPC_RT(ldpc_tab_1_4N, 45);
		}
		if (code == cR_1_3 ){
			q     = 120;
			dbits = 21600;
		    pbits = tbits - dbits;
		    name = "n13.tab";
			LDPC_RT(ldpc_tab_1_3N, 60);
		}
		if (code == cR_2_5 ){
			q     = 108;
			dbits = 25920;
		    pbits = tbits - dbits;
		    name = "n25.tab";
			LDPC_RT(ldpc_tab_2_5N, 72);
		}
		if (code == cR_1_2 ){
			q     = 90;
			dbits = 32400;
		    pbits = tbits - dbits;
		    name = "n12.tab";
			LDPC_RT(ldpc_tab_1_2N, 90);
		}
		if (code == cR_3_5 ){
			q     = 72;
			dbits = 38880;
		    pbits = tbits - dbits;
		    name = "n35.tab";
			LDPC_RT(ldpc_tab_3_5N, 108);
		}
		if (code == cR_2_3 ){
			q     = 60;
			dbits = 43200;
		    pbits = tbits - dbits;
		    name = "n23.tab";
			LDPC_RT(ldpc_tab_2_3N, 120);
		}
		if (code == cR_3_4 ){
			q     = 45;
			dbits = 48600;
		    pbits = tbits - dbits;
		    name = "n34.tab";
			LDPC_RT(ldpc_tab_3_4N, 135);
		}
		if (code == cR_4_5 ){
			q     = 36;
			dbits = 51840;
		    pbits = tbits - dbits;
		    name = "n45.tab";
			LDPC_RT(ldpc_tab_4_5N, 144);
		}
		if (code == cR_5_6 ){
			q     = 30;
			dbits = 54000;
		    pbits = tbits - dbits;
		    name = "n56.tab";
			LDPC_RT(ldpc_tab_5_6N, 150);
		}
		if (code == cR_8_9 ){
			q     = 20;
			dbits = 57600;
		    pbits = tbits - dbits;
		    name = "n89.tab";
			LDPC_RT(ldpc_tab_8_9N, 160);
		}
		if (code == cR_9_10){
			q     = 18;
			dbits = 58320;
		    pbits = tbits - dbits;
		    name = "n910.tab";
			LDPC_RT(ldpc_tab_9_10N, 162);
		}

		// DVB-S2X formats
		if (code == cR_2_9){
			q     = 140;
			dbits = 14400;
		    pbits = tbits - dbits;
		    name = "n29.tab";
			LDPC_RT(ldpc_tab_2_9NX, 40);
		}
		if (code == cR_13_45){
			q     = 128;
			dbits = 18720;
		    pbits = tbits - dbits;
		    name = "n1345.tab";
			LDPC_RT(ldpc_tab_13_45NX, 52);
		}
		if (code == cR_9_20){
			q     = 99;
			dbits = 29160;
		    pbits = tbits - dbits;
		    name = "n920.tab";
			LDPC_RT(ldpc_tab_9_20NX, 81);
		}
		if (code == cR_11_20){
			q     = 81;
			dbits = 35640;
		    pbits = tbits - dbits;
		    name = "n1120.tab";
			LDPC_RT(ldpc_tab_11_20NX, 99);
		}
		if (code == cR_26_45){
			q     = 76;
			dbits = 37440;
		    pbits = tbits - dbits;
		    name = "n2645.tab";
			LDPC_RT(ldpc_tab_26_45NX, 104);
		}
		if (code == cR_28_45){
			q     = 68;
			dbits = 40320;
		    pbits = tbits - dbits;
		    name = "n2845.tab";
			LDPC_RT(ldpc_tab_28_45NX, 112);
		}
		if (code == cR_23_36){
			q     = 65;
			dbits = 41400;
		    pbits = tbits - dbits;
		    name = "n2336.tab";
			LDPC_RT(ldpc_tab_23_36NX, 115);
		}
		if (code == cR_25_36){
			q     = 55;
			dbits = 45000;
		    pbits = tbits - dbits;
		    name = "n2536.tab";
			LDPC_RT(ldpc_tab_25_36NX, 125);
		}
		if (code == cR_13_18){
			q     = 50;
			dbits = 46800;
		    pbits = tbits - dbits;
		    name = "n1318.tab";
			LDPC_RT(ldpc_tab_13_18NX, 130);
		}
		if (code == cR_7_9){
			q     = 40;
			dbits = 50400;
		    pbits = tbits - dbits;
		    name = "n79.tab";
			LDPC_RT(ldpc_tab_7_9NX, 140);
		}
		if (code == cR_90_180){
			q     = 90;
			dbits = 32400;
		    pbits = tbits - dbits;
		    name = "n90180.tab";
			LDPC_RT(ldpc_tab_90_180NX, 90);
		}
		if (code == cR_96_180){
			q     = 84;
			dbits = 34560;
		    pbits = tbits - dbits;
		    name = "n96180.tab";
			LDPC_RT(ldpc_tab_96_180NX, 96);
		}
		if (code == cR_100_180){
			q     = 80;
			dbits = 36000;
		    pbits = tbits - dbits;
		    name = "n100180.tab";
			LDPC_RT(ldpc_tab_100_180NX, 100);
		}
		if (code == cR_104_180){
			q     = 76;
			dbits = 37440;
		    pbits = tbits - dbits;
		    name = "n104180.tab";
			LDPC_RT(ldpc_tab_104_180NX, 104);
		}
		if (code == cR_116_180){
			q     = 64;
			dbits = 41760;
		    pbits = tbits - dbits;
		    name = "n116180.tab";
			LDPC_RT(ldpc_tab_116_180NX, 116);
		}
		if (code == cR_124_180){
			q     = 56;
			dbits = 44640;
		    pbits = tbits - dbits;
		    name = "n124180.tab";
			LDPC_RT(ldpc_tab_124_180NX, 124);
		}
		if (code == cR_128_180){
			q     = 52;
			dbits = 46080;
		    pbits = tbits - dbits;
		    name = "n128180.tab";
			LDPC_RT(ldpc_tab_128_180NX, 128);
		}
		if (code == cR_132_180){
			q     = 48;
			dbits = 47520;
		    pbits = tbits - dbits;
		    name = "n132180.tab";
			LDPC_RT(ldpc_tab_132_180NX, 132);
		}
		if (code == cR_135_180){
			q     = 45;
			dbits = 48600;
		    pbits = tbits - dbits;
		    name = "n135180.tab";
			LDPC_RT(ldpc_tab_135_180NX, 135);
		}
		if (code == cR_140_180){
			q     = 40;
			dbits = 50400;
		    pbits = tbits - dbits;
		    name = "n140180.tab";
			LDPC_RT(ldpc_tab_140_180NX, 140);
		}
		if (code == cR_154_180){
			q     = 26;
			dbits = 55440;
		    pbits = tbits - dbits;
		    name = "n154180.tab";
			LDPC_RT(ldpc_tab_154_180NX, 154);
		}
		if (code == cR_18_30){
			q     = 72;
			dbits = 38880;
		    pbits = tbits - dbits;
		    name = "n1830.tab";
			LDPC_RT(ldpc_tab_18_30NX, 108);
		}
		if (code == cR_20_30){
			q     = 60;
			dbits = 43200;
		    pbits = tbits - dbits;
		    name = "n2030.tab";
			LDPC_RT(ldpc_tab_20_30NX, 120);
		}
		if (code == cR_22_30){
			q     = 48;
			dbits = 47520;
		    pbits = tbits - dbits;
		    name = "n2230.tab";
			LDPC_RT(ldpc_tab_22_30NX, 132);
		}
	}

	if (frame == frame_MEDIUM)
	{
		// DVB-S2X formats
		tbits = FRAME_SIZE_MEDIUM;
		if (code == cR_1_5){
			q     = 72;
			dbits = 6480;
		    pbits = tbits - dbits;
		    name = "m15.tab";
			LDPC_RT(ldpc_tab_1_5MX, 18);
		}
		if (code == cR_11_45){
			q     = 68;
			dbits = 7920;
		    pbits = tbits - dbits;
		    name = "m1145.tab";
			LDPC_RT(ldpc_tab_11_45MX, 22);
		}
		if (code == cR_1_3){
			q     = 60;
			dbits = 10800;
		    pbits = tbits - dbits;
		    name = "m13.tab";
			LDPC_RT(ldpc_tab_1_3MX, 30);
		}
	}

	if (frame == frame_SHORT)
	{
		// DVB-S2 formats
		tbits = FRAME_SIZE_SHORT;
		if (code == cR_1_4){
			q     = 36;
			dbits = 3240;
		    pbits = tbits - dbits;
		    name = "s14.tab";
			LDPC_RT(ldpc_tab_1_4S, 9);
		}
		if (code == cR_1_3){
			q     = 30;
			dbits = 5400;
		    pbits = tbits - dbits;
		    name = "s13.tab";
			LDPC_RT(ldpc_tab_1_3S, 15);
		}
		if (code == cR_2_5){
			q     = 27;
			dbits = 6480;
		    pbits = tbits - dbits;
		    name = "s25.tab";
			LDPC_RT(ldpc_tab_2_5S, 18);
		}
		if (code == cR_1_2){
			q     = 25;
			dbits = 7200;
		    pbits = tbits - dbits;
		    name = "s12.tab";
			LDPC_RT(ldpc_tab_1_2S, 20);
		}
		if (code == cR_3_5){
			q     = 18;
			dbits = 9720;
		    pbits = tbits - dbits;
		    name = "s35.tab";
			LDPC_RT(ldpc_tab_3_5S, 27);
		}
		if (code == cR_2_3){
			q     = 15;
			dbits = 10800;
		    pbits = tbits - dbits;
		    name = "s23.tab";
			LDPC_RT(ldpc_tab_2_3S, 30);
		}
		if (code == cR_3_4){
			q     = 12;
			dbits = 11880;
		    pbits = tbits - dbits;
		    name = "s34.tab";
			LDPC_RT(ldpc_tab_3_4S, 33);
		}
		if (code == cR_4_5){
			q     = 10;
			dbits = 12960;
		    pbits = tbits - dbits;
		    name = "s45.tab";
			LDPC_RT(ldpc_tab_4_5S, 35);
		}
		if (code == cR_5_6){
			q     = 8;
			dbits = 13320;
		    pbits = tbits - dbits;
		    name = "s56.tab";
			LDPC_RT(ldpc_tab_5_6S, 37);
		}
		if (code == cR_8_9){
			q     = 5;
			dbits = 14400;
		    pbits = tbits - dbits;
		    name = "s89.tab";
			LDPC_RT(ldpc_tab_8_9S, 40);
		}

		// DVB-S2X formats
		if (code == cR_11_45){
			q     = 34;
			dbits = 3960;
		    pbits = tbits - dbits;
		    name = "s1145.tab";
			LDPC_RT(ldpc_tab_11_45SX, 11);
		}
		if (code == cR_4_15){
			q     = 33;
			dbits = 4320;
		    pbits = tbits - dbits;
		    name = "s415.tab";
			LDPC_RT(ldpc_tab_4_15SX, 12);
		}
		if (code == cR_14_45){
			q     = 31;
			dbits = 5040;
		    pbits = tbits - dbits;
		    name = "s1445.tab";
			LDPC_RT(ldpc_tab_14_45SX, 14);
		}
		if (code == cR_7_15){
			q     = 24;
			dbits = 7560;
		    pbits = tbits - dbits;
		    name = "s715.tab";
			LDPC_RT(ldpc_tab_7_15SX, 21);
		}
		if (code == cR_8_15){
            q     = 21;
			dbits = 8640;
		    pbits = tbits - dbits;
		    name = "s815.tab";
			LDPC_RT(ldpc_tab_8_15SX, 24);
		}
		if (code == cR_26_45){
			q     = 19;
			dbits = 9360;
		    pbits = tbits - dbits;
		    name = "s2645.tab";
			LDPC_RT(ldpc_tab_26_45SX, 26);
		}
		if (code == cR_32_45){
			q     = 13;
			dbits = 11520;
		    pbits = tbits - dbits;
		    name = "s3245.tab";
			LDPC_RT(ldpc_tab_32_45SX, 32);
		}
	}

	TLEntry *hvn_temp    = NULL;
	TLEntry *hcn_temp    = NULL;
	TableDB entry;

	if( name != NULL ){
		// If we have built this table before use the copy on disk
		size_t bytes = load_table_from_disk(name, &entry);
		if(bytes > 0 ){
			save_table(&entry);
			printf("%zu bytes loaded from %s\n", bytes, name);
			if( H != NULL ) free(H);
			return arg;
		}
	}

	// Table not found on disk so build it
	printf("%s Not found so building it\n", name);
	if( H != NULL ){
		// Add the parity bit mapping to the checknodes
		ldpc_cn_to_bn( H,   dbits, pbits);
		// Calculate the number of edges in the Tanner graph
		calculate_d( H, pbits, tbits, D, edges);
		// Allocate the memory needed to store the information

		hvn_temp             = (TLEntry*) calloc(edges,sizeof(TLEntry));
		hcn_temp             = (TLEntry*) calloc(edges,sizeof(TLEntry));
		uint32_t *hvn        = (uint32_t*)calloc(edges,sizeof(uint32_t));
		uint32_t *hcn        = (uint32_t*)calloc(edges,sizeof(uint32_t));
		uint32_t *hvn_starts = (uint32_t*)calloc(edges,sizeof(uint32_t));
		uint32_t *hcn_starts = (uint32_t*)calloc(edges,sizeof(uint32_t));
		uint32_t *map        = (uint32_t*)calloc(edges,sizeof(uint32_t));

		uint16_t hvn_nstarts = 0;
		uint16_t hcn_nstarts = 0;

		build_hcn_from_h_linear(         H, pbits, tbits, hcn_temp, hcn_starts, hcn_nstarts );
		build_hvn_from_hcn_and_h_linear( H, pbits, tbits, hcn_temp, hvn_temp, edges, hvn_starts, hvn_nstarts );

		// Compress the temporary tables
		compress_hvn_linear( hvn, hvn_temp, edges );
		compress_hcn_linear( hcn, hcn_temp, edges );
		compress_map_linear( map, hcn_temp, edges );// Initial mapping of rx symbols to edges

		printf("There are %zu Edges in this table\n",edges);

		entry.n_edges     = edges;
		entry.tbits       = tbits;
		entry.map         = map;
		entry.hvn         = hvn;
		entry.hvn_starts  = hvn_starts;
		entry.hvn_nstarts = hvn_nstarts;

		entry.hcn         = hcn;
		entry.hcn_starts  = hcn_starts;
		entry.hcn_nstarts = hcn_nstarts;

		entry.code        = code;
		entry.frame       = frame;

		save_table(&entry);

		if(name != NULL ){
			size_t bytes = save_table_to_disk( name, &entry);
			printf("%zu bytes saved to %s\n",bytes,name);
		}
	}
	// Free working memory
	if(hvn_temp != NULL) free(hvn_temp);
	if(hcn_temp != NULL) free(hcn_temp);
	if( H != NULL ) free(H);
	return arg;
}
//
// Generate all the tables using multiple threads.
// If available on disk load them, otherwise build them
//
void ldpc_generate_tables(void)
{
	pthread_t threads[MAX_TABLES];
    uint32_t  params[MAX_TABLES][2];

    printf("Loading / Building lookup tables this may take some time\n");

    m_table_index = 0;

    pthread_mutex_init( &mutex_tab, NULL );

    params[0][0] = frame_NORMAL;
    params[0][1] = cR_1_4;
	pthread_create( &threads[0], NULL, ldpc_generate_tables, params[0] );

	params[1][0] = frame_NORMAL;
    params[1][1] = cR_1_3;
    pthread_create( &threads[1], NULL, ldpc_generate_tables, params[1] );

	params[2][0] = frame_NORMAL;
    params[2][1] = cR_2_5;
	pthread_create( &threads[2], NULL, ldpc_generate_tables, params[2] );

	params[3][0] = frame_NORMAL;
    params[3][1] = cR_1_2;
	pthread_create( &threads[3], NULL, ldpc_generate_tables, params[3] );

	params[4][0] = frame_NORMAL;
    params[4][1] = cR_3_5;
	pthread_create( &threads[4], NULL, ldpc_generate_tables, params[4] );

	params[5][0] = frame_NORMAL;
    params[5][1] = cR_2_3;
	pthread_create( &threads[5], NULL, ldpc_generate_tables, params[5] );

	params[6][0] = frame_NORMAL;
    params[6][1] = cR_3_4;
	pthread_create( &threads[6], NULL, ldpc_generate_tables, params[6] );

	params[7][0] = frame_NORMAL;
    params[7][1] = cR_4_5;
	pthread_create( &threads[7], NULL, ldpc_generate_tables, params[7] );

	params[8][0] = frame_NORMAL;
    params[8][1] = cR_5_6;
	pthread_create( &threads[8], NULL, ldpc_generate_tables, params[8] );

	params[9][0] = frame_NORMAL;
    params[9][1] = cR_8_9;
	pthread_create( &threads[9], NULL, ldpc_generate_tables, params[9] );

	params[10][0] = frame_NORMAL;
    params[10][1] = cR_9_10;
	pthread_create( &threads[10], NULL, ldpc_generate_tables, params[10] );

    // DVB-S2 Short frames

	params[11][0] = frame_SHORT;
    params[11][1] = cR_1_4;
	pthread_create( &threads[11], NULL, ldpc_generate_tables, params[11] );

	params[12][0] = frame_SHORT;
    params[12][1] = cR_1_3;
	pthread_create( &threads[12], NULL, ldpc_generate_tables, params[12] );

	params[13][0] = frame_SHORT;
    params[13][1] = cR_2_5;
	pthread_create( &threads[13], NULL, ldpc_generate_tables, params[13] );

	params[14][0] = frame_SHORT;
    params[14][1] = cR_1_2;
	pthread_create( &threads[14], NULL, ldpc_generate_tables, params[14] );

	params[15][0] = frame_SHORT;
    params[15][1] = cR_3_5;
	pthread_create( &threads[15], NULL, ldpc_generate_tables, params[15] );

	params[16][0] = frame_SHORT;
    params[16][1] = cR_2_3;
	pthread_create( &threads[16], NULL, ldpc_generate_tables, params[16] );

	params[17][0] = frame_SHORT;
    params[17][1] = cR_3_4;
	pthread_create( &threads[17], NULL, ldpc_generate_tables, params[17] );

	params[18][0] = frame_SHORT;
    params[18][1] = cR_4_5;
	pthread_create( &threads[18], NULL, ldpc_generate_tables, params[18] );

	params[19][0] = frame_SHORT;
    params[19][1] = cR_5_6;
	pthread_create( &threads[19], NULL, ldpc_generate_tables, params[19] );

	params[20][0] = frame_SHORT;
    params[20][1] = cR_8_9;
	pthread_create( &threads[20], NULL, ldpc_generate_tables, params[20] );

	// DVB-S2X Normal formats
	params[21][0] = frame_NORMAL;
	params[21][1] = cR_2_9;
	pthread_create( &threads[21], NULL, ldpc_generate_tables, params[21] );

	params[22][0] = frame_NORMAL;
	params[22][1] = cR_13_45;
	pthread_create( &threads[22], NULL, ldpc_generate_tables, params[22] );

	params[23][0] = frame_NORMAL;
	params[23][1] = cR_9_20;
	pthread_create( &threads[23], NULL, ldpc_generate_tables, params[23] );

	params[24][0] = frame_NORMAL;
	params[24][1] = cR_11_20;
	pthread_create( &threads[24], NULL, ldpc_generate_tables, params[24] );

	params[25][0] = frame_NORMAL;
	params[25][1] = cR_26_45;
	pthread_create( &threads[25], NULL, ldpc_generate_tables, params[25] );

	params[26][0] = frame_NORMAL;
	params[26][1] = cR_28_45;
	pthread_create( &threads[26], NULL, ldpc_generate_tables, params[26] );

	params[27][0] = frame_NORMAL;
	params[27][1]= cR_23_36;
	pthread_create( &threads[27], NULL, ldpc_generate_tables, params[27] );

	params[28][0] = frame_NORMAL;
	params[28][1] = cR_25_36;
	pthread_create( &threads[28], NULL, ldpc_generate_tables, params[28] );

	params[29][0] = frame_NORMAL;
	params[29][1] = cR_13_18;
	pthread_create( &threads[29], NULL, ldpc_generate_tables, params[29] );

	params[30][0] = frame_NORMAL;
	params[30][1] = cR_7_9;
	pthread_create( &threads[30], NULL, ldpc_generate_tables, params[30] );

	params[31][0] = frame_NORMAL;
	params[31][1] = cR_90_180;
	pthread_create( &threads[31], NULL, ldpc_generate_tables, params[31] );

	params[32][0] = frame_NORMAL;
	params[32][1] = cR_96_180;
	pthread_create( &threads[32], NULL, ldpc_generate_tables, params[32] );

	params[33][0] = frame_NORMAL;
	params[33][1] = cR_100_180;
	pthread_create( &threads[33], NULL, ldpc_generate_tables, params[33] );

	params[34][0] = frame_NORMAL;
	params[34][1] = cR_104_180;
	pthread_create( &threads[34], NULL, ldpc_generate_tables, params[34] );

	params[35][0] = frame_NORMAL;
	params[35][1] = cR_116_180;
	pthread_create( &threads[35], NULL, ldpc_generate_tables, params[35] );

	params[36][0] = frame_NORMAL;
	params[36][1] = cR_124_180;
	pthread_create( &threads[36], NULL, ldpc_generate_tables, params[36] );

	params[37][0] = frame_NORMAL;
	params[37][1] = cR_128_180;
	pthread_create( &threads[37], NULL, ldpc_generate_tables, params[37] );

	params[38][0] = frame_NORMAL;
	params[38][1] = cR_132_180;
	pthread_create( &threads[38], NULL, ldpc_generate_tables, params[38] );

	params[39][0] = frame_NORMAL;
	params[39][1] = cR_135_180;
	pthread_create( &threads[39], NULL, ldpc_generate_tables, params[39] );

	params[40][0] = frame_NORMAL;
	params[40][1] = cR_140_180;
	pthread_create( &threads[40], NULL, ldpc_generate_tables, params[40] );

	params[41][0] = frame_NORMAL;
	params[41][1] = cR_154_180;
	pthread_create( &threads[41], NULL, ldpc_generate_tables, params[41] );

	params[42][0] = frame_NORMAL;
	params[42][1] = cR_18_30;
	pthread_create( &threads[42], NULL, ldpc_generate_tables, params[42] );

	params[43][0] = frame_NORMAL;
	params[43][1] = cR_20_30;
	pthread_create( &threads[43], NULL, ldpc_generate_tables, params[43] );

	params[44][0] = frame_NORMAL;
	params[44][1] = cR_22_30;
	pthread_create( &threads[44], NULL, ldpc_generate_tables, params[44] );

	// DVB-S2X Medium formats
	params[45][0] = frame_MEDIUM;
	params[45][1] = cR_1_5;
	pthread_create( &threads[45], NULL, ldpc_generate_tables, params[45] );

	params[46][0] = frame_MEDIUM;
	params[46][1] = cR_11_45;
	pthread_create( &threads[46], NULL, ldpc_generate_tables, params[46] );

	params[47][0] = frame_MEDIUM;
	params[47][1] = cR_1_3;
	pthread_create( &threads[47], NULL, ldpc_generate_tables, params[47] );

    // DVB-S2X Short formats
	params[48][0] = frame_SHORT;
	params[48][1] = cR_11_45;
	pthread_create( &threads[48], NULL, ldpc_generate_tables, params[48] );

	params[49][0] = frame_SHORT;
	params[49][1] = cR_4_15;
	pthread_create( &threads[49], NULL, ldpc_generate_tables, params[49] );

	params[50][0] = frame_SHORT;
	params[50][1] = cR_14_45;
	pthread_create( &threads[50], NULL, ldpc_generate_tables, params[50] );

	params[51][0] = frame_SHORT;
	params[51][1] = cR_7_15;
	pthread_create( &threads[51], NULL, ldpc_generate_tables, params[51] );

	params[52][0] = frame_SHORT;
	params[52][1] = cR_8_15;
	pthread_create( &threads[52], NULL, ldpc_generate_tables, params[52] );

	params[53][0] = frame_SHORT;
	params[53][1] = cR_26_45;
	pthread_create( &threads[53], NULL, ldpc_generate_tables, params[53] );

	params[54][0] = frame_SHORT;
	params[54][1] = cR_32_45;
	pthread_create( &threads[54], NULL, ldpc_generate_tables, params[54] );

	// Wait until all threads have finished
    for( int i = 0; i <= 54; i++ ) pthread_join(threads[i], NULL);

    // Mutex no longer required
    pthread_mutex_destroy( &mutex_tab );

    printf("Tables are now ready\n");
}
void CUDART_CB ldpc2_callback( cudaStream_t stream, cudaError_t status, void *data){
	rxb_output_serial( m_h_bytes, m_h_checks );
}
//
// Process next 128 LDPC frames, with callback
//
void ldpc2_decode( LLR *d_m ){

	initial_update  <<<m_device_tab.hvn_nstarts,32>>>((uint32_t*)d_m, m_device_tab.hvn_starts, m_device_tab.hvn, (uint32_t *)m_device_tab.msg );
	checknode_update<<<m_device_tab.hcn_nstarts,32>>>(      m_device_tab.hcn_starts, m_device_tab.hcn, (uint32_t *)m_device_tab.msg );
	for(int i = 0; i < g_format.ldpc_iterations; i++ ){
		bitnode_update  <<<m_device_tab.hvn_nstarts,32>>>((uint32_t*)d_m, m_device_tab.hvn_starts, m_device_tab.hvn,(uint32_t*) m_device_tab.msg );
		checknode_update<<<m_device_tab.hcn_nstarts,32>>>(      m_device_tab.hcn_starts, m_device_tab.hcn, (uint32_t *)m_device_tab.msg );
	}
	final_update<<<g_format.nbch,32>>>((uint32_t *) d_m, m_device_tab.hvn_starts, m_device_tab.hvn, (uint32_t *)m_device_tab.msg, m_d_bits );
	compactto8<<<NP_FRAMES,1>>>( m_d_bits, m_d_bytes, g_format.nbch );
	bch_device_nframe_check( m_d_bytes, m_d_checks );
	cudaMemcpyAsync(m_h_bytes,m_d_bytes,sizeof(uint8_t)*g_format.nbch*NP_FRAMES/8, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(m_h_checks,m_d_checks,sizeof(uint8_t)*NP_FRAMES, cudaMemcpyDeviceToHost);
	cudaStreamAddCallback( 0, ldpc2_callback, NULL, 0);
}
//
// Process next 128 LDPC frames, no callback
//
void ldpc2_decode(LLR *d_m, uint8_t *d_bytes, uint8_t *d_checks){

	initial_update  <<<m_device_tab.hvn_nstarts,32>>>((uint32_t*)d_m, m_device_tab.hvn_starts, m_device_tab.hvn, (uint32_t *)m_device_tab.msg );
	checknode_update<<<m_device_tab.hcn_nstarts,32>>>(      m_device_tab.hcn_starts, m_device_tab.hcn, (uint32_t *)m_device_tab.msg );
	for(int i = 0; i < g_format.ldpc_iterations; i++ ){
		bitnode_update  <<<m_device_tab.hvn_nstarts,32>>>((uint32_t*)d_m, m_device_tab.hvn_starts, m_device_tab.hvn,(uint32_t*) m_device_tab.msg );
		checknode_update<<<m_device_tab.hcn_nstarts,32>>>(      m_device_tab.hcn_starts, m_device_tab.hcn, (uint32_t *)m_device_tab.msg );
	}
	final_update<<<g_format.nbch,32>>>((uint32_t *) d_m, m_device_tab.hvn_starts, m_device_tab.hvn, (uint32_t *)m_device_tab.msg, m_d_bits );
	compactto8<<<NP_FRAMES,1>>>( m_d_bits, d_bytes, g_format.nbch );
	bch_device_nframe_check( d_bytes, d_checks );
}
//
// Set the LDPC decoder
//
void ldpc2_decode_set_fec(void){

	load_table(g_format.frame_type, g_format.code_rate);
}
//
// Open the LDPC decoder
//
void ldpc2_decode_open(void){
	ldpc_generate_tables();

	// Clear the device table
	m_device_tab.hvn          = NULL;
	m_device_tab.hcn          = NULL;
	m_device_tab.hvn_starts   = NULL;
	m_device_tab.hcn_starts   = NULL;
	m_device_tab.map          = NULL;
	m_device_tab.hvn_nstarts  = 0;
	m_device_tab.hcn_nstarts  = 0;
	m_device_tab.n_edges      = 0;
	CHECK(cudaMalloc((void**)&m_d_bits,   sizeof(Bit)*FRAME_SIZE_NORMAL*NP_FRAMES));
	CHECK(cudaMalloc((void**)&m_d_bytes,  sizeof(uint8_t)*FRAME_SIZE_NORMAL*NP_FRAMES/8));
	CHECK(cudaMalloc((void**)&m_d_checks, sizeof(uint8_t)*NP_FRAMES));
	// Allocate host pinned memory for use with the callback
	CHECK(cudaHostAlloc((void**)&m_h_bytes,  sizeof(uint8_t)*FRAME_SIZE_NORMAL*NP_FRAMES/8, cudaHostAllocDefault))
	CHECK(cudaHostAlloc((void**)&m_h_checks, sizeof(uint8_t)*NP_FRAMES, cudaHostAllocDefault))
}
//
// Close the LDPC decoder
//
void ldpc2_decode_close(void){

	if(m_device_tab.hvn        != NULL)cudaFree(m_device_tab.hvn);
	if(m_device_tab.hcn        != NULL)cudaFree(m_device_tab.hcn);
	if(m_device_tab.hvn_starts != NULL)cudaFree(m_device_tab.hvn_starts);
	if(m_device_tab.hcn_starts != NULL)cudaFree(m_device_tab.hcn_starts);
	if(m_d_bits                != NULL)cudaFree(m_d_bits);
	if(m_d_bytes               != NULL)cudaFree(m_d_bytes);
	if(m_d_checks              != NULL)cudaFree(m_d_checks);
	if(m_h_bytes               != NULL)cudaFreeHost(m_h_bytes);
	if(m_h_checks              != NULL)cudaFreeHost(m_h_checks);

}
