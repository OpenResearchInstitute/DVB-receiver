#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "dvbs2_rx.h"
#include "noise.h"

extern RxFormat g_format;
FComplex mem[FRAME_SIZE_NORMAL];
FComplex m_payload[FRAME_SIZE_NORMAL*NP_FRAMES];

void device_display_array(FComplex *ds, int len ){
	FComplex *b = (FComplex *)malloc(sizeof(FComplex)*len);
	CHECK(cudaMemcpy( b, ds, sizeof(FComplex)*len, cudaMemcpyDeviceToHost));
	for( int i = 0; i < len; i++){
		printf("%d %f %f\n", i, b[i].re, b[i].im);
	}
	free(b);
}
void device_display_array(float *ds, int len ){
	float *b = (float *)malloc(sizeof(float)*len);
	CHECK(cudaMemcpy( b, ds, sizeof(float)*len, cudaMemcpyDeviceToHost));
	for( int i = 0; i < len; i++){
		printf("%d %f\n", i, b[i]);
	}
	free(b);
}
void device_display_array(float *da, float *db, int len ){
	float *a = (float *)malloc(sizeof(float)*len);
	float *b = (float *)malloc(sizeof(float)*len);
	CHECK(cudaMemcpy( a, da, sizeof(float)*len, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy( b, db, sizeof(float)*len, cudaMemcpyDeviceToHost));
	for( int i = 0; i < len; i++){
		printf("%d %f \t %f\n", i, a[i], b[i]);
	}
	free(b);
}
void device_display_array(uint8_t *ds, int len ){
	uint8_t *b = (uint8_t *)malloc(sizeof(uint8_t)*len);
	CHECK(cudaMemcpy( b, ds, sizeof(uint8_t)*len, cudaMemcpyDeviceToHost));
	for( int i = 0; i < len; i++){
		printf("%d %.2X\n", i, b[i]);
	}
	free(b);
}
void device_display_array(uint16_t *ds, int len ){
	uint16_t *b = (uint16_t *)malloc(sizeof(uint16_t)*len);
	CHECK(cudaMemcpy( b, ds, sizeof(uint16_t)*len, cudaMemcpyDeviceToHost));
	for( int i = 0; i < len; i++){
		printf("%d %.4X\n", i, b[i]);
	}
	free(b);
}

void host_display_array(FComplex *b, int len ){
	for( int i = 0; i < len; i++){
		printf("%d %f %f\n", i, b[i].re, b[i].im);
	}
}
void host_display_array(float *b, int len ){
	for( int i = 0; i < len; i++){
		printf("%d %f\n", i, b[i]);
	}
}
void host_display_array(uint8_t *b, int len ){
	for( int i = 0; i < len; i++){
		printf("%d %.2X\n", i, b[i]);
	}
}
void host_display_array(uint16_t *b, int len ){
	for( int i = 0; i < len; i++){
		printf("%d %.4X\n", i, b[i]);
	}
}

double cpuSec(void){
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC,&tp);
	return((double)tp.tv_sec + (double)tp.tv_nsec*1.e-9);
}
double cpuSecGetRes(void){
	struct timespec tp;
	clock_getres(CLOCK_MONOTONIC,&tp);
	return((double)tp.tv_sec + (double)tp.tv_nsec*1.e-9);
}

void benchtest_data_output_transportstream(uint8_t *in, int len, BBHeader *h);

static uint8_t  *d_bytes;
static uint8_t  *d_checks;
static FComplex *d_s;
static LLR *d_m;

void test(const char *filename, float sn){
	FILE *fp;
	int index = 0;
	char text[256];
	float *m = (float*)mem;
    FComplex *payload;
    int len;

    receiver_open();

	CHECK(cudaMalloc((void**)&d_bytes,  sizeof(uint8_t)*FRAME_SIZE_NORMAL*NP_FRAMES/8));
	CHECK(cudaMalloc((void**)&d_checks, sizeof(uint8_t)*NP_FRAMES));
	CHECK(cudaMalloc((void**)&d_s,      sizeof(FComplex)*FRAME_SIZE_NORMAL*NP_FRAMES));
	CHECK(cudaMalloc((void**)&d_m,      sizeof(LLR)*FRAME_SIZE_NORMAL*NP_FRAMES));

	float va = sn;
	float sa = 0;

	noise_init();
	noise_set_es_no(sn);
	noise_on();

	if((fp=fopen(filename,"r"))!=NULL){
		while(fgets(text,255,fp)!=NULL){
			m[index++] = atof(text);
		}
		fclose(fp);
		// Now call routine to process samples
		len = index/2;

		// descramble payload data
		payload = &mem[90];
		for( int i = 0; i < len-90; i++){
			payload[i] = descramble_symbol(payload[i], i);
		}
	}else{
		fprintf(stderr,"%s not found\n",filename);
		return;
	}

	va = noise_add(mem, 90);

	uint8_t modcod = pl_decode(&mem[26]);
	pl_new_modcod(modcod);

	printf("MODCOD = %d %s\n",modcod,g_format.format_text);
    //
	// Make multiple copies of the payload if needed
	//
	int idx = 0;
	for( int i = 0; i < g_format.nsyms; i++ ){
	    for( int j = 0; j < NP_FRAMES; j++ ){
	    	m_payload[idx++] = payload[i];
	    }
	}
	va += noise_add(m_payload,idx);
	va = va/2.0f;

	sa = 10*log10(1.0/va);

	printf("Test %s\n",filename);
	printf("SN REQUESTED %f\n",sn);
	printf("SN ACHIEVED %f\n",sa);

	// Copy from host memory space to device memory space
	CHECK(cudaMemcpy(d_s,m_payload,sizeof(FComplex)*g_format.nsyms*NP_FRAMES, cudaMemcpyHostToDevice));
	// Demap and de interleave (if necessary)
	printf("De-mapper\n");
	printf("Variance %f\n",va);

	demapin(d_s, d_m, va, g_format.itype, g_format.nsyms);//Eb/No = (Es/No)*0.5 for QPSK
//	device_display_array( d_m, 30 );

//	printf("Timing resolution %10f\n",cpuSecGetRes());
	// LDPC Decode
	printf("LDPC Decoder %d Iterations, ",g_format.ldpc_iterations);
	double iStart = cpuSec();
	ldpc2_decode( d_m, d_bytes, d_checks );
	double iEnd = cpuSec();
	printf("Elapsed time %f sec\n", iEnd - iStart);

//	device_display_array(d_bit, 20 );

	uint8_t *ma = (uint8_t*)malloc((g_format.nbch*sizeof(uint8_t)*NP_FRAMES/8));
	uint8_t *mc = (uint8_t*)malloc(sizeof(uint8_t)*NP_FRAMES);
    CHECK(cudaMemcpy( ma, d_bytes,  sizeof(uint8_t)*g_format.nbch*NP_FRAMES/8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy( mc, d_checks, sizeof(uint8_t)*NP_FRAMES, cudaMemcpyDeviceToHost));

    data_output_open();

	stats_open();
	int es = 0;
	int bbh_errors = 0;
    int c_errors = 0;

    stats_tp_er_reset();
    iStart = cpuSec();

    for( int i = 0; i < NP_FRAMES; i++){
	    // BCH decode
    	uint8_t *b = &ma[g_format.nbch*i/8];
	    if(mc[i]){
	    	es += bch_host_byte_decode( b );
	    	c_errors++;
	    }
	    // We now have a octet stream for further processing
//	    printf("De Randomise BB frame\n");
	    int plen = g_format.kbch/8;
	    bb_derandomise( b, plen );
    //host_display_array(m_out, 400 );
	    BBHeader h;
	    bb_header_decode( b, &h);
	    plen = plen - 10;
	    if(h.crc_ok){
//		    printf("BB Header CRC OK\n");
	        if(h.ts_gs == BB_TS) benchtest_data_output_transportstream( &b[10], plen, &h);
	    }
	    else{
		    bbh_errors++;
	    }
//        for( int i = 0; i < 10; i++) printf("%.2x ",b[i]);
//        printf("\n");
	}
    iEnd = cpuSec();
    printf("Frame Errors %d\n",c_errors);
    printf("BCH decoder %d Errors, Elapsed time %f sec\n",es,iEnd - iStart);
    printf("BBH CRC Errors %d\n",bbh_errors);
    printf("TP Errors %ld\n",stats_tp_er_read());
    stats_close();
	free(ma);
	free(mc);
	printf("End of test\n");
	receiver_close();
}
/*
void test_new(const char *filename, float sn){
	FILE *fp;
	int index = 0;
	char text[256];
	float *m = (float*)mem;
    FComplex *payload;
    int len;

    receiver_open();

	CHECK(cudaMalloc((void**)&d_bit, sizeof(Bit)*FRAME_SIZE_NORMAL*NP_FRAMES));
	CHECK(cudaMalloc((void**)&d_s,   sizeof(FComplex)*FRAME_SIZE_NORMAL*NP_FRAMES));
	CHECK(cudaMalloc((void**)&d_m,   sizeof(LLR)*FRAME_SIZE_NORMAL*NP_FRAMES));

	float va = sn;
	float sa = 0;

	noise_init();
	noise_set_es_no(sn);
	noise_on();

	if((fp=fopen(filename,"r"))!=NULL){
		while(fgets(text,255,fp)!=NULL){
			m[index++] = atof(text);
		}
		fclose(fp);
		// Now call routine to process samples
		len = index/2;

		// descramble payload data
		payload = &mem[90];
		for( int i = 0; i < len-90; i++){
			payload[i] = descramble_symbol(payload[i], i);
		}
	}else{
		fprintf(stderr,"%s not found\n",filename);
		return;
	}

	va = noise_add(mem, 90);

	uint8_t modcod = pl_decode(&mem[26]);
	pl_new_modcod(modcod);

	printf("MODCOD = %d %s\n",modcod,modcod_trace(modcod));
    //
	// Make multiple copies of the payload if needed
	//
	int idx = 0;
	for( int i = 0; i < g_format.nsyms; i++ ){
	    for( int j = 0; j < NP_FRAMES; j++ ){
	    	m_payload[idx++] = payload[i];
	    }
	}
	va += noise_add(m_payload,idx);
	sa = 10*log10(1.0/va);

	printf("Test %s\n",filename);
	printf("SN REQUESTED %f\n",sn);
	printf("SN ACHIEVED %f\n",sa);

	// Copy from host memory space to device memory space
	CHECK(cudaMemcpy(d_s,m_payload,sizeof(FComplex)*g_format.nsyms*NP_FRAMES, cudaMemcpyHostToDevice));
	// Demap and de interleave (if necessary)
	printf("De-mapper\n");
	printf("Variance %f\n",va);

	demapin(d_s, d_m, va, g_format.itype, g_format.nsyms);//Eb/No = (Es/No)*0.5 for QPSK
//	device_display_array( d_m, 30 );

//	printf("Timing resolution %10f\n",cpuSecGetRes());
	// LDPC Decode
	printf("LDPC Decoder %d Iterations, ",g_format.ldpc_iterations);
	double iStart = cpuSec();
#ifdef USE_LDPC2
	ldpc2_decode( d_m, d_bit );
#else
	ldpc_decode( d_m, d_bit );
#endif
	double iEnd = cpuSec();
	printf("Elapsed time %f sec\n", iEnd - iStart);

//	device_display_array(d_bit, 20 );

    data_output_open();

	stats_open();
	int es = 0;
	int bbh_errors = 0;
    stats_tp_errors_reset();

    iStart = cpuSec();
	es = bch_device_decode( d_bit, m_out);
    iEnd = cpuSec();

    printf("BCH decoder %d Errors, Elapsed time %f sec\n",es,iEnd - iStart);

	for( int i = 0; i < NP_FRAMES; i++){
	    // We now have a octet stream for further processing
//	    printf("De Randomise BB frame\n");
	    int plen = g_format.kbch/8;
	    bb_derandomise(&m_out[plen*i], plen );
    //host_display_array(m_out, 400 );
	    BBHeader h;
	    bb_header_decode( &m_out[plen*i], &h);
	    if(h.crc_ok){
//		    printf("BB Header CRC OK\n");
	        if(h.ts_gs == BB_TS) benchtest_data_output_transportstream( &m_out[(plen*i)+10], plen - 10, &h);
	    }
	    else{
		    bbh_errors++;
	    }
	}
    printf("BBH CRC Errors %d\n",bbh_errors);
    printf("TP Errors %d\n",stats_tp_errors_read());
    stats_close();
	printf("End of test\n");
	receiver_close();
}
*/
