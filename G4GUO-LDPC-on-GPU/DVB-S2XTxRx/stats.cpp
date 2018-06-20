#include <semaphore.h>
#include <pthread.h>

#include "dvbs2_rx.h"

static pthread_mutex_t mutex_stats;


static int m_bbh_errors;
static int m_bch_errors;
static int m_freq_errors;
static float m_freq_error_accum;
static int m_mag_count;
static float m_mag_accum;
static bool m_stats_update;
static uint64_t m_tp_rxed;
static uint64_t m_tp_errors;
static uint32_t m_ldpc_fes;
static float m_mer;

void stats_open(void){
    pthread_mutex_init( &mutex_stats, NULL );
}
void stats_close(void){
    pthread_mutex_destroy( &mutex_stats );
}
void stats_update(void){
    pthread_mutex_lock( &mutex_stats );
    m_stats_update = true;
    pthread_mutex_unlock( &mutex_stats );
}
bool stats_update_read(void){
    pthread_mutex_lock( &mutex_stats );
    bool res = m_stats_update;
    m_stats_update = false;
    pthread_mutex_unlock( &mutex_stats );
    return res;
}

void stats_bbh_errors_reset(void){
    pthread_mutex_lock( &mutex_stats );
    //m_bbh_errors = 0;
    pthread_mutex_unlock( &mutex_stats );
}
int stats_bbh_errors_read(void){
    pthread_mutex_lock( &mutex_stats );
    int e = m_bbh_errors;
    pthread_mutex_unlock( &mutex_stats );
    return e;
}
void stats_bbh_errors_update(int n){
    pthread_mutex_lock( &mutex_stats );
    m_bbh_errors = n;
    pthread_mutex_unlock( &mutex_stats );
}

void stats_bch_errors_reset(void){
    pthread_mutex_lock( &mutex_stats );
    //m_bch_errors = 0;
    pthread_mutex_unlock( &mutex_stats );
}
int stats_bch_errors_read(void){
    pthread_mutex_lock( &mutex_stats );
    int e = m_bch_errors;
    pthread_mutex_unlock( &mutex_stats );
    return e;
}
void stats_bch_errors_update(int n){
    pthread_mutex_lock( &mutex_stats );
    m_bch_errors = n;
    pthread_mutex_unlock( &mutex_stats );
}

void stats_freq_error_update(float error){
    pthread_mutex_lock( &mutex_stats );
    m_freq_error_accum += error;
    m_freq_errors++;
    pthread_mutex_unlock( &mutex_stats );
}
void stats_freq_error_reset(void){
    pthread_mutex_lock( &mutex_stats );
    m_freq_error_accum = 0;
    m_freq_errors      = 0;
    pthread_mutex_unlock( &mutex_stats );
}
float stats_freq_error_read(void){
    pthread_mutex_lock( &mutex_stats );
    float e = 0;
    if(m_freq_errors > 0) e = m_freq_error_accum/m_freq_errors;
    pthread_mutex_unlock( &mutex_stats );
    return e;
}

void stats_mag_update(float mag){
    pthread_mutex_lock( &mutex_stats );
    m_mag_accum += mag;
    m_mag_count++;
    pthread_mutex_unlock( &mutex_stats );
}
void stats_mag_reset(void){
    pthread_mutex_lock( &mutex_stats );
    m_mag_accum = 0;
    m_mag_count = 0;
    pthread_mutex_unlock( &mutex_stats );
}
float stats_mag_read(void){
    pthread_mutex_lock( &mutex_stats );
    float e = 0;
    if(m_mag_count > 0) e = m_mag_accum/m_mag_count;
    pthread_mutex_unlock( &mutex_stats );
    return e;
}

void stats_tp_rx_update(uint32_t n){
    pthread_mutex_lock( &mutex_stats );
    m_tp_rxed += n;
    pthread_mutex_unlock( &mutex_stats );
}
void stats_tp_rx_reset(void){
    pthread_mutex_lock( &mutex_stats );
    m_tp_rxed = 0;
    pthread_mutex_unlock( &mutex_stats );
}
uint64_t stats_tp_rx_read(void){
    pthread_mutex_lock( &mutex_stats );
    uint64_t rx = m_tp_rxed;
    pthread_mutex_unlock( &mutex_stats );
    return rx;
}

void stats_tp_er_update(uint32_t n){
    pthread_mutex_lock( &mutex_stats );
    m_tp_errors += n;
    pthread_mutex_unlock( &mutex_stats );
}
void stats_tp_er_reset(void){
    pthread_mutex_lock( &mutex_stats );
    m_tp_errors = 0;
    pthread_mutex_unlock( &mutex_stats );
}
uint64_t stats_tp_er_read(void){
    pthread_mutex_lock( &mutex_stats );
    uint64_t e = m_tp_errors;
    pthread_mutex_unlock( &mutex_stats );
    return e;
}

void stats_ldpc_fes_update(uint32_t fes){
	m_ldpc_fes = fes;
}
uint32_t stats_ldpc_fes_read(void){
	return m_ldpc_fes;
}
void stats_mer_update(float mer){
	m_mer = mer;
}
float stats_mer_read(void){
	return m_mer;
}
