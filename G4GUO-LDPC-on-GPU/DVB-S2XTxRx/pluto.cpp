#include "iio.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include "dvbs2_rx.h"

static bool m_running;
static bool m_rx_configured;
static bool stop;
/* static scratch mem for strings */
static char tmpstr[64];
/* helper macros */
#define MHZ(x) ((long long)(x*1000000.0 + .5))
#define GHZ(x) ((long long)(x*1000000000.0 + .5))

static char error_string[256];
#define report_error(s) printf("%s\n",s)

#define FMC_ERROR(expr) { \
	if (!(expr)) { \
		(void) sprintf(error_string, "errorion failed (%s:%d)\n", __FILE__, __LINE__); \
         report_error(error_string); \
		 (void)abort(); \
	} \
}
/* RX is input, TX is output */
enum iodev { RX, TX };

/* common RX and TX streaming params */
struct stream_cfg {
	long long bw_hz; // Analog bandwidth in Hz
	long long fs_hz; // Baseband sample rate in Hz
	long long lo_hz; // Local oscillator frequency in Hz
	const char* rfport; // Port name
};

/* IIO structs required for streaming */
static struct iio_context *m_ctx   = NULL;
static struct iio_channel *m_rx0_i = NULL;
static struct iio_channel *m_rx0_q = NULL;
static struct iio_channel *m_tx0_i = NULL;
static struct iio_channel *m_tx0_q = NULL;
static struct iio_buffer  *m_rxbuf = NULL;
static struct iio_buffer  *m_txbuf = NULL;
// Streaming devices
struct iio_device *m_tx = NULL;
struct iio_device *m_rx = NULL;


static void handle_sig(int sig)
{
	printf("Waiting for process to finish...\n");
	stop = true;
}

/* check return value of attr_write function */
static void errchk(int v, const char* what) {
	if (v < 0) { fprintf(stderr, "FMC_ERROR %d writing to channel \"%s\"\nvalue may not be supported.\n", v, what); pluto_close(); }
}
/* read attribute: long long int */
static void rd_ch_lli(struct iio_channel *chn, const char* what, long long *val)
{
	errchk(iio_channel_attr_read_longlong(chn, what, val), what);
}

/* write attribute: long long int */
static void wr_ch_lli(struct iio_channel *chn, const char* what, long long val)
{
	errchk(iio_channel_attr_write_longlong(chn, what, val), what);
}

/* write attribute: string */
static void wr_ch_str(struct iio_channel *chn, const char* what, const char* str)
{
	errchk(iio_channel_attr_write(chn, what, str), what);
}

/* helper function generating channel names */
static char* get_ch_name(const char* type, int id)
{
	snprintf(tmpstr, sizeof(tmpstr), "%s%d", type, id);
	return tmpstr;
}

/* returns ad9361 phy device */
static struct iio_device* get_ad9361_phy(struct iio_context *ctx)
{
	struct iio_device *dev = iio_context_find_device(ctx, "ad9361-phy");
	FMC_ERROR(dev && "No ad9361-phy found");
	return dev;
}

/* finds AD9361 streaming IIO devices */
static bool get_ad9361_stream_dev(struct iio_context *ctx, enum iodev d, struct iio_device **dev)
{
	switch (d) {
	case TX: *dev = iio_context_find_device(ctx, "cf-ad9361-dds-core-lpc"); return *dev != NULL;
	case RX: *dev = iio_context_find_device(ctx, "cf-ad9361-lpc");  return *dev != NULL;
	default: FMC_ERROR(0); return false;
	}
}

/* finds AD9361 streaming IIO channels */
static bool get_ad9361_stream_ch(struct iio_context *ctx, enum iodev d, struct iio_device *dev, int chid, struct iio_channel **chn)
{
	*chn = iio_device_find_channel(dev, get_ch_name("voltage", chid), d == TX);
	if (!*chn)
		*chn = iio_device_find_channel(dev, get_ch_name("altvoltage", chid), d == TX);
	return *chn != NULL;
}

/* finds AD9361 phy IIO configuration channel with id chid */
static bool get_phy_chan(struct iio_context *ctx, enum iodev d, int chid, struct iio_channel **chn)
{
	switch (d) {
	case RX: *chn = iio_device_find_channel(get_ad9361_phy(ctx), get_ch_name("voltage", chid), false); return *chn != NULL;
	case TX: *chn = iio_device_find_channel(get_ad9361_phy(ctx), get_ch_name("voltage", chid), true);  return *chn != NULL;
	default: FMC_ERROR(0); return false;
	}
}

/* finds AD9361 local oscillator IIO configuration channels */
static bool get_lo_chan(struct iio_context *ctx, enum iodev d, struct iio_channel **chn)
{
	switch (d) {
		// LO chan is always output, i.e. true
	case RX: *chn = iio_device_find_channel(get_ad9361_phy(ctx), get_ch_name("altvoltage", 0), true); return *chn != NULL;
	case TX: *chn = iio_device_find_channel(get_ad9361_phy(ctx), get_ch_name("altvoltage", 1), true); return *chn != NULL;
	default: FMC_ERROR(0); return false;
	}
}

/* applies streaming configuration through IIO */
bool cfg_ad9361_streaming_ch(struct iio_context *ctx, struct stream_cfg *cfg, enum iodev type, int chid)
{
	struct iio_channel *chn = NULL;

	// Configure phy and lo channels
	printf("* Acquiring AD9361 phy channel %d\n", chid);
	if (!get_phy_chan(ctx, type, chid, &chn)) { return false; }
	wr_ch_str(chn, "rf_port_select", cfg->rfport);
	wr_ch_lli(chn, "rf_bandwidth", cfg->bw_hz);
	wr_ch_lli(chn, "sampling_frequency", cfg->fs_hz);

	// Configure LO channel
	printf("* Acquiring AD9361 %s lo channel\n", type == TX ? "TX" : "RX");
	if (!get_lo_chan(ctx, type, &chn)) { return false; }
	wr_ch_lli(chn, "frequency", cfg->lo_hz);
	return true;
}
//
//
void fmc_load_filter(short *fir, int taps, int ratio, bool enable) {
	if (m_running == false) return;
	if (m_ctx == NULL) return;
	int res;
	struct iio_channel *chn = NULL;
	iio_device* dev = NULL;
	dev = get_ad9361_phy(m_ctx);
	int buffsize = 8192;
	char *buf = (char *)malloc(buffsize);
	int clen = 0;
	clen += sprintf(buf + clen, "RX 3 GAIN 0 DEC %d\n", ratio);
	clen += sprintf(buf + clen, "TX 3 GAIN 0 INT %d\n", ratio);
	for (int i = 0; i < taps; i++) clen += sprintf(buf + clen, "%d,%d\n", fir[i], fir[i]);
	//for (int i = 0; i < taps; i++) clen += sprintf_s(buf + clen, buffsize - clen, "%d\n", fir[i]);
	clen += sprintf(buf + clen, "\n");
	res = iio_device_attr_write_raw(dev, "filter_fir_config", buf, clen);
	chn = iio_device_find_channel(dev, "voltage0", true);
	res = iio_channel_attr_write_bool(chn, "filter_fir_en", false);
	chn = iio_device_find_channel(dev, "voltage0", false);
	res = iio_channel_attr_write_bool(chn, "filter_fir_en", false);
	if (!get_phy_chan(m_ctx, TX, 0, &chn)) { return; }
	res = iio_channel_attr_write_bool(chn,"filter_fir_en", false);
	chn = iio_device_find_channel(dev, "out", false);
	res = iio_channel_attr_write_bool(chn, "voltage_filter_fir_en", true);

	//	if (!get_phy_chan(m_ctx, RX, 0, &chn)) { return; }
	free(buf);
}

void pluto_load_rrc_filter(float roff) {
	if (m_running == false) return;
	short *fir = rrc_make_filter(roff, 4, 128);
	fmc_load_filter(fir, 128, 4, true);
}

void fmc_set_tx_level(double level) {
	if (m_running == false) return;
	if (m_ctx == NULL) return;
	if (level > 0) level = 0;
	if (level < -89) level = -89;

	struct iio_channel *chn = NULL;
	if (!get_phy_chan(m_ctx, TX, 0, &chn)) { return; }
	wr_ch_lli(chn, "hardwaregain", (long long int)level);
}
void fmc_set_rx_gain(double level) {
	if (m_running == false) return;
	if (m_ctx == NULL) return;
	if (level > 0) level = 0;
	if (level < -89) level = -89;

	iio_device* dev = NULL;
	dev = get_ad9361_phy(m_ctx);
	struct iio_channel *chn = NULL;
	chn = iio_device_find_channel(dev, "voltage0", false);
	wr_ch_lli(chn, "hardwaregain", (long long int)level);
}

void fmc_set_transmit(void) {
	if (m_running == false) return;
	if (m_ctx == NULL) return;
//	fmc_set_tx_level(get_current_tx_level() - 47);
}
void fmc_set_receive(void) {
	if (m_running == false) return;
	if (m_ctx == NULL) return;
	fmc_set_tx_level(-89);
}

void fmc_set_tx_frequency(double freq) {
	if (m_running == false) return;
	struct iio_channel *chn = NULL;
	if (m_ctx == NULL) return;
	if (!get_lo_chan(m_ctx, TX, &chn)) { return; }
	wr_ch_lli(chn, "frequency", (long long int)freq);
}
void fmc_configure_x8_int_dec(long long int  sr) {
	if (m_running == false) return;
	if (m_ctx == NULL) return;
	iio_device* dev = NULL;
	struct iio_channel *chn = NULL;
	// Receive
	dev = iio_context_find_device(m_ctx, "cf-ad9361-lpc");
	if (dev == NULL) return;
	chn = iio_device_find_channel(dev, "voltage0", false);
	wr_ch_lli(chn, "sampling_frequency", sr);
	// Transmit
	dev = iio_context_find_device(m_ctx, "cf-ad9361-dds-core-lpc");
	if (dev == NULL) return;
	chn = iio_device_find_channel(dev, "voltage0", true);
	wr_ch_lli(chn, "sampling_frequency", sr);
}

void fmc_set_tx_sr(long long int sr) {
	if (m_running == false) return;
	struct iio_channel *chn = NULL;
	if (m_ctx == NULL) return;
	if (!get_phy_chan(m_ctx, TX, 0, &chn)) { return; }
	wr_ch_lli(chn, "sampling_frequency", (long long int)sr);
}

void pluto_set_rx_sr(double sr) {
	if (m_running == false) return;
	struct iio_channel *chn = NULL;
	if (m_ctx == NULL) return;
	if(sr < 520000){
		if (!get_phy_chan(m_ctx, RX, 0, &chn)) { return; }
	    wr_ch_lli(chn, "sampling_frequency", (long long int)sr*8);
	    fmc_configure_x8_int_dec(sr);
	}else{
		if (!get_phy_chan(m_ctx, RX, 0, &chn)) { return; }
	    wr_ch_lli(chn, "sampling_frequency", (long long int)sr);
	    fmc_configure_x8_int_dec(sr);
	}
}

double pluto_get_rx_sr(void) {
	if (m_running == false) return 0;
	struct iio_channel *chn = NULL;
	if (m_ctx == NULL) return 0;
	long long int sr;
	if (!get_phy_chan(m_ctx, RX, 0, &chn)) { return 0; }
	rd_ch_lli(chn, "sampling_frequency", (long long int*)&sr);
	return sr;
}

void pluto_set_rx_frequency(double freq) {
	if (m_running == false) return;
	struct iio_channel *chn = NULL;
	if (m_ctx == NULL) return;
	if (!get_lo_chan(m_ctx, RX, &chn)) { return; }
	wr_ch_lli(chn, "frequency", (long long int)freq);
}

void fmc_set_tx_analog_lpf(double bw) {
	if (m_running == false) return;
	struct iio_channel *chn = NULL;
	if (!get_phy_chan(m_ctx, TX, 0, &chn)) return;
	wr_ch_lli(chn, "rf_bandwidth", (int64_t)bw);
}
static uint32_t m_max_len; // Maximum size of the buffer
static uint32_t m_offset;  // Current offset into the buffer

void fmc_set_tx_buff_length(uint32_t len) {
	// Change the size of the buffer
	m_max_len = len;
	if (m_txbuf) { iio_buffer_destroy(m_txbuf); m_txbuf = NULL; }
	m_txbuf = iio_device_create_buffer( m_tx, len, false);
	iio_buffer_set_blocking_mode(m_txbuf, true);
	m_offset = 0;
}

void fmc_set_rx_buff_length(uint32_t len) {
	// Change the size of the buffer
	if (m_rxbuf) { iio_buffer_destroy(m_rxbuf); m_rxbuf = NULL; }
	m_rxbuf = iio_device_create_buffer( m_rx, len, false);
	iio_buffer_set_blocking_mode(m_rxbuf, true);
}

int pluto_rx_samples( SComplex **s) {
	if (m_running == false) return 0;
	if (m_rx_configured == false) return 0;
	// Refill RX buffer
	int nbytes_rx = iio_buffer_refill(m_rxbuf);
	// Get position of first sample in the buffer
	s[0] = (SComplex*)iio_buffer_first(m_rxbuf, m_rx0_i);
    return nbytes_rx/4;// returns the number of symbols
}

int pluto_open( double frequency, double syrate ){
	int n_devices = 0;
	m_running = false;
	m_rx_configured = false;
	// Stream configurations
	struct stream_cfg rxcfg;
	struct stream_cfg txcfg;

	// Listen to ctrl+c and FMC_ERROR
	signal(SIGINT, handle_sig);

	// RX stream config
	rxcfg.bw_hz = 10000000;   // rf bandwidth set to 5 MHz
	rxcfg.fs_hz = 30000000;   // rx sample rate is twice baudrate
	rxcfg.lo_hz = frequency;   // rf frequency
	rxcfg.rfport = "A_BALANCED"; // port A (select for rf freq.)
								 // TX stream config
	txcfg.bw_hz = 10000000; // 5 MHz rf bandwidth
	txcfg.fs_hz = 3000000;   // 2.5 MS/s tx sample rate
	txcfg.lo_hz = frequency; // 2.5 GHz rf frequency
	txcfg.rfport = "A"; // port A (select for rf freq.)

	if((m_ctx = iio_create_network_context("192.168.2.1"))== NULL) return -1;
	if((n_devices=iio_context_get_devices_count(m_ctx)) <= 0) return -1;

	printf("Pluto Devices found %d\n",n_devices);

	printf("* Acquiring AD9361 streaming devices\n");
	FMC_ERROR(get_ad9361_stream_dev(m_ctx, TX, &m_tx) && "No tx dev found");
	FMC_ERROR(get_ad9361_stream_dev(m_ctx, RX, &m_rx) && "No rx dev found");

	printf("* Configuring AD9361 for streaming\n");
	FMC_ERROR(cfg_ad9361_streaming_ch(m_ctx, &txcfg, TX, 0) && "TX port 0 not found");
	FMC_ERROR(cfg_ad9361_streaming_ch(m_ctx, &rxcfg, RX, 0) && "RX port 0 not found");


	printf("* Initializing AD9361 IIO streaming channels\n");
	FMC_ERROR(get_ad9361_stream_ch(m_ctx, RX, m_rx, 0, &m_rx0_i) && "RX chan i not found");
	FMC_ERROR(get_ad9361_stream_ch(m_ctx, RX, m_rx, 1, &m_rx0_q) && "RX chan q not found");
	FMC_ERROR(get_ad9361_stream_ch(m_ctx, TX, m_tx, 0, &m_tx0_i) && "TX chan i not found");
	FMC_ERROR(get_ad9361_stream_ch(m_ctx, TX, m_tx, 1, &m_tx0_q) && "TX chan q not found");

	printf("* Enabling IIO streaming channels\n");
	iio_channel_enable(m_rx0_i);
	iio_channel_enable(m_rx0_q);
//	iio_channel_enable(m_tx0_i);
//	iio_channel_enable(m_tx0_q);


	// Everything should be setup now so return a success
	m_running = true;
	pluto_load_rrc_filter(0.35f);
	iio_context_set_timeout(m_ctx, 0);
//	fmc_set_tx_buff_length(360000);
	fmc_set_rx_buff_length(360000);
//	iio_device_set_kernel_buffers_count(m_tx, 16);
	iio_device_set_kernel_buffers_count(m_rx, 16);
	fmc_set_receive();
    pluto_set_rx_sr(syrate*2);
	pluto_set_rx_frequency(frequency);
//	fmc_set_rx_gain(0);

	m_rx_configured = true;

	return 0;
}
/* cleanup and exit */
void pluto_close(void)
{
	m_running = false;
	m_rx_configured = false;

	if (m_ctx != NULL) {
		printf("* Destroying buffers\n");
		if (m_rxbuf) { iio_buffer_destroy(m_rxbuf); m_rxbuf = NULL; }
		if (m_txbuf) { iio_buffer_destroy(m_txbuf); m_txbuf = NULL; }

		printf("* Disabling streaming channels\n");
		if (m_rx0_i) { iio_channel_disable(m_rx0_i); }
		if (m_rx0_q) { iio_channel_disable(m_rx0_q); }
		if (m_tx0_i) { iio_channel_disable(m_tx0_i); }
		if (m_tx0_q) { iio_channel_disable(m_tx0_q); }

		printf("* Destroying context\n");
		if (m_ctx) { iio_context_destroy(m_ctx); }
	}
}

