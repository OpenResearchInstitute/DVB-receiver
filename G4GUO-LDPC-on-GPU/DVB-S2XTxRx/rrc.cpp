#define USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include "dvbs2_rx.h"

static short *m_filter = NULL;
//
// Create a single oversampled Root raised cosine filter
//
void build_rrc_filter(float *filter, float rolloff, int ntaps, int samples_per_symbol) {
	double a, b, c, d;
	double B = rolloff+0.0001;// Rolloff factor .0001 stops infinite filter coefficient with 0.25 rolloff
	double t = -(ntaps - 1) / 2;// First tap
	double Ts = samples_per_symbol;// Samples per symbol
	// Create the filter
	for (int i = 0; i < (ntaps); i++) {
		a = 2.0 * B / (M_PI*sqrt(Ts));
		b = cos((1.0 + B)*M_PI*t / Ts);
		// Check for infinity in calculation (computers don't have infinite precision)
		if (t == 0)
			c = (1.0 - B)*M_PI / (4 * B);
		else
			c = sin((1.0 - B)*M_PI*t / Ts) / (4.0*B*t / Ts);

		d = (1.0 - (4.0*B*t / Ts)*(4.0*B*t / Ts));
		//filter[i] = (b+c)/(a*d);//beardy
		filter[i] = (float)(a*(b + c) / d);//nasa
		t = t + 1.0;
	}
}
void set_filter_gain(float *filter, float gain, int ntaps) {
	float sum = 0;
	for (int i = 0; i < ntaps; i++) {
		sum += filter[i];
	}
	gain = gain / sum;
	for (int i = 0; i < ntaps; i++) {
		filter[i] = filter[i] * gain;
	}
}
void window_filter(float *filter, int N) {
	// Build the window
	for (int i = -N / 2, j = 0; i < N / 2; i++, j++)
	{
		filter[j] = (0.5f*(1.0f + cosf((2.0f*(float)M_PI*i) / N)))*filter[j];
	}
}
void make_short(short *out, float *in, int len) {
	for (int i = 0; i < len; i++) {
		out[i] = (short)(in[i] * 32768);
	}
}

//
// the length must always be even and a multiple of 16
//
short *rrc_make_filter(float roff, int ratio, int taps) {
	// Create the over sampled mother filter
	float *filter = (float*)malloc(sizeof(float)*taps);
	// Set last coefficient to zero
	filter[taps-1] = 0;
	// RRC filter must always be odd length
	build_rrc_filter(filter, roff, taps-1, ratio);
	window_filter(filter, taps-1);
	set_filter_gain(filter, 0.999f, taps);
	// Free memory from last filter if it exsists
	if (m_filter != NULL) free(m_filter);
	// Allocate memory for new global filter
	m_filter = (short*)malloc(sizeof(short)*taps);
	// convert the filter into shorts
	make_short(m_filter, filter, taps);
	free(filter);
	return m_filter;
}
