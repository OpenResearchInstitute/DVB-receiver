/*
 * zigzag.cpp
 *
 *  Created on: 14 Mar 2018
 *      Author: Charles
 */
#define USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "dvbs2_rx.h"

extern RxFormat g_format;

//
// Zigzag the tuning frequency while hunting for a preamble
//

double m_acc;
double m_inc;
uint16_t   m_max;
uint16_t   m_idx;

void zigzag_reset(void){
	m_acc = 0;
	m_idx = 0;
}

double zigzag_delta(void){
	m_idx++;
	m_acc = m_inc * m_idx;
	if(m_idx > m_max ){
		m_idx = 0;
		m_inc = - m_inc;
	}
//	printf("%f\n",m_acc);
	return m_acc;
}

void zigzag_set_inc_and_max(double freq, uint16_t max){

	m_inc = 2*M_PI*freq/g_format.req_syrate;
	m_max = max;
}


