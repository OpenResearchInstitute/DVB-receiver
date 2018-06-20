#pragma once
#include "dvbs2_rx.h"
#define FLT double

FLT noise_get_sn(void);
int noise_is_enabled(void);
void noise_on(void);
void noise_off(void);
void noise_init(void);
void noise_set_es_no(FLT sn);
float noise_add(FComplex *s, int len);
