#include "dvbs2_rx.h"

static RxFrame *m_d;

void (*deinterleaver)(float *m);


int m_index;
int m_index_max;

void deinterleave_0_normal(float *m){
	m_d[m_index].m = m[0];
	m_index++;
}
void deinterleave_0_medium(float *m){
	m_d[m_index].m = m[0];
	m_index++;
}
void deinterleave_0_short(float *m){
	m_d[m_index].m = m[0];
	m_index++;

}
void deinterleave_00_normal(float *m){
	m_d[m_index++].m = m[0];
	m_d[m_index++].m = m[1];
}
void deinterleave_00_medium(float *m){
	m_d[m_index++].m = m[0];
	m_d[m_index++].m = m[1];
}
void deinterleave_00_short(float *m){
	m_d[m_index++].m = m[0];
	m_d[m_index++].m = m[1];
}

void deinterleave_012_normal(float *m){
	m_d[m_index].m         = m[0];
	m_d[m_index + 21600].m = m[1];
	m_d[m_index + 43200].m = m[2];
	m_index++;
}
void deinterleave_012_short(float *m){
	m_d[m_index].m         = m[0];
	m_d[m_index + 5400].m  = m[1];
	m_d[m_index + 10800].m = m[2];
	m_index++;
}

void deinterleave_102_normal(float *m){
	m_d[m_index].m         = m[1];
	m_d[m_index + 21600].m = m[0];
	m_d[m_index + 43200].m = m[2];
	m_index++;
}
void deinterleave_102_short(float *m){
	m_d[m_index].m         = m[1];
	m_d[m_index + 5400].m  = m[0];
	m_d[m_index + 10800].m = m[2];
	m_index++;
}

void deinterleave_210_normal(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 21600].m = m[1];
	m_d[m_index + 43200].m = m[0];
	m_index++;
}
void deinterleave_210_short(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 5400].m  = m[1];
	m_d[m_index + 10800].m = m[0];
	m_index++;
}

void deinterleave_0123_normal(float *m){
	m_d[m_index].m         = m[0];
	m_d[m_index + 16200].m = m[1];
	m_d[m_index + 32400].m = m[2];
	m_d[m_index + 48600].m = m[3];
	m_index++;
}
void deinterleave_0123_short(float *m){
	m_d[m_index].m         = m[0];
	m_d[m_index + 4050].m  = m[1];
	m_d[m_index + 8100].m  = m[2];
	m_d[m_index + 12150].m = m[3];
	m_index++;
}

void deinterleave_0321_normal(float *m){
	m_d[m_index].m         = m[0];
	m_d[m_index + 16200].m = m[3];
	m_d[m_index + 32400].m = m[2];
	m_d[m_index + 48600].m = m[1];
	m_index++;
}

void deinterleave_2103_normal(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 16200].m = m[1];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 48600].m = m[3];
	m_index++;
}
void deinterleave_2103_short(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 4050].m  = m[1];
	m_d[m_index + 8100].m  = m[0];
	m_d[m_index + 12150].m = m[3];
	m_index++;
}

void deinterleave_2301_normal(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 16200].m = m[3];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 48600].m = m[1];
	m_index++;
}
void deinterleave_2310_normal(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 16200].m = m[3];
	m_d[m_index + 32400].m = m[1];
	m_d[m_index + 48600].m = m[0];
	m_index++;
}

void deinterleave_3201_normal(float *m){
	m_d[m_index].m         = m[3];
	m_d[m_index + 16200].m = m[2];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 48600].m = m[1];
	m_index++;
}
void deinterleave_3201_short(float *m){
	m_d[m_index].m         = m[3];
	m_d[m_index + 4050].m  = m[2];
	m_d[m_index + 8100].m  = m[0];
	m_d[m_index + 12150].m = m[1];
	m_index++;
}

void deinterleave_3210_normal(float *m){
	m_d[m_index].m         = m[3];
	m_d[m_index + 16200].m = m[2];
	m_d[m_index + 32400].m = m[1];
	m_d[m_index + 48600].m = m[0];
	m_index++;
}

void deinterleave_3012_normal(float *m){
	m_d[m_index].m         = m[3];
	m_d[m_index + 16200].m = m[0];
	m_d[m_index + 32400].m = m[1];
	m_d[m_index + 48600].m = m[2];
	m_index++;
}

void deinterleave_3021_normal(float *m){
	m_d[m_index].m         = m[3];
	m_d[m_index + 16200].m = m[0];
	m_d[m_index + 32400].m = m[2];
	m_d[m_index + 48600].m = m[1];
	m_index++;
}

void deinterleave_2130_normal(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 16200].m = m[1];
	m_d[m_index + 32400].m = m[3];
	m_d[m_index + 48600].m = m[0];
	m_index++;
}
void deinterleave_2130_short(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 4050].m  = m[1];
	m_d[m_index + 8100].m  = m[3];
	m_d[m_index + 12150].m = m[0];
	m_index++;
}

void deinterleave_01234_normal(float *m){
	m_d[m_index].m = m[0];
	m_d[m_index + 12960].m = m[1];
	m_d[m_index + 25920].m = m[2];
	m_d[m_index + 38880].m = m[3];
	m_d[m_index + 51840].m = m[4];
	m_index++;
}
void deinterleave_01234_short(float *m){
	m_d[m_index].m         = m[0];
	m_d[m_index + 3240].m  = m[1];
	m_d[m_index + 6480].m  = m[2];
	m_d[m_index + 9720].m  = m[3];
	m_d[m_index + 12960].m = m[4];
	m_index++;
}

void deinterleave_21430_normal(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 12960].m = m[1];
	m_d[m_index + 25920].m = m[4];
	m_d[m_index + 38880].m = m[3];
	m_d[m_index + 51840].m = m[0];
	m_index++;
}

void deinterleave_40312_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 12960].m = m[0];
	m_d[m_index + 25920].m = m[3];
	m_d[m_index + 38880].m = m[1];
	m_d[m_index + 51840].m = m[2];
	m_index++;
}

void deinterleave_40213_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 12960].m = m[0];
	m_d[m_index + 25920].m = m[2];
	m_d[m_index + 38880].m = m[1];
	m_d[m_index + 51840].m = m[3];
	m_index++;
}

void deinterleave_41230_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 12960].m = m[1];
	m_d[m_index + 25920].m = m[2];
	m_d[m_index + 38880].m = m[3];
	m_d[m_index + 51840].m = m[0];
	m_index++;
}
void deinterleave_41230_short(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 3240].m  = m[1];
	m_d[m_index + 6480].m  = m[2];
	m_d[m_index + 9720].m  = m[3];
	m_d[m_index + 12960].m = m[0];
	m_index++;
}

void deinterleave_10423_normal(float *m){
	m_d[m_index].m         = m[1];
	m_d[m_index + 12960].m = m[0];
	m_d[m_index + 25920].m = m[4];
	m_d[m_index + 38880].m = m[2];
	m_d[m_index + 51840].m = m[3];
	m_index++;
}
void deinterleave_10423_short(float *m){
	m_d[m_index].m         = m[1];
	m_d[m_index + 3240].m  = m[0];
	m_d[m_index + 6480].m  = m[4];
	m_d[m_index + 9720].m  = m[2];
	m_d[m_index + 12960].m = m[3];
	m_index++;
}

void deinterleave_201543_normal(float *m){
	m_d[m_index].m         = m[2];
	m_d[m_index + 10800].m = m[0];
	m_d[m_index + 21600].m = m[1];
	m_d[m_index + 32400].m = m[5];
	m_d[m_index + 43200].m = m[4];
	m_d[m_index + 54000].m = m[3];
	m_index++;
}

void deinterleave_124053_normal(float *m){
	m_d[m_index].m         = m[1];
	m_d[m_index + 10800].m = m[2];
	m_d[m_index + 21600].m = m[4];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 43200].m = m[5];
	m_d[m_index + 54000].m = m[3];
	m_index++;
}

void deinterleave_421053_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 10800].m = m[2];
	m_d[m_index + 21600].m = m[1];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 43200].m = m[5];
	m_d[m_index + 54000].m = m[3];
	m_index++;
}

void deinterleave_305214_normal(float *m){
	m_d[m_index].m         = m[3];
	m_d[m_index + 10800].m = m[0];
	m_d[m_index + 21600].m = m[5];
	m_d[m_index + 32400].m = m[2];
	m_d[m_index + 43200].m = m[1];
	m_d[m_index + 54000].m = m[4];
	m_index++;
}

void deinterleave_520143_normal(float *m){
	m_d[m_index].m         = m[5];
	m_d[m_index + 10800].m = m[2];
	m_d[m_index + 21600].m = m[0];
	m_d[m_index + 32400].m = m[1];
	m_d[m_index + 43200].m = m[4];
	m_d[m_index + 54000].m = m[3];
	m_index++;
}

// Needs looking at as 7 prime (see note in spec)

void deinterleave_4250316_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 10800].m = m[2];
	m_d[m_index + 21600].m = m[5];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 43200].m = m[3];
	m_d[m_index + 54000].m = m[1];
	m_d[m_index + 54000].m = m[6];
	m_index++;
}
void deinterleave_4130256_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 10800].m = m[1];
	m_d[m_index + 21600].m = m[3];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 43200].m = m[2];
	m_d[m_index + 54000].m = m[5];
	m_d[m_index + 54000].m = m[6];
	m_index++;
}

void deinterleave_40372156_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 8100].m  = m[0];
	m_d[m_index + 16200].m = m[3];
	m_d[m_index + 24300].m = m[7];
	m_d[m_index + 32400].m = m[2];
	m_d[m_index + 40500].m = m[1];
	m_d[m_index + 48600].m = m[5];
	m_d[m_index + 56700].m = m[6];
	m_index++;
}
void deinterleave_01234567_normal(float *m){
	m_d[m_index].m = m[0];
	m_d[m_index + 8100].m = m[1];
	m_d[m_index + 16200].m = m[2];
	m_d[m_index + 24300].m = m[3];
	m_d[m_index + 32400].m = m[4];
	m_d[m_index + 40500].m = m[5];
	m_d[m_index + 48600].m = m[6];
	m_d[m_index + 56700].m = m[7];
	m_index++;
}
void deinterleave_46320571_normal(float *m){
	m_d[m_index].m         = m[4];
	m_d[m_index + 8100].m  = m[6];
	m_d[m_index + 16200].m = m[3];
	m_d[m_index + 24300].m = m[2];
	m_d[m_index + 32400].m = m[0];
	m_d[m_index + 40500].m = m[5];
	m_d[m_index + 48600].m = m[7];
	m_d[m_index + 56700].m = m[1];
	m_index++;
}
void deinterleave_75642301_normal(float *m){
	m_d[m_index].m         = m[7];
	m_d[m_index + 8100].m  = m[5];
	m_d[m_index + 16200].m = m[6];
	m_d[m_index + 24300].m = m[4];
	m_d[m_index + 32400].m = m[2];
	m_d[m_index + 40500].m = m[3];
	m_d[m_index + 48600].m = m[0];
	m_d[m_index + 56700].m = m[1];
	m_index++;
}
void deinterleave_50743612_normal(float *m){
	m_d[m_index].m         = m[5];
	m_d[m_index + 8100].m  = m[0];
	m_d[m_index + 16200].m = m[7];
	m_d[m_index + 24300].m = m[4];
	m_d[m_index + 32400].m = m[3];
	m_d[m_index + 40500].m = m[6];
	m_d[m_index + 48600].m = m[1];
	m_d[m_index + 56700].m = m[2];
	m_index++;
}
//
// New frame set the de-interleaver to use
//
void set_deinterleaver(InterleaveType type){
	switch (type){
	case I_0_N:
		deinterleaver = &deinterleave_0_normal;
		m_index_max = 64800;
		break;
	case I_0_M:
		deinterleaver = &deinterleave_0_medium;
		m_index_max = 32400;
		break;
	case I_0_S:
		deinterleaver = &deinterleave_0_short;
		m_index_max = 16200;
		break;
	case I_00_N:
		deinterleaver = &deinterleave_0_normal;
		m_index_max = 64800;
		break;
	case I_00_M:
		deinterleaver = &deinterleave_00_medium;
		m_index_max = 32400;
		break;
	case I_00_S:
		deinterleaver = &deinterleave_00_short;
		m_index_max = 16200;
		break;
	case I_012_N:
		deinterleaver = &deinterleave_012_normal;
		m_index_max = 21600;
		break;
	case I_012_S:
		deinterleaver = &deinterleave_012_short;
		m_index_max = 5400;
		break;
	case I_102_N:
		deinterleaver = &deinterleave_102_normal;
		m_index_max = 21600;
		break;
	case I_102_S:
		deinterleaver = &deinterleave_102_short;
		m_index_max = 5400;
		break;
	case I_210_N:
		deinterleaver = &deinterleave_210_normal;
		m_index_max = 21600;
		break;
	case I_210_S:
		deinterleaver = &deinterleave_210_short;
		m_index_max = 5400;
		break;
	case I_0123_N:
		deinterleaver = &deinterleave_0123_normal;
		m_index_max = 16200;
		break;
	case I_0123_S:
		deinterleaver = &deinterleave_0123_short;
		m_index_max = 4050;
		break;
	case I_0321_N:
		deinterleaver = &deinterleave_0321_normal;
		m_index_max = 16200;
		break;
	case I_2103_N:
		deinterleaver = &deinterleave_2103_normal;
		m_index_max = 16200;
		break;
	case I_2103_S:
		deinterleaver = &deinterleave_2103_short;
		m_index_max = 4050;
		break;
	case I_2130_S:
		deinterleaver = &deinterleave_2130_short;
		m_index_max = 4050;
		break;
	case I_2301_N:
		deinterleaver = &deinterleave_2301_normal;
		m_index_max = 16200;
		break;
	case I_2310_N:
		deinterleaver = &deinterleave_2310_normal;
		m_index_max = 16200;
		break;
	case I_3201_N:
		deinterleaver = &deinterleave_3201_normal;
		m_index_max = 16200;
		break;
	case I_3201_S:
		deinterleaver = &deinterleave_3201_short;
		m_index_max = 4050;
		break;
	case I_3210_N:
		deinterleaver = &deinterleave_3210_normal;
		m_index_max = 16200;
		break;
	case I_3012_N:
		deinterleaver = &deinterleave_3012_normal;
		m_index_max = 16200;
		break;
	case I_3021_N:
		deinterleaver = &deinterleave_3021_normal;
		m_index_max = 16200;
		break;
	case I_01234_N:
		deinterleaver = &deinterleave_01234_normal;
		m_index_max = 12960;
		break;
	case I_01234_S:
		deinterleaver = &deinterleave_01234_short;
		m_index_max = 3240;
		break;
	case I_21430_N:
		deinterleaver = &deinterleave_21430_normal;
		m_index_max = 12960;
		break;
	case I_40312_N:
		deinterleaver = &deinterleave_40312_normal;
		m_index_max = 12960;
		break;
	case I_40213_N:
		deinterleaver = &deinterleave_40213_normal;
		m_index_max = 12960;
		break;
	case I_41230_N:
		deinterleaver = &deinterleave_41230_normal;
		m_index_max = 12960;
		break;
	case I_41230_S:
		deinterleaver = &deinterleave_41230_short;
		m_index_max = 3240;
		break;
	case I_10423_N:
		deinterleaver = &deinterleave_10423_normal;
		m_index_max = 12960;
		break;
	case I_10423_S:
		deinterleaver = &deinterleave_10423_short;
		m_index_max = 3240;
		break;
	case I_201543_N:
		deinterleaver = &deinterleave_201543_normal;
		m_index_max = 10800;
		break;
	case I_124053_N:
		deinterleaver = &deinterleave_124053_normal;
		m_index_max = 10800;
		break;
	case I_421053_N:
		deinterleaver = &deinterleave_124053_normal;
		m_index_max = 10800;
		break;
	case I_305214_N:
		deinterleaver = &deinterleave_305214_normal;
		m_index_max = 10800;
		break;
	case I_520143_N:
		deinterleaver = &deinterleave_520143_normal;
		m_index_max = 10800;
		break;
	case I_4130256_N:
		deinterleaver = &deinterleave_4130256_normal;
		m_index_max = 8100;
		break;
	case I_4250316_N:
		deinterleaver = &deinterleave_4250316_normal;
		m_index_max = 10800;
		break;
	case I_40372156_N:
		deinterleaver = &deinterleave_40372156_normal;
		m_index_max = 8100;
		break;
	case I_01234567_N:
		deinterleaver = &deinterleave_01234567_normal;
		m_index_max = 8100;
		break;
	case I_46320571_N:
		deinterleaver = &deinterleave_46320571_normal;
		m_index_max = 8100;
		break;
	case I_75642301_N:
		deinterleaver = &deinterleave_75642301_normal;
		m_index_max = 8100;
		break;
	case I_50743612_N:
		deinterleaver = &deinterleave_50743612_normal;
		m_index_max = 8100;
		break;
	default:
		break;
	}
}

//
// De - interleave a new frame
// results placed back in the RxFrame structure
//

void deinterleaver_new_frame(RxFrame *r, RxFormat *f){
	m_d = r;
	set_deinterleaver(f->itype);
	m_index = 0;
}

