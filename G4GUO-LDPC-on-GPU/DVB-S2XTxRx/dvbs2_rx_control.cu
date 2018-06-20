#include <stdio.h>
#include "dvbs2_rx.h"

extern RxFormat g_format;

// Could do it as a constant table

const int params[68][9]={
{0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,0,0,0,0},
{ frame_NORMAL, m_QPSK, cR_1_4,    BCH_N12, 135, 16008, 16200, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_1_4,    BCH_S12, 36,  3072,  3240,  8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_1_3,    BCH_N12, 120, 21408, 21600, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_1_3,    BCH_S12, 30,  5232,  5400,  8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_2_5,    BCH_N12, 108, 25728, 25920, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_2_5,    BCH_S12, 27,  6312,  6480,  8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_1_2,    BCH_N12, 90,  32208, 32400, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_1_2,    BCH_S12, 25,  7032,  7200,  8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_3_5,    BCH_N12, 72,  38688, 38880, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_3_5,    BCH_S12, 18,  9552,  9720,  8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_2_3,    BCH_N10, 60,  43040, 43200, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_2_3,    BCH_S12, 15,  10632, 10800, 8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_3_4,    BCH_N12, 45,  48408, 48600, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_3_4,    BCH_S12, 12,  11712, 11880, 8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_4_5,    BCH_N12, 36,  51648, 51840, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_4_5,    BCH_S12, 10,  12432, 12600, 8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_5_6,    BCH_N10, 30,  53840, 54000, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_5_6,    BCH_S12, 8,   13152, 13320, 8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_8_9,    BCH_N8,  20,  57472, 57600, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_8_9,    BCH_S12, 5,   14232, 14400, 8100,  I_00_S    },
{ frame_NORMAL, m_QPSK, cR_9_10,   BCH_N8,  18,  58192, 58320, 32400, I_00_N    },
{ frame_SHORT,  m_QPSK, cR_9_10,   BCH_S12, 0,   0,     0,     0,     I_00_S    },// Invalid
{ frame_NORMAL, m_8PSK, cR_3_5,    BCH_N12, 72,  38688, 38880, 21600, I_210_N   },
{ frame_SHORT,  m_8PSK, cR_3_5,    BCH_S12, 18,  9552,  9720,  5400,  I_210_S   },
{ frame_NORMAL, m_8PSK, cR_2_3,    BCH_N10, 60,  43040, 43200, 21600, I_012_N   },
{ frame_SHORT,  m_8PSK, cR_2_3,    BCH_S12, 15,  10632, 10800, 5400,  I_012_S   },
{ frame_NORMAL, m_8PSK, cR_3_4,    BCH_N12, 45,  48408, 48600, 21600, I_012_N   },
{ frame_SHORT,  m_8PSK, cR_3_4,    BCH_S12, 12,  11712, 11880, 5400,  I_012_S   },
{ frame_NORMAL, m_8PSK, cR_5_6,    BCH_N10, 30,  53840, 54000, 21600, I_012_N   },
{ frame_SHORT,  m_8PSK, cR_5_6,    BCH_S12, 8,   13152, 13200, 5400,  I_012_S   },
{ frame_NORMAL, m_8PSK, cR_8_9,    BCH_N8,  20,  57472, 57600, 21600, I_012_N   },
{ frame_SHORT,  m_8PSK, cR_8_9,    BCH_S12, 5,   14232, 14400, 5400,  I_012_S   },
{ frame_NORMAL, m_8PSK, cR_9_10,   BCH_N8,  18,  58192, 58320, 21600, I_012_N   },
{ frame_SHORT,  m_16APSK, cR_2_3,  BCH_S12, 15,  10632, 10800, 4050,  I_0123_S  },
{ frame_NORMAL, m_16APSK, cR_2_3,  BCH_N10, 60,  43040, 43200, 16200, I_0123_N  },
{ frame_SHORT,  m_8PSK, cR_9_10,   BCH_S12, 0,   0,     0,     5400,  I_012_S   },// Invalid
{ frame_NORMAL, m_16APSK, cR_3_4,  BCH_N12, 45,  48408, 48600, 16200, I_0123_N  },
{ frame_SHORT,  m_16APSK, cR_3_4,  BCH_S12, 12,  11712, 11880, 4050,  I_0123_S  },
{ frame_NORMAL, m_16APSK, cR_4_5,  BCH_N12, 36,  16008, 16200, 16200, I_0123_N  },
{ frame_SHORT,  m_16APSK, cR_4_5,  BCH_S12, 10,  12432, 12600, 4050,  I_0123_S  },
{ frame_NORMAL, m_16APSK, cR_5_6,  BCH_N10, 30,  53840, 54000, 16200, I_0123_N  },
{ frame_SHORT,  m_16APSK, cR_5_6,  BCH_S12, 8,   13152, 13200, 4050,  I_0123_S  },
{ frame_NORMAL, m_16APSK, cR_8_9,  BCH_N8,  20,  57472, 57600, 16200, I_0123_N  },
{ frame_SHORT,  m_16APSK, cR_8_9,  BCH_S12, 5,   14232, 14400, 4050,  I_0123_S  },
{ frame_NORMAL, m_16APSK, cR_9_10, BCH_N8,  18,  58192, 58320, 16200, I_0123_N  },
{ frame_SHORT,  m_16APSK, cR_9_10, BCH_S12, 0,   0,     0,     4050,  I_0123_S  },// Invalid
{ frame_NORMAL, m_32APSK, cR_3_4,  BCH_N12, 45,  48408, 48600, 12960, I_01234_N },
{ frame_SHORT,  m_32APSK, cR_3_4,  BCH_S12, 12,  11712, 11880, 3240,  I_01234_S },
{ frame_NORMAL, m_32APSK, cR_4_5,  BCH_N12, 36,  16008, 16200, 12960, I_01234_N },
{ frame_SHORT,  m_32APSK, cR_4_5,  BCH_S12, 10,  12432, 12600, 3240,  I_01234_S },
{ frame_NORMAL, m_32APSK, cR_5_6,  BCH_N10, 30,  53840, 54000, 12960, I_01234_N },
{ frame_SHORT,  m_32APSK, cR_5_6,  BCH_S12, 8,   13152, 13200, 3240,  I_01234_S },
{ frame_NORMAL, m_32APSK, cR_8_9,  BCH_N8,  20,  57472, 57600, 12960, I_01234_N },
{ frame_SHORT,  m_32APSK, cR_8_9,  BCH_S12, 5,   14232, 14400, 3240,  I_01234_S },
{ frame_NORMAL, m_32APSK, cR_9_10, BCH_N8,  18,  58192, 58320, 12960, I_01234_N },
{ frame_SHORT,  m_32APSK, cR_9_10, BCH_S12, 0,   0,     0,     3240,  I_01234_S },// Invalid

};

const char cformat[32][256]={
		{"Dummy "},
		{"QPSK 1/4 "},
		{"QPSK 1/3 "},
		{"QPSK 2/5 "},
		{"QPSK 1/2 "},
		{"QPSK 3/5 "},
		{"QPSK 2/3 "},
		{"QPSK 3/4 "},
		{"QPSK 4/5 "},
		{"QPSK 5/6 "},
		{"QPSK 8/9 "},
		{"QPSK 9/10 "},
		{"8PSK 3/5 "},
		{"8PSK 2/3 "},
		{"8PSK 3/4 "},
		{"8PSK 5/6 "},
		{"8PSK 8/9 "},
		{"8PSK 9/10 "},
		{"16APSK 2/3 "},
		{"16APSK 3/4 "},
		{"16APSK 4/5 "},
		{"16APSK 5/6 "},
		{"16APSK 8/9 "},
		{"16APSK 9/10 "},
		{"32APSK 3/4 "},
		{"32APSK 4/5 "},
		{"32APSK 5/6 "},
		{"32APSK 8/9 "},
		{"32APSK 9/10 "},
		{"Reserved "},
		{"Reserved "},
		{"Reserved "}
};

void modcod_S2_trace(void){
	if(g_format.modcod < 128 ){
		sprintf(g_format.format_text,"DVB-S2 %s",cformat[g_format.modcod>>2]);
		if(g_format.modcod&2)
			strcat(g_format.format_text,"S ");
		else
			strcat(g_format.format_text,"N ");

		if(g_format.modcod&1) strcat(g_format.format_text,"P");

	}
}

//
// Set a new mode
//
// Only DVB-S2 Normal frames are supported at the moment
//
//
ModcodStatus modcod_decode( uint8_t code ){
	
	ModcodStatus status = MODCOD_OK;
	int npreams = 0; // Number of symbols in the preamble

	g_format.modcod = code;
	npreams = 90;

	if (code & 1)
		g_format.pilots = 1;
	else
		g_format.pilots = 0;

	if ((code>>1) < 58){
		// DVB-S2
		g_format.s2_type = DVB_S2;


		if (code & 2)
			g_format.nldpc = SHORT_FRAME_BITS;
		else
			g_format.nldpc = NORMAL_FRAME_BITS;

		code = code>>1;

		g_format.frame_type = (FrameType)params[code][0];
		g_format.mod        = (Modulation)params[code][1];
		g_format.mod_class  = (Modulation)params[code][1];
		g_format.code_rate  = (CodeRate)params[code][2];
		g_format.bch        = (BchTypes)params[code][3];
		g_format.q          = params[code][4];
		g_format.kbch       = params[code][5];
		g_format.nbch       = params[code][6];
		g_format.nsyms      = params[code][7];// Number of information carrying symbols
		if(g_format.pilots){
			if((g_format.nsyms%1440)){
				// We don't send the pilot block at the end of the frame
		        g_format.bsyms = g_format.nsyms + (((g_format.nsyms/1440))*36);
			}else{
			    // Exact whole number so no pilot block required at the end
			    g_format.bsyms = g_format.nsyms + (((g_format.nsyms/1440)-1)*36);
			}
	    }
		else{
			// No pilots
			g_format.bsyms  = g_format.nsyms;// Number of symbols in the body
		}
		g_format.itype      = (InterleaveType)params[code][8];
		g_format.nuldpc     = g_format.nbch;
		g_format.pbch       = g_format.nbch - g_format.kbch;
		g_format.pldpc      = g_format.nldpc- g_format.nuldpc;
		g_format.fsyms      = (npreams + g_format.bsyms);// Total number of symbols in a frame
		g_format.nsams      = g_format.fsyms*SAMPLES_PER_SYMBOL;

		if(g_format.nbch == 0 ) status = MODCOD_ERROR;
		modcod_S2_trace();
		return status;
	}
	else
	{
return MODCOD_ERROR;
		// DVB-S2X
		g_format.s2_type = DVB_S2X;

		g_format.frame_type = frame_UNKNOWN;
        g_format.pilots = code&1;
		// Blank off the 1 LSB as not used by the switch
		switch (code & 0xFFFFFFFE){
			// S2X codes
		case 129:
			//129 VL SNR set1 See Table 18a
			g_format.s2_type = DVB_S2XVLSNR;
			status = MODCOD_TYPE1;
			npreams += 900;
			break;
		case 131:
			//131 VL SNR set2 See Table 18a
			g_format.s2_type = DVB_S2XVLSNR;
			status = MODCOD_TYPE2;
			npreams += 900;
			break;
		case 132:
			//132 QPSK 13 / 45 QPSK 13 / 45 Normal
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_13_45;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_00_N;
			g_format.nsyms = 64800/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 13/45 N ");
			break;
		case 134:
			//134 QPSK 9 / 20 QPSK 9 / 20 Normal
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_9_20;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_00_N;
			g_format.nsyms = 64800/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 9/20 N ");
			break;
		case 136:
			//136 QPSK 11 / 20 QPSK 11 / 20 Normal
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_11_20;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_00_N;
			g_format.nsyms = 64800/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 11/20 N ");
			break;
		case 138:
			//138 8APSK 5 / 9 - L 2 + 4 + 2APSK 100 / 180 Normal
			g_format.mod = m_2_4_2APSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_100_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_012_N;
			g_format.nsyms = 64800/3;
			snprintf(g_format.format_text,80,"DVB-S2X 2_4_2APSK 100/180 N ");
			break;
		case 140:
			//140 8APSK 26 / 45 - L 2 + 4 + 2APSK 104 / 180 Normal
			g_format.mod = m_2_4_2APSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_104_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_012_N;
			g_format.nsyms = 64800/3;
			snprintf(g_format.format_text,80,"DVB-S2X 2_4_2APSK 104/180 N ");
			break;
		case 142:
			//142 8PSK 23 / 36 8PSK 23 / 36 Normal
			g_format.mod = m_8PSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_23_36;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_012_N;
			g_format.nsyms = 64800/3;
			snprintf(g_format.format_text,80,"DVB-S2X 8PSK 23/36 N ");
			break;
		case 144:
			//144 8PSK 25 / 36 8PSK 25 / 36 Normal
			g_format.mod = m_8PSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_25_36;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_102_N;
			g_format.nsyms = 64800/3;
			snprintf(g_format.format_text,80,"DVB-S2X 8PSK 25/36 N ");
			break;
		case 146:
			//146 8PSK 13 / 18 8PSK 13 / 18 Normal
			g_format.mod = m_8PSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_13_18;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_102_N;
			g_format.nsyms = 64800/3;
			snprintf(g_format.format_text,80,"DVB-S2X 8PSK 13/18 N ");
			break;
		case 148:
			//148 16APSK 1 / 2 - L 8 + 8APSK 90 / 180 Normal
			g_format.mod = m_8_8APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_90_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_3210_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 8APSK 90/100 N ");
			break;
		case 150:
			//150 16APSK 8 / 15 - L 8 + 8APSK 96 / 180 Normal
			g_format.mod = m_8_8APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_96_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_2310_N;
			g_format.nsyms = 64800/4;
			break;
		case 152:
			//152 16APSK 5 / 9 - L 8 + 8APSK 100 / 180 Normal
			g_format.mod = m_8_8APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_100_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_2301_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 8_8PSK 100/180 N ");
			break;
		case 154:
			//154 16APSK 26 / 45 4 + 12APSK 26 / 45 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_26_45;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_3201_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 26/45 N ");
			break;
		case 156:
			//156 16APSK 3 / 5 4 + 12APSK 3 / 5 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_3_5;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_3210_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 3/5 N ");
			break;
		case 158:
			//158 16APSK 3 / 5 - L 8 + 8APSK 18 / 30 Normal
			g_format.mod = m_8_8APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_18_30;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_0123_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 8_8APSK 18/30 N ");
			break;
		case 160:
			//160 16APSK 28 / 45 4 + 12APSK 28 / 45 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_28_45;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_3012_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 28/45 N ");
			break;
		case 162:
			//162 16APSK 23 / 36 4 + 12APSK 23 / 36 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_23_36;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_3021_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 23/36 N ");
			break;
		case 164:
			//164 16APSK 2 / 3 - L 8 + 8APSK 20 / 30 Normal
			g_format.mod = m_8_8APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_20_30;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_0123_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 8_8APSK 20/30 N ");
			break;
		case 166:
			//166 16APSK 25 / 36 4 + 12APSK 25 / 36 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_25_36;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_2310_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 25/36 N ");
			break;
		case 168:
			//168 16APSK 13 / 18 4 + 12APSK 13 / 18 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_13_18;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_3021_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 13/18 N ");
			break;
		case 170:
			//170 16APSK 7 / 9 4 + 12APSK 140 / 180 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_140_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_3210_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 140/180 N ");
			break;
		case 172:
			//172 16APSK 77 / 90 4 + 12APSK 154 / 180 Normal
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_154_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_0321_N;
			g_format.nsyms = 64800/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 154/180 N ");
			break;
		case 174:
			//174 32APSK 2 / 3 - L 4 + 12 + 16rbAPSK 2 / 3 Normal
			g_format.mod = m_4_12_16rbAPSK;
			g_format.mod_class = m_32APSK;
			g_format.code_rate = cR_2_3;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_21430_N;
			g_format.nsyms = 64800/5;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12_16rbAPSK 2/3 N ");
			break;
		case 178:
			//178 32APSK 32 / 45 4 + 8 + 4 + 16APSK 128 / 180 Normal
			g_format.mod = m_4_8_4_16APSK;
			g_format.mod_class = m_32APSK;
			g_format.code_rate = cR_128_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_40312_N;
			g_format.nsyms = 64800/5;
			snprintf(g_format.format_text,80,"DVB-S2X 4_8_4_16APSK 128/180 N ");
			break;
		case 180:
			//180 32APSK 11 / 15 4 + 8 + 4 + 16APSK 132 / 180 Normal
			g_format.mod = m_4_8_4_16APSK;
			g_format.mod_class = m_32APSK;
			g_format.code_rate = cR_132_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_40312_N;
			g_format.nsyms = 64800/5;
			snprintf(g_format.format_text,80,"DVB-S2X 4_8_4_16APSK 132/180 N ");
			break;
		case 182:
			//182 32APSK 7 / 9 4 + 8 + 4 + 16APSK 140 / 180 Normal
			g_format.mod = m_4_8_4_16APSK;
			g_format.mod_class = m_32APSK;
			g_format.code_rate = cR_140_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_40213_N;
			g_format.nsyms = 64800/5;
			snprintf(g_format.format_text,80,"DVB-S2X 4_8_4_16APSK 140/180 N ");
			break;
		case 184:
			//184 64APSK 32 / 45 - L 16 + 16 + 16 + 16APSK 128 / 180 Normal
			g_format.mod = m_16_16_16_16APSK;
			g_format.mod_class = m_64APSK;
			g_format.code_rate = cR_128_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_305214_N;
			g_format.nsyms = 64800/6;
			snprintf(g_format.format_text,80,"DVB-S2X 16_16_16_16APSK 128/180 N ");
			break;
		case 186:
			//186 64APSK 11 / 15 4 + 12 + 20 + 28APSK 132 / 180 Normal
			g_format.mod = m_4_12_20_28APSK;
			g_format.mod_class = m_64APSK;
			g_format.code_rate = cR_132_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_520143_N;
			g_format.nsyms = 64800/6;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12_20_28APSK 132/180 N ");
			break;
		case 190:
			//190 64APSK 7 / 9 8 + 16 + 20 + 20APSK 7 / 9 Normal
			g_format.mod = m_8_16_20_20APSK;
			g_format.mod_class = m_64APSK;
			g_format.code_rate = cR_7_9;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_201543_N;
			g_format.nsyms = 64800/6;
			snprintf(g_format.format_text,80,"DVB-S2X 8_16_20_28APSK 7/9 N ");
			break;
		case 194:
			//194 64APSK 4 / 5 8 + 16 + 20 + 20APSK 4 / 5 Normal
			g_format.mod = m_8_16_20_20APSK;
			g_format.mod_class = m_64APSK;
			g_format.code_rate = cR_4_5;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_124053_N;
			g_format.nsyms = 64800/6;
			snprintf(g_format.format_text,80,"DVB-S2X 8_16_20_20APSK 4/5 N ");
			break;
		case 198:
			//198 64APSK 5 / 6 8 + 16 + 20 + 20APSK 5 / 6 Normal
			g_format.mod = m_8_16_20_20APSK;
			g_format.mod_class = m_64APSK;
			g_format.code_rate = cR_5_6;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_421053_N;
			g_format.nsyms = 64800/6;
			snprintf(g_format.format_text,80,"DVB-S2X 8_16_20_20APSK 5/6 N ");
			break;
		case 200:
			//200 128APSK 3 / 4 128APSK 135 / 180 Normal
			g_format.mod = m_128APSK;
			g_format.mod_class = m_128APSK;
			g_format.code_rate = cR_135_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_4250316_N;
			g_format.nsyms = (64800/7)+1;
			snprintf(g_format.format_text,80,"DVB-S2X 128APSK 135/180 N ");
			break;
		case 202:
			//202 128APSK 7 / 9 128APSK 140 / 180 Normal
			g_format.mod = m_128APSK;
			g_format.mod_class = m_128APSK;
			g_format.code_rate = cR_140_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_4130256_N;
			snprintf(g_format.format_text,80,"DVB-S2X 128APSK 140/180 N ");
			g_format.nsyms = (64800/7)+1;
			break;
		case 204:
			//204 256APSK 29 / 45 - L 256APSK 116 / 180 Normal
			g_format.mod = m_256APSK;
			g_format.mod_class = m_256APSK;
			g_format.code_rate = cR_116_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_40372156_N;
			g_format.nsyms = 64800/8;
			snprintf(g_format.format_text,80,"DVB-S2X 256APSK 116/180 N ");
			break;
		case 206:
			//206 256APSK 2 / 3 - L 256APSK 20 / 30 Normal
			g_format.mod = m_256APSK;
			g_format.mod_class = m_256APSK;
			g_format.code_rate = cR_20_30;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_01234567_N;
			g_format.nsyms = 64800/8;
			snprintf(g_format.format_text,80,"DVB-S2X 256APSK 20/30 N ");
			break;
		case 208:
			//208 256APSK 31 / 45 - L 256APSK 124 / 180 Normal
			g_format.mod = m_256APSK;
			g_format.mod_class = m_256APSK;
			g_format.code_rate = cR_124_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_46320571_N;
			g_format.nsyms = 64800/8;
			snprintf(g_format.format_text,80,"DVB-S2X 256APSK 124/180 N ");
			break;
		case 210:
			//210 256APSK 32 / 45 256APSK 128 / 180 Normal
			g_format.mod = m_256APSK;
			g_format.mod_class = m_256APSK;
			g_format.code_rate = cR_128_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_75642301_N;
			g_format.nsyms = 64800/8;
			snprintf(g_format.format_text,80,"DVB-S2X 256APSK 128/180 N ");
			break;
		case 212:
			//212 256APSK 11 / 15 - L 256APSK 22 / 30 Normal
			g_format.mod = m_256APSK;
			g_format.mod_class = m_256APSK;
			g_format.code_rate = cR_22_30;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_01234567_N;
			g_format.nsyms = 64800/8;
			snprintf(g_format.format_text,80,"DVB-S2X 256APSK 22/30 N ");
			break;
		case 214:
			//214 256APSK 3 / 4 256APSK 135 / 180 Normal
			g_format.mod = m_256APSK;
			g_format.mod_class = m_256APSK;
			g_format.code_rate = cR_135_180;
			g_format.frame_type = frame_NORMAL;
			g_format.itype = I_50743612_N;
			g_format.nsyms = 64800/8;
			snprintf(g_format.format_text,80,"DVB-S2X 256APSK 135/180 N ");
			break;
		case 216:
			//216 QPSK 11 / 45 QPSK 11 / 45 Short
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_11_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_00_S;
			g_format.nsyms = 16200/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 11/45 S ");
			break;
		case 218:
			//218 QPSK 4 / 15 QPSK 4 / 15 Short
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_4_15;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_00_S;
			g_format.nsyms = 16200/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 4/15 S ");
			break;
		case 220:
			//220 QPSK 14 / 45 QPSK 14 / 45 Short
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_14_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_00_S;
			g_format.nsyms = 16200/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 14/45 S ");
			break;
		case 222:
			//222 QPSK 7 / 15 QPSK 7 / 15 Short
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_7_15;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_00_S;
			g_format.nsyms = 16200/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 7/15 S ");
			break;
		case 224:
			//224 QPSK 8 / 15 QPSK 8 / 15 Short
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_8_15;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_00_S;
			g_format.nsyms = 16200/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 8/15 S ");
			break;
		case 226:
			//226 QPSK 32 / 45 QPSK 32 / 45 Short
			g_format.mod = m_QPSK;
			g_format.mod_class = m_QPSK;
			g_format.code_rate = cR_32_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_00_S;
			g_format.nsyms = 16200/2;
			snprintf(g_format.format_text,80,"DVB-S2X QPSK 32/45 S ");
			break;
		case 228:
			//228 8PSK 7 / 15 8PSK 7 / 15 Short
			g_format.mod = m_8PSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_7_15;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_102_S;
			g_format.nsyms = 16200/2;
			snprintf(g_format.format_text,80,"DVB-S2X 8PSK 7/15 S ");
			break;
		case 230:
			//230 8PSK 8 / 15 8PSK 8 / 15 Short
			g_format.mod = m_8PSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_8_15;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_102_S;
			g_format.nsyms = 16200/3;
			snprintf(g_format.format_text,80,"DVB-S2X 8PSK 8/15 S ");
			break;
		case 232:
			//232 8PSK 26 / 45 8PSK 26 / 45 Short
			g_format.mod = m_8PSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_26_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_102_S;
			g_format.nsyms = 16200/3;
			snprintf(g_format.format_text,80,"DVB-S2X 8PSK 26/45 S ");
			break;
		case 234:
			//234 8PSK 32 / 45 8PSK 32 / 45 Short
			g_format.mod = m_8PSK;
			g_format.mod_class = m_8PSK;
			g_format.code_rate = cR_32_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_012_S;
			g_format.nsyms = 16200/3;
			snprintf(g_format.format_text,80,"DVB-S2X 8PSK 32/45 S ");
			break;
		case 236:
			//236 16APSK 7 / 15 4 + 12APSK 7 / 15 Short
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_7_15;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_2103_S;
			g_format.nsyms = 16200/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 7/15 S ");
			break;
		case 238:
			//238 16APSK 8 / 15 4 + 12APSK 8 / 15 Short
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_8_15;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_2103_S;
			g_format.nsyms = 16200/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 8/15 S ");
			break;
		case 240:
			//240 16APSK 26 / 45 4 + 12APSK 26 / 45 Short
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_26_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_2130_S;
			g_format.nsyms = 16200/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 26/45 S ");
			break;
		case 242:
			//242 16APSK 3 / 5 4 + 12APSK 3 / 5 Short
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_3_5;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_3201_S;
			g_format.nsyms = 16200/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 3/5 S ");
			break;
		case 244:
			//244 16APSK 32 / 45 4 + 12APSK 32 / 45 Short
			g_format.mod = m_4_12APSK;
			g_format.mod_class = m_16APSK;
			g_format.code_rate = cR_32_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_0123_S;
			g_format.nsyms = 16200/4;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12APSK 32/45 S ");
			break;
		case 246:
			//246 32APSK 2 / 3 4 + 12 + 16rbAPSK 2 / 3 Short
			g_format.mod = m_4_12_16rbAPSK;
			g_format.mod_class = m_32APSK;
			g_format.code_rate = cR_2_3;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_41230_S;
			g_format.nsyms = 16200/5;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12_16rbAPSK 2/3 S ");
			break;
		case 248:
			//248 32APSK 32 / 45 4 + 12 + 16rbAPSK 32 / 45 Short
			g_format.mod = m_4_12_16rbAPSK;
			g_format.mod_class = m_32APSK;
			g_format.code_rate = cR_32_45;
			g_format.frame_type = frame_SHORT;
			g_format.itype = I_10423_S;
			g_format.nsyms = 16200/5;
			snprintf(g_format.format_text,80,"DVB-S2X 4_12_16rbAPSK 3/45 S ");
			break;
		case 128:
			//128 8 - ary - normal - pilots off 21 690
			status = MODCOD_RESERVED;
			g_format.nsyms = 21690;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 130:
			//130 16 - ary - normal - pilots off 16 290
			status = MODCOD_RESERVED;
			g_format.nsyms = 16290;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 176:
			//176 32 - ary - normal - pilots off 13 050
			status = MODCOD_RESERVED;
			g_format.nsyms = 13050;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 177:
			//177 32 - ary - normal - pilots on 13 338
			status = MODCOD_RESERVED;
			g_format.nsyms = 13338;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 188:
			//188 64 - ary - normal - pilots off 10 890
			status = MODCOD_RESERVED;
			g_format.nsyms = 10890;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 189:
			//189 64 - ary - normal - pilots on 11 142
			status = MODCOD_RESERVED;
			g_format.nsyms = 11142;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 192:
			//192 64 - ary - normal - pilots off 10 890
			status = MODCOD_RESERVED;
			g_format.nsyms = 10890;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 193:
			//193 64 - ary - normal - pilots on 11 142
			status = MODCOD_RESERVED;
			g_format.nsyms = 11142;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 196:
			//196 64 - ary - normal - pilots off 10 890
			status = MODCOD_RESERVED;
			g_format.nsyms = 10890;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 197:
			//197 64 - ary - normal - pilots on 11 142
			status = MODCOD_RESERVED;
			g_format.nsyms = 11142;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 250:
			//250 8 - ary - normal - pilots on 22 194
			status = MODCOD_RESERVED;
			g_format.nsyms = 22194;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 251:
			//251 16 - ary - normal - pilots on 16 686
			status = MODCOD_RESERVED;
			g_format.nsyms = 16686;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 252:
			//252 32 - ary - normal - pilots on 13 338
			status = MODCOD_RESERVED;
			g_format.nsyms = 13338;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 253:
			//253 64 - ary - normal - pilots on 11 142
			status = MODCOD_RESERVED;
			g_format.nsyms = 11142;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 254:
			//254 256 - ary - normal - pilots on 8 370
			status = MODCOD_RESERVED;
			g_format.nsyms = 8370;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		case 255:
			//255
			status = MODCOD_RESERVED;
			g_format.nsyms = 6714;
			snprintf(g_format.format_text,80,"DVB-S2X RESERVED ");
			break;
		default:
			status = MODCOD_ERROR;
			snprintf(g_format.format_text,80,"DVB-S2X ERROR ");
			return status;
		}
        if(code&1) strcat(g_format.format_text,"P");

		if(g_format.frame_type == frame_SHORT){
			g_format.nldpc = 16200;
			g_format.bch = BCH_S12;
			switch(g_format.code_rate){
			case cR_1_4:
				g_format.q = 36;
				g_format.kbch = 3072;
				g_format.nbch = 3240;
				break;
			case cR_1_3:
				g_format.q = 30;
				g_format.kbch = 5232;
				g_format.nbch = 5400;
				break;
			case cR_2_5:
				g_format.q = 27;
				g_format.kbch = 6312;
				g_format.nbch = 6480;
				break;
			case cR_1_2:
				g_format.q = 25;
				g_format.kbch = 7032;
				g_format.nbch = 7200;
				break;
			case cR_3_5:
				g_format.q = 18;
				g_format.kbch = 9552;
				g_format.nbch = 9720;
				break;
			case cR_2_3:
				g_format.q = 15;
				g_format.kbch = 10632;
				g_format.nbch = 10800;
				break;
			case cR_3_4:
				g_format.q = 12;
				g_format.kbch = 11712;
				g_format.nbch = 11880;
				break;
			case cR_4_5:
				g_format.q = 10;
				g_format.kbch = 12432;
				g_format.nbch = 12600;
				break;
			case cR_5_6:
				g_format.q = 8;
				g_format.kbch = 13152;
				g_format.nbch = 13320;
				break;
			case cR_8_9:
				g_format.q = 5;
				g_format.kbch = 14232;
				g_format.nbch = 14400;
				break;
			case cR_11_45:
				g_format.q = 34;
				g_format.kbch = 3792;
				g_format.nbch = 3690;
				g_format.nldpc = 15390;
				break;
			case cR_4_15:
				g_format.q = 33;
				g_format.kbch = 4152;
				g_format.nbch = 4320;
				g_format.nldpc = 14976;
				break;
			case cR_14_45:
				g_format.q = 31;
				g_format.kbch = 4872;
				g_format.nbch = 5040;
				break;
			case cR_7_15:
				g_format.q = 24;
				g_format.kbch = 7392;
				g_format.nbch = 7560;
				break;
			case cR_8_15:
				g_format.q = 21;
				g_format.kbch = 8472;
				g_format.nbch = 8640;
				break;
			case cR_26_45:
				g_format.q = 19;
				g_format.kbch = 9192;
				g_format.nbch = 9360;
				break;
			case cR_32_45:
				g_format.q = 13;
				g_format.kbch = 11352;
				g_format.nbch = 11520;
				break;
			}
		}
		if(g_format.frame_type == frame_MEDIUM){
			g_format.nldpc = 32400;
			g_format.bch = BCH_M12;
			switch(g_format.code_rate){
			case cR_1_5:
				g_format.q = 72;
				g_format.kbch = 5660;
				g_format.nbch = 5480;
				g_format.nldpc = 30780;
				break;
			case cR_11_45:
				g_format.q = 68;
				g_format.kbch = 7740;
				g_format.nbch = 7920;
				g_format.nldpc = 30780;
				break;
			case cR_1_3:
				g_format.q = 60;
				g_format.kbch = 10620;
				g_format.nbch = 10800;
				g_format.nldpc = 30780;
				break;
			}
		}
		if(g_format.frame_type == frame_NORMAL){
			g_format.nldpc = 64800;
			g_format.bch = BCH_N12;
			switch(g_format.code_rate){
			case cR_1_4:
				g_format.q = 135;
				g_format.kbch = 16008;
				g_format.nbch = 16200;
				break;
			case cR_1_3:
				g_format.q = 120;
				g_format.kbch = 21408;
				g_format.nbch = 21600;
				break;
			case cR_2_5:
				g_format.q = 108;
				g_format.kbch = 25728;
				g_format.nbch = 25920;
				break;
			case cR_1_2:
				g_format.q = 90;
				g_format.kbch = 32208;
				g_format.nbch = 32400;
				break;
			case cR_3_5:
				g_format.q = 72;
				g_format.kbch = 38688;
				g_format.nbch = 38880;
				break;
			case cR_2_3:
				g_format.q = 60;
				g_format.kbch = 43040;
				g_format.nbch = 43200;
				break;
			case cR_3_4:
				g_format.q = 45;
				g_format.kbch = 48408;
				g_format.nbch = 48600;
				break;
			case cR_4_5:
				g_format.q = 36;
				g_format.kbch = 51648;
				g_format.nbch = 51840;
				break;
			case cR_5_6:
				g_format.q = 30;
				g_format.kbch = 53840;
				g_format.nbch = 54000;
				break;
			case cR_8_9:
				g_format.q = 20;
				g_format.kbch = 57472;
				g_format.nbch = 57600;
				break;
			case cR_9_10:
				g_format.q = 18;
				g_format.kbch = 58192;
				g_format.nbch = 58320;
				break;
			case cR_2_9:
				g_format.q = 140;
				g_format.kbch = 14208;
				g_format.nbch = 14400;
				g_format.nldpc = 61560;
				break;
			case cR_13_45:
				g_format.q = 128;
				g_format.kbch = 18528;
				g_format.nbch = 18720;
				break;
			case cR_9_20:
				g_format.q = 99;
				g_format.kbch = 28968;
				g_format.nbch = 29160;
				break;
			case cR_90_180:
				g_format.q = 90;
				g_format.kbch = 32208;
				g_format.nbch = 32400;
				break;
			case cR_96_180:
				g_format.q = 84;
				g_format.kbch = 34368;
				g_format.nbch = 34560;
				break;
			case cR_11_20:
				g_format.q = 81;
				g_format.kbch = 35448;
				g_format.nbch = 35640;
				break;
			case cR_100_180:
				g_format.q = 80;
				g_format.kbch = 35808;
				g_format.nbch = 36000;
				break;
			case cR_104_180:
				g_format.q = 76;
				g_format.kbch = 37248;
				g_format.nbch = 37440;
				break;
			case cR_26_45:
				g_format.q = 76;
				g_format.kbch = 37248;
				g_format.nbch = 37440;
				break;
			case cR_18_30:
				g_format.q = 72;
				g_format.kbch = 38688;
				g_format.nbch = 38880;
				break;
			case cR_28_45:
				g_format.q = 68;
				g_format.kbch = 40128;
				g_format.nbch = 40320;
				break;
			case cR_23_36:
				g_format.q = 65;
				g_format.kbch = 41208;
				g_format.nbch = 41400;
				break;
			case cR_116_180:
				g_format.q = 64;
				g_format.kbch = 41568;
				g_format.nbch = 41760;
				break;
			case cR_20_30:
				g_format.q = 60;
				g_format.kbch = 43008;
				g_format.nbch = 43200;
				break;
			case cR_124_180:
				g_format.q = 56;
				g_format.kbch = 44448;
				g_format.nbch = 44640;
				break;
			case cR_25_36:
				g_format.q = 55;
				g_format.kbch = 44808;
				g_format.nbch = 45000;
				break;
			case cR_128_180:
				g_format.q = 52;
				g_format.kbch = 45888;
				g_format.nbch = 46080;
				break;
			case cR_13_18:
				g_format.q = 50;
				g_format.kbch = 46608;
				g_format.nbch = 46800;
				break;
			case cR_132_180:
				g_format.q = 48;
				g_format.kbch = 47328;
				g_format.nbch = 47520;
				break;
			case cR_22_30:
				g_format.q = 48;
				g_format.kbch = 47328;
				g_format.nbch = 47520;
				break;
			case cR_135_180:
				g_format.q = 45;
				g_format.kbch = 48408;
				g_format.nbch = 48600;
				break;
			case cR_140_180:
				g_format.q = 40;
				g_format.kbch = 50208;
				g_format.nbch = 50400;
				break;
			case cR_7_9:
				g_format.q = 40;
				g_format.kbch = 50208;
				g_format.nbch = 50400;
				break;
			case cR_154_180:
				g_format.q = 26;
				g_format.kbch = 55248;
				g_format.nbch = 55440;
				break;
			}
		}

		if(g_format.pilots){
			if((g_format.nsyms%1440)){
				// We don't send the pilot block at the end of the frame
		        g_format.bsyms = g_format.nsyms + (((g_format.nsyms/1440))*36);
			}else{
			    // Exact whole number so no pilot block required at the end
			    g_format.bsyms = g_format.nsyms + (((g_format.nsyms/1440)-1)*36);
			}
	    }
		else{
			// No pilots
			g_format.bsyms  = g_format.nsyms;// Number of symbols in the body
		}
		g_format.nuldpc     = g_format.nbch;
		g_format.pbch       = g_format.nbch - g_format.kbch;
		g_format.pldpc      = g_format.nldpc- g_format.nuldpc;
		g_format.fsyms      = (npreams + g_format.bsyms);// Total number of symbols in a frame
		g_format.nsams      = g_format.fsyms*SAMPLES_PER_SYMBOL;

		if(g_format.nbch == 0 ) status = MODCOD_ERROR;
	}
	return status;// mocod valid
}
bool isvalid_modcod(uint32_t modcod ){

	if(modcod_decode( modcod )== MODCOD_OK ) return true;
	return false;
 }

const char *rolloff(void){
	static char text[25];
	switch(g_format.bbh.ro){
	case BB_RO_35:
        sprintf(text,"Rolloff 0.35");
		break;
	case BB_RO_25:
        sprintf(text,"Rolloff 0.25");
		break;
	case BB_RO_20:
        sprintf(text,"Rolloff 0.20");
		break;
	case BB_RO_15:
        sprintf(text,"Rolloff 0.15");
		break;
	case BB_RO_10:
        sprintf(text,"Rolloff 0.10");
		break;
	case BB_RO_05:
        sprintf(text,"Rolloff 0.05");
		break;
	default:
        sprintf(text," ");
		break;
	}
	return text;
}

#define gotoxy(x,y) printf("\033[%d;%dH", (x), (y))
#define clear() printf("\033[H\033[J")

void display_status(char *text, int n){
	char temp[20];

	float freq = (g_format.phase_delta*(g_format.act_sarate)/(2*M_PI));
    freq = g_format.req_freq - freq;
	if(receiver_islocked() == false){
		if(g_format.lock == RX_UNLOCKED_COARSE){
			snprintf(temp,20,"UNLOCKED0");
		}else{
			snprintf(temp,20,"UNLOCKED1");
		}
		double srate = hw_get_sarate();
	    snprintf(text,n,"%s\nFreq %.1f\nSRate %.0f\n",temp,freq,srate/2);
		stats_bch_errors_reset();
		stats_bbh_errors_reset();
		stats_tp_rx_reset();
		stats_tp_er_reset();
	}else{
		uint64_t tp_error = stats_tp_er_read();
		uint64_t tp_total = stats_tp_er_read() + stats_tp_rx_read();
		if(stats_update_read() == true){
		    float ferror = stats_freq_error_read()*g_format.act_sarate/(2*M_PI);
		    snprintf(text,n,"LOCKED*\nFreq %.1f\nSRate %.0f\n%s\n%s\nMag %.5f\nMER %.1f\n\nErrors\nFrequency %4.1f Hz\nLDPC FE %d\nBCH %3d BBH %3d\n\nTP Total %lu\nTP Error %lu\n",\
		    		freq, g_format.act_sarate/2,g_format.format_text, \
				    rolloff(), stats_mag_read(), stats_mer_read(),ferror,stats_ldpc_fes_read(),stats_bch_errors_read(),stats_bbh_errors_read(),tp_total,tp_error);
		    stats_bch_errors_reset();
		    stats_bbh_errors_reset();
	        stats_freq_error_reset();
	        stats_mag_reset();
		}else{
		    float ferror = stats_freq_error_read()*g_format.act_sarate/(2*M_PI);
		    snprintf(text,n,"LOCKED\nFreq %.1f\nSRate %.0f\n%s\n%s\nMag %.5f\nMER %.1f\n\nErrors\nFrequency %4.1f Hz\nLDPC FE %d\nBCH %3d BBH %3d\n\nTP Total %lu\nTP Error %lu\n", \
		    		freq, g_format.act_sarate/2,g_format.format_text, \
				    rolloff(), stats_mag_read(), stats_mer_read(),ferror, stats_ldpc_fes_read(),stats_bch_errors_read(),stats_bbh_errors_read(),tp_total,tp_error);
		}
	}
}
