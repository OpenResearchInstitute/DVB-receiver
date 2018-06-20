#include  <stdio.h>
#include "dvbs2_rx.h"

static uint8_t  m_crc_tab[256];

#define CRC_POLY 0xAB
// Reversed
#define CRC_POLYR 0xD5

static int m_ro;
static int m_last_ro;

void bb_update_rolloff(int ro){

	if(m_last_ro != ro){
        switch(ro){
        case BB_RO_35:
        	hw_set_rolloff( 0.35f );
        	break;
        case BB_RO_25:
        	hw_set_rolloff( 0.25f );
        	break;
        case BB_RO_20:
        	hw_set_rolloff( 0.20f );
        	break;
        case BB_RO_15:
        	hw_set_rolloff( 0.15f );
        	break;
        case BB_RO_10:
        	hw_set_rolloff( 0.10f );
        	break;
        case BB_RO_05:
        	hw_set_rolloff( 0.05f );
        	break;
        default:
        	break;
        }
		m_last_ro = ro;
	}
}
void bb_build_crc8_table( void )
{
    int r,crc;

    for( int i = 0; i < 256; i++ )
    {
        r = i;
        crc = 0;
        for( int j = 7; j >= 0; j-- )
        {
            if((r&(1<<j)?1:0) ^ ((crc&0x80)?1:0))
                crc = (crc<<1)^CRC_POLYR;
            else
                crc <<= 1;
        }
        m_crc_tab[i] = crc;
    }
}

uint8_t bb_calc_crc8( uint8_t *b, int len )
{
    uint8_t crc = 0;
    for( int i = 0; i < len; i++ ) crc = m_crc_tab[b[i]^crc];

    return crc;
}

void bb_header_decode(uint8_t *b, BBHeader *h){
	// Now check the CRC
	uint8_t crc;
	if((crc=bb_calc_crc8( b, 10 )) == 0)
	{
		h->ts_gs   = (b[0]>>6)&0b11;
		h->sis_mis = (b[0]>>5)&0b1;
		h->ccm_acm = (b[0]>>4)&0b1;
		h->issyi   = (b[0]>>3)&0b1;
		h->npd     = (b[0]>>2)&0b1;
		h->ro      = (b[0]>>0)&0b11;

		h->mat2    = b[1];
		h->upl     = b[2];
		h->upl   <<= 8;
		h->upl    |= b[3];
		h->dfl     = b[4];
		h->dfl   <<= 8;
		h->dfl    |= b[5];
		h->sync    = b[6];
		h->syncd   = b[7];
		h->syncd <<= 8;
		h->syncd  |= b[8];
		h->crc_ok  = 1;

		unsigned int temp = (m_ro<<2)|h->ro;
		switch(temp){
			case 0b0000:
			case 0b0101:
			case 0b1010:
				// Standard rolloff
				bb_update_rolloff(temp);
				break;
			case 0b0011:
			case 0b0111:
			case 0b1011:
				// Tight rolloff
				h->ro = m_ro | 0b100;
				bb_update_rolloff(h->ro);
				break;
			case 0b1100:
			case 0b1101:
			case 0b1110:
				// Tight rolloff
				h->ro |= 0b100;
				bb_update_rolloff(h->ro);
				break;
			default:
				// Error in format, panic!
				break;
		}
		m_ro = h->ro;
	}
	else
	{
		h->crc_ok = 0;
		m_ro     = -1;
	}
}
void bbh_crc_table(void){
	uint8_t cr = bb_calc_crc8( m_crc_tab, 256 );
}
void bb_header_open(void){
	bb_build_crc8_table();
	m_ro      = -1;
	m_last_ro = -1;
	bbh_crc_table();
}
void bb_header_close(void){

}
