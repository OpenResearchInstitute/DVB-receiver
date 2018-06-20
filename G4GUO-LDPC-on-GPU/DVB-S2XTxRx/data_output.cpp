#include <stdio.h>
#include <memory.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include "dvbs2_rx.h"

//extern RxFormat g_format;

static int m_tx_sock;
static struct sockaddr_in m_udp_client;

#define BBH_N 10

uint8_t m_old[188];

void parse_tsp(uint8_t *b){
	static int m_f;
	static int count;
	static uint8_t last_seq;

	uint8_t seq;
	uint32_t pid = b[1]&0x1F;
	pid <<= 8;
	pid |= b[2];
	seq = b[3]&0x0F;
    if(pid == 256){
	    if(((last_seq+1)&0x0F)!= seq){
		    printf("%d SEQ error %d %d after %d\n",count,last_seq,seq,m_f);
		    for( int i = 0; i < 188; i++) printf("%.2X ",m_old[i]);
		    printf("\n");

		    for( int i = 0; i < 188; i++) printf("%.2X ",b[i]);
		    printf("\n");
		    m_f = 0;
	    }
 //       printf(" %d %d\n",m_f,seq);
	    last_seq = seq;
	    memcpy(m_old,b,188);
    }
    m_f++;
    count++;

}
int udp_send_tp( uint8_t *b, int len  )
{
    return sendto(m_tx_sock, b, len, 0,(struct sockaddr *) &m_udp_client, sizeof(m_udp_client));
}

int udp_tx_init( void )
{
    // Create a socket for transmitting UDP TS packets
    if ((m_tx_sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        printf("Failed to create UDP TX socket\n");
        return -1;
    }
    // Construct the client sockaddr_in structure
    memset(&m_udp_client, 0, sizeof(m_udp_client));// Clear struct
    m_udp_client.sin_family = AF_INET;             // Internet/IP
    m_udp_client.sin_addr.s_addr = inet_addr("127.0.0.1");  // IP address (loopback)
    m_udp_client.sin_port = htons(1234);          // server port

    return 0;
}

static uint8_t m_tsp[1000];// Memory for transports stream packet
static int     m_tsp_cnt;
static int m_tp_errors;

void update_tp(uint8_t b ){
	uint8_t crc;
	m_tsp[m_tsp_cnt+1] = b;
	m_tsp_cnt++;
	if(m_tsp_cnt >= 188){
		if((crc=bb_calc_crc8( &m_tsp[1], 188)) == 0){
			// Send output packet
			udp_send_tp( m_tsp, 188 );
			stats_tp_rx_update(1);

		//	parse_tsp( m_tsp);
		}else{
		//	udp_send_tp( m_tsp, 188 );
		//	printf("CRC %.2X\n",crc);
    		m_tp_errors++;
    		stats_tp_er_update(1);
		}
		m_tsp_cnt = 0;
	}
}
// Force sync
void sync_tp(void){
	m_tsp_cnt = 0;
}
//
// The SYNCD points to the first bit in the user data of a CRC field
// The user data field starts after the 80 bits 0 bytes of the BB header.
// I believe that the data between the end of the neader and the first bit of the CRC is a packet from the last frame.
//
void data_output_transportstream(uint8_t *in, int len, BBHeader *h){
	uint8_t  sync;
	uint16_t start;
    uint16_t packet_size;
    uint16_t length;

    if(h->crc_ok){
        packet_size = h->upl/8;// size of a packet.
	    // SYNCD points to the first CRC bit at the end of the packet as the CRC field is 8 bits we add one to the length
	    start = (h->syncd/8)+1; // Start of the valid data in the payload
	    for( int i = 0; i < start; i++) update_tp(in[i]);
	    m_tsp[0] = h->sync; // Save the sync value
	    // We are now at the first valid byte of the next packet, force a sync
	    sync_tp();
        // Calculate the number of bytes in the payload
	    len = h->dfl/8 ;
        // Output the bytes or octets (in telecoms speak)
        for( int i = start; i < len; i++) update_tp(in[i]);
    }else{
    	// We cannot rely on the header information as the CRC was bad.
    	// Assume we have sync and that the data type is the same as before
        for( int i = 0; i < len; i++) update_tp(in[i]);
    }
}

void benchtest_data_output_transportstream(uint8_t *in, int len, BBHeader *h){
	uint8_t  sync;
	uint16_t start;
    uint16_t packet_size;
    uint16_t length;

    m_tp_errors = 0;
    sync_tp();//Test frame data always starts at the begining.

    if(h->crc_ok){
        packet_size = h->upl/8;// size of a packet.
	    // SYNCD points to the first CRC bit at the end of the packet as the CRC field is 8 bits we add one to the length
	    start = (h->syncd/8)+1; // Start of the valid data in the payload
	    for( int i = 0; i < start; i++) update_tp(in[i]);
	    m_tsp[0] = h->sync; // Save the sync value
	    // We are now at the first valid byte of the next packet, force a sync
	    sync_tp();
        // Calculate the number of bytes in the payload
	    length = h->dfl/8;
        // Output the bytes or octets (in telecoms speak)
        for( int i = start; i < len; i++) update_tp(in[i]);
    }else{
        for( int i = 0; i < len; i++) update_tp(in[i]);
    }

}

void data_output_open(void){
	m_tp_errors = 0;
	udp_tx_init();
}
void data_output_close(void){

}
