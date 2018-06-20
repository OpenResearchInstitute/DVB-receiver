#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#include <gtk/gtk.h>
#include "dvbs2_rx.h"
//

// Externally visible global variables
//

int      g_trace = 0;
RxFormat g_format;
//
// Private module variables
//
static char m_testfile[256];
static float m_sn;
static int   m_its = 0;

void display_help(void){
	printf("-testfile <filename>  <sn>       Test file to decode, signal to noise in decibels\n");
	printf("-pluto    <frequency> <srate>    Pluto Radio, frequency, symbol rate\n");
	printf("-lime     <frequency> <srate>    Lime  Radio, frequency, symbol rate\n");
	printf("-help                            Help\n");
}
SdrType parse_args(int argc, char *argv[]){

	if( argc == 1 ){
		display_help();
		return ERROR_SDR;
	}
	if( argc == 2 ){
	    if(strncmp(argv[1],"-help",5)==0){
	    	display_help();
			return HELP_SDR;
		}
	}
	if( argc >= 4 ){
		if(strncmp(argv[1],"-testfile",9)==0){
			strcpy(m_testfile,argv[2]);
			m_sn = atof(argv[3]);
			if(argc == 5 ) m_its = atoi(argv[4]);
			return TEST_SDR;
		}
		if(strncmp(argv[1],"-lime",5)==0){
			g_format.req_freq   = atof(argv[2]);
			g_format.req_sarate = atof(argv[3])*SAMPLES_PER_SYMBOL;
			g_format.act_sarate = g_format.req_sarate;
			return LIME_SDR;
		}
		if(strncmp(argv[1],"-pluto",6)==0){
			g_format.req_freq   = atof(argv[2]);
			g_format.req_syrate = atof(argv[3]);
		    g_format.act_sarate = g_format.req_sarate = g_format.req_syrate*SAMPLES_PER_SYMBOL;// two samples per symbol
			return PLUTO_SDR;
		}
	}else{
		printf("Not enough parameters\n\n");
    	display_help();
	}
	return ERROR_SDR;
}
static pthread_t m_threads[3];
static int m_threads_running;

//
// Convert from short to float
//
// I don't like this memory copy either
//
void short_to_float( SComplex *in, FComplex *out, int n){
	for( int i = 0; i < n; i++ ){
		out[i].re = in[i].re * 0.00003;
		out[i].im = in[i].im * 0.00003;
	}
}

void rx_frequency_translate( SComplex *in, FComplex *out, int len ){
	static float agc_sum;
	static FComplex sum;
	float mag,agc;
	FComplex nco;
	FComplex var;

	for( int i = 0; i < len; i++){
		// Make 1.15 a float
		var.re  = in[i].re;
		var.im  = in[i].im;
		sum.re  = sum.re*0.999999 + var.re*0.000001;
		sum.im  = sum.im*0.999999 + var.im*0.000001;
		var.re -= sum.re;
		var.im -= sum.im;
        // AGC
		mag     = var.re*var.re + var.im*var.im;
        agc_sum = agc_sum*0.9999 + sqrt(mag)*0.0001;
        agc     = 0.01/agc_sum;
        var.re *= agc;
        var.im *= agc;
        // Mixer
		nco.re  = cosf(g_format.phase_acc);
		nco.im  = sinf(g_format.phase_acc);
		out[i].re = cmultReal(var,nco);
		out[i].im = cmultImag(var,nco);
		if(receiver_islocked()==false) g_format.phase_delta = (float)zigzag_delta();

		g_format.phase_acc += g_format.phase_delta;
		if(g_format.phase_acc >  M_PI) g_format.phase_acc -= 2*M_PI;
		if(g_format.phase_acc < -M_PI) g_format.phase_acc += 2*M_PI;
	}
	// Restrict range of the phase accumulator
//	double acc = g_format.phase_acc/(2*M_PI);
//	double n;
//	acc = modf(acc,&n);
//	g_format.phase_acc = acc*(2*M_PI);
}

void *rx_pluto_thread( void *arg )
{
	FComplex *s = NULL;
	FComplex *p = NULL;
	int  n = 0;

	receiver_open();
	if(pluto_open( g_format.req_freq, g_format.req_syrate) == 0){
	    while( m_threads_running )
        {
		    SComplex *sa = NULL;
		    int ns = pluto_rx_samples(&sa);
		    if((sa != NULL)&&(ns != 0)){
		        // See if we need to increase the buffer size
		        if( ns > n ){
		    	    if( s != NULL ) free( s );
		    	    s = (FComplex*)malloc((sizeof(FComplex)*ns)+(sizeof(FComplex)*(RX_SAMPLE_HISTORY+KN)));
		    	    p = &s[RX_SAMPLE_HISTORY];
		    	    n = ns;
		        }
		        int len = ns;
		        rx_apply_ferror_adjust( 1.0 );
		        // Turn samples into floats and translate in frequency
		        rx_frequency_translate( &sa[0], p, len );
		        // Now call the receiver
		        receiver_samples( p, len );
            }
        }
    	if(s != NULL) free(s);
        pluto_close();
	}else{
		m_threads_running = 0;
	}
    receiver_close();
    return arg;
}

#define LIME_SB_SIZE 360000

void *rx_lime_thread( void *arg )
{
	SComplex *sa = NULL;
	FComplex *s  = NULL;
	FComplex *p  = NULL;
	int       n  = 0;

	sa = (SComplex*)malloc(sizeof(SComplex)*LIME_SB_SIZE);

	if(lime_open( g_format.req_freq, g_format.req_sarate) == 0){
	    receiver_open();
	    while( m_threads_running )
        {
		    int ns = lime_rx_buffer(sa, 0, LIME_SB_SIZE);
		   //printf("NS %d\n",ns);
		    if((sa != NULL)&&(ns != 0)){
		        // See if we need to increase the buffer size
		        if( ns > n ){
		    	    if( s != NULL ) free( s );
		    	    s = (FComplex*)malloc((sizeof(FComplex)*ns)+(sizeof(FComplex)*(RX_SAMPLE_HISTORY+KN)));
		    	    p = &s[RX_SAMPLE_HISTORY];
		    	    n = ns;
		        }
		        int len = ns;
		        rx_apply_ferror_adjust( 1.0 );
		        // Turn samples into floats and translate in frequency
		        rx_frequency_translate( &sa[0], p, ns );
			    // Now call the receiver
			    receiver_samples( p, len );
			   // printf("%d %d\n",sa[0].re, sa[0].im);
		    }
        }
	    receiver_close();
        lime_close();
	} else{
		m_threads_running = 0;

	}
	if(s != NULL) free(s);
    return arg;
}

extern FComplex *m_frame[2];

gboolean draw_callback(GtkWidget *widget, cairo_t *cr, gpointer data)
{
    guint width,height;
    GdkRGBA color;
    GtkStyleContext *context;

    gint offset = 0;
    gint cx,cy;

    context = gtk_widget_get_style_context(widget);
    width   = gtk_widget_get_allocated_width(widget);
    height  = gtk_widget_get_allocated_height(widget);
    height -= offset;
    cx = (height/2) + offset;
    cy = (height/2) + offset;

    color.red   = 0;
    color.green = 0;
    color.blue  = 0;
    color.alpha = 255;

    gdk_cairo_set_source_rgba(cr,&color);
    cairo_rectangle ( cr, offset, offset, height, height );
    cairo_fill (cr);

    // Draw the axis
    color.red   = 128;
    color.green = 128;
    color.blue  = 0;
    color.alpha = 255;

    cairo_set_line_cap(cr, CAIRO_LINE_CAP_SQUARE);
    gdk_cairo_set_source_rgba(cr,&color);
    cairo_set_line_width (cr, 1);
    cairo_move_to (cr, offset+(height/4), (height/2)+offset);
    cairo_line_to (cr, (height*3/4)+offset, (height/2)+offset);
    cairo_close_path (cr);
    cairo_stroke (cr);

    cairo_move_to (cr, (height/2)+offset, offset+(height/4));
    cairo_line_to (cr, (height/2)+offset, (height*3/4)+offset);
    cairo_close_path (cr);
    cairo_stroke (cr);

    cairo_arc(cr, (height/2)+offset, (height/2)+offset, height/4, 0, 2*M_PI);
    cairo_close_path (cr);
    cairo_stroke (cr);

    float fx,fy;
    gint x,y;

    FComplex *pframe = m_frame[(g_format.fn+1)%2];

    if(pframe != NULL){

    	int inc = g_format.nsyms*NP_FRAMES/1000;

        gdk_cairo_set_source_rgba(cr,&color);

        cairo_set_source_rgb (cr, 255, 255, 255);
        for( int i = 0; i < g_format.nsyms*NP_FRAMES; i += inc){
    	    fx = pframe[i].re;
    	    fy = pframe[i].im;

    	    fx = fabs(fx) < 2.0 ? fx : 0;
    	    fy = fabs(fy) < 2.0 ? fy : 0;
    	    x = (gint)((fx*height)/4) + cx;
    	    y = (gint)((fy*height)/4) + cy;
    		cairo_rectangle (cr, x, y, 1, 1);
        }
		cairo_fill (cr);
		gtk_widget_queue_draw(widget);
   }
   // Mark the constellation points
   cairo_set_source_rgb (cr, 255, 0, 0);
   x = (gint)((0.707106781*height)/4) + cx;
   y = (gint)((0.707106781*height)/4) + cy;
   cairo_rectangle (cr, x-2, y-2, 6, 6);
   x = (gint)((-0.707106781*height)/4) + cx;
   y = (gint)(( 0.707106781*height)/4) + cy;
   cairo_rectangle (cr, x-2, y-2, 6, 6);
   x = (gint)(( 0.707106781*height)/4) + cx;
   y = (gint)((-0.707106781*height)/4) + cy;
   cairo_rectangle (cr, x-2, y-2, 6, 6);
   x = (gint)((-0.707106781*height)/4) + cx;
   y = (gint)((-0.707106781*height)/4) + cy;
   cairo_rectangle (cr, x-2, y-2, 6, 6);
   cairo_fill (cr);

   return TRUE;
}

static gboolean status_update(gpointer data)
{
    GtkLabel *label = (GtkLabel*)data;
    char buf[2048];
    display_status(buf,2048);
    gtk_label_set_label(label, buf);
    return 1;
}
static gboolean constellation_update(gpointer data)
{
    GtkWidget *window = (GtkWidget*)data;
    gint width  = 200;
    gint height = 200;
    gtk_widget_queue_draw_area(window, 0, 0, height, height);
    return 1;
}

static void activate (GtkApplication* app, gpointer user_data)
{
  GtkWidget *window;
  GtkWidget *label;
  GtkWidget *boxh;
  GtkWidget *boxv;
  GtkWidget *drawing_area;

  window = gtk_application_window_new (app);
  gtk_window_set_title (GTK_WINDOW (window), "DVB-S2 Receiver");
  gtk_window_set_default_size (GTK_WINDOW (window), 600, 400);
  //gtk_window_set_resizable (GTK_WINDOW(window), FALSE);

  boxh = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 10);
  gtk_container_add(GTK_CONTAINER(window), boxh);

  drawing_area = gtk_drawing_area_new ();
  gtk_container_add (GTK_CONTAINER (boxh), drawing_area);
  gtk_widget_set_size_request(drawing_area,400,400);
  g_signal_connect(G_OBJECT(drawing_area),"draw",
  G_CALLBACK(draw_callback),NULL);

  label = gtk_label_new("Status");
  gtk_label_set_justify(GTK_LABEL(label), GTK_JUSTIFY_LEFT);
  gtk_container_add (GTK_CONTAINER (boxh), label);
 // gtk_box_pack_start(GTK_BOX(boxh), label, 0, 0, 0);

  gtk_widget_show_all (window);
  g_timeout_add_seconds(1, status_update, label);
  g_timeout_add_seconds(1, constellation_update, drawing_area);

}


gint main(gint argc, gchar *argv[]){

	GtkApplication *app;
	int status;

	app = gtk_application_new ("g4guo.dvbs2", G_APPLICATION_FLAGS_NONE);
	g_signal_connect (app, "activate", G_CALLBACK (activate), NULL);

	SdrType type = parse_args(argc, argv);

	g_format.sdr_type = type;

	// Set the frequency step and the range 100 Hz 300 steps = +/-30 KHz
	zigzag_reset();
	zigzag_set_inc_and_max( 1.0, 3000);

	if(type == HELP_SDR){
		return 0;
	}

	if(type == ERROR_SDR){
		return 0;
	}

	if(type == TEST_SDR){
		if(m_its > 0 )
		    g_format.ldpc_iterations = m_its;
		else
		    g_format.ldpc_iterations = 40;
		test(m_testfile, m_sn);
		return 0;
	}

	m_threads_running = 1;

	if(type == PLUTO_SDR){
		// This is not a test
		g_format.ldpc_iterations = 100;
	    if(pthread_create( &m_threads[0], NULL, rx_pluto_thread, NULL ) != 0 )
	    {
	        printf("Unable to start rx thread\n");
	        m_threads_running = 0;
	        return 0;
	    }
        printf("Pluto RX thread started\n");
	}

	if(type == LIME_SDR){
		// This is not a test
		g_format.ldpc_iterations = 100;
	    if(pthread_create( &m_threads[0], NULL, rx_lime_thread, NULL ) != 0 )
	    {
	        printf("Unable to start rx thread\n");
	        m_threads_running = 0;
	        return 0;
	    }
	}

	status = g_application_run (G_APPLICATION (app), 1, argv);
//	status = g_application_run (G_APPLICATION (app), argc, argv);
	g_object_unref (app);

//    getchar();
    printf("\nProgram exited, user command\n");
    m_threads_running = 0;
	return 0;
}
