
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// includes, project
#include <cutil_inline.h>

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))



// harris detector code to run on the host
void harrisDetectorHost(unsigned int *h_idata, unsigned int w, unsigned int h, 
                int ws,               // window size
                int threshold,        // threshold value to detect corners
                unsigned int * reference)
{
    int i,j,k,l;  // indexes in image
    int Ix, Iy;   // gradient in XX and YY
    int R;        // R metric
    int sumIx2, sumIy2, sumIxIy;

    for(i=0; i<h; i++) //height image
    {
        for(j=0; j<w; j++) //width image
        {
            reference[i*w+j]=h_idata[i*w+j]/4; // to obtain a faded background image
        }
    }

    for(i=ws+1; i<h-ws; i++) //height image
    {
        for(j=ws+1; j<w-ws; j++) //width image
        {
           sumIx2=0.0;sumIy2=0.0;sumIxIy=0.0;
           for(k=-ws; k<=ws; k++) //height window
              {
                  for(l=-ws; l<=ws; l++) //width window
                  {
                        Ix = ((int)h_idata[(i+k-1)*w + j+l] - (int)h_idata[(i+k)*w + j+l])/32;         
                        Iy = ((int)h_idata[(i+k)*w + j+l-1] - (int)h_idata[(i+k)*w + j+l])/32;         
			sumIx2 += Ix*Ix;
			sumIy2 += Iy*Iy;
			sumIxIy += Ix*Iy;
                  }
              }

              R = sumIx2*sumIy2-sumIxIy*sumIxIy-0.05*(sumIx2+sumIy2)*(sumIx2+sumIy2);
              if(R > threshold) {
                   reference[i*w+j]=255; 
              }
        }
    }
}   

// harris detector code to run on the GPU
void harrisDetectorDevice(unsigned int *h_idata, unsigned int w, unsigned int h, 
                  unsigned int ws, unsigned int threshold, 
                  unsigned int * h_odata)
{
    //TODO
}

// print command line format
void usage(char *command) 
{
    printf("Usage: %s [-h] [-d device] [-i inputfile] [-o outputfile] [-r referenceFile] [-w windowsize] [-t threshold]\n",command);
}

// main
int main( int argc, char** argv) 
{

    // default command line options
    int deviceId = 0;
    char *fileIn=(char *)"chess.pgm",*fileOut=(char *)"chessOut.pgm",*referenceOut=(char *)"reference.pgm";
    int ws = 2, threshold = 500;

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"d:i:o:r:w:t:h")) !=-1)
    {
        switch(opt)
        {

            case 'd':
                if(sscanf(optarg,"%d",&deviceId)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;

            case 'i':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                fileIn = strdup(optarg);
                break;
            case 'o':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileOut = strdup(optarg);
                break;
            case 'r':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                referenceOut = strdup(optarg);
                break;
            case 'w':
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&ws)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 't':
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&threshold)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 'h':
                usage(argv[0]);
                exit(0);
                break;

        }
    }

    // select cuda device
    cutilSafeCall( cudaSetDevice( deviceId ) );
    
    // create events to measure host harris detector time and device harris detector time

    cudaEvent_t startH, stopH, startD, stopD;
    cudaEventCreate(&startH);
    cudaEventCreate(&stopH);
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);



    // allocate host memory
    unsigned int* h_idata=NULL;
    unsigned int h,w;

    //load pgm
    if (cutLoadPGMi(fileIn, &h_idata, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", fileIn);
        exit(1);
    }

    // allocate mem for the result on host side
    unsigned int* h_odata = (unsigned int*) malloc( h*w*sizeof(unsigned int));
    unsigned int* reference = (unsigned int*) malloc( h*w*sizeof(unsigned int));
 
    // detect corners at host

    cudaEventRecord( startH, 0 );
    harrisDetectorHost(h_idata, w, h, ws, threshold, reference);   
    cudaEventRecord( stopH, 0 ); 
    cudaEventSynchronize( stopH );

    // detect corners at GPU
    cudaEventRecord( startD, 0 );
    harrisDetectorDevice(h_idata, w, h, ws, threshold, h_odata);   
    cudaEventRecord( stopD, 0 ); 
    cudaEventSynchronize( stopD );
    
    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    float timeH, timeD;
    cudaEventElapsedTime( &timeH, startH, stopH );
    printf( "Host processing time: %f (ms)\n", timeH);
    cudaEventElapsedTime( &timeD, startD, stopD );
    printf( "Device processing time: %f (ms)\n", timeD);

    // save output images
    if (cutSavePGMi(referenceOut, reference, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (cutSavePGMi(fileOut, h_odata, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    cutFree( h_idata);
    free( h_odata);
    free( reference);

    cutilDeviceReset();
}
