

#include <time.h>
#include <stdio.h>
#include <cuda.h>

// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"



// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                          const char *file,
                          int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}

int gpuDevSel() { //method to select device with largest number of max threads in a block
		// returns that maxThread number

	int dev_count;
	GPU_CHECKERROR( cudaGetDeviceCount( &dev_count ) );


	cudaDeviceProp dev_prop;
	unsigned int  maxThreads = 0;
	unsigned int devNum = 0;
	for(int i = 0; i < dev_count; i++) { //loops through CUDA devices
		GPU_CHECKERROR( cudaGetDeviceProperties( &dev_prop, i ) );
		if(dev_prop.maxThreadsPerBlock > maxThreads) { //sets devNum to device with
			maxThreads =(unsigned int) dev_prop.maxThreadsPerBlock; //highest num threads per 
			devNum = i;				//block
		}
	}
	GPU_CHECKERROR( cudaSetDevice(devNum) ); //selects CUDA device

	return maxThreads;

}

__global__ void gpuCalLum(float * array, int width, int height) {
	//origin is top left pixel of the block
	int originX = blockDim.x * blockIdx.x; 
	int originY = blockDim.y * blockIdx.y;

	//indexZ is number of pixels right or down of the origin in the block
	int indexX = threadIdx.x;
	int indexY = threadIdx.y;

	//maps the current thread/pixel to a position in the 1D array of RGB values
	int index = 3*((originX+indexX)+(originY + indexY)*width);
/*
	float red = **(array + index)*0.2126f; 
	float green = **(array + index + 1)*0.7152f;
	float blue = **(array + index + 1)*0.0722f;
*/

	if(((originY + indexY) < height) &&  ((originX + indexX) < width)) { 
		float L = 0.2126f*array[index]+
			  0.7152f*array[index+1]+
			  0.0722f*array[index+2];
		
		*(array+index)= L;
		*(array+index+1)= L;
		*(array+index+2)= L;
	}	

}


//
// your __global__ kernel can go here, if you want:
//


int main (int argc, char *argv[])
{
 

	clock_t timer1, timer2;
    printf("reading openEXR file %s\n", argv[1]);
        
    int w, h;   // the width & height of the image, used frequently!


    // First, convert the openEXR file into a form we can use on the CPU
    // and the GPU: a flat array of floats:
    // This makes an array h*w*sizeof(float)*3, with sequential r/g/b indices
    // don't forget to free it at the end


    timer1 = clock();
    float *h_imageArray;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);

    // 
    // serial code: saves the image in "hw1_serial.exr"
    //

    // for every pixel in p, get it's Rgba structure, and convert the
    // red/green/blue values there to luminance L, effectively converting
    // it to greyscale:

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            unsigned int idx = ((y * w) + x) * 3;
            
            float L = 0.2126f*h_imageArray[idx] + 
                      0.7152f*h_imageArray[idx+1] + 
                      0.0722f*h_imageArray[idx+2];

            h_imageArray[idx] = L;
            h_imageArray[idx+1] = L;
            h_imageArray[idx+2] = L;

       }
    }
    
    printf("writing output image hw1_serial.exr\n");
    writeOpenEXRFile ("hw1_serial.exr", h_imageArray, w, h);
    free(h_imageArray); // make sure you free it: if you use this variable
                        // again, readOpenEXRFile will allocate more memory

	timer1 = clock() - timer1;

    //
    // Now the GPU version: it will save whatever is in h_imageArray
    // to the file "hw1_gpu.exr"
    //
    
	timer2 = clock();
    // read the file again - the file read allocates memory for h_imageArray:
    readOpenEXRFile (argv[1], &h_imageArray, w, h);



    // at this point, h_imageArray has sequenial floats for red, green , and
    // blue for each pixel: r,g,b,r,g,b,r,g,b,r,g,b. You need to copy
    // this array to GPU global memory, and have one thread per pixel compute
    // the luminance value, with which you will overwrite each r,g,b, triple.

    //
    // process it on the GPU: 1) copy it to device memory, 2) process
    // it with a 2d grid of 2d blocks, with each thread assigned to a 
    // pixel. then 3) copy it back.
    //
	unsigned int numPixels = w * h;
	unsigned int arraySize = 3 * numPixels;
	unsigned int maxThreads;
    	maxThreads = gpuDevSel();

	unsigned int threadDim = sqrt(maxThreads);
//	printf("threadDim =%d\n", threadDim);
	//unsigned int numBlocks =  ceil( numPixels/ (float) maxThreads );
	int gridWidth = ceil( (float) w/threadDim );
	int gridHeight = ceil( (float) h/threadDim );
//	printf("gridWidth = %d, gridHeight = %d\n", gridWidth, gridHeight);
	dim3 grid(gridWidth,gridHeight);
	dim3 threads(threadDim, threadDim);	

//	printf("width = %d, height = %d\n", w, h);


	//creates and allocates memory to the device array
	//copies array from host to device
	float * d_imageArray;
	GPU_CHECKERROR( cudaMalloc((void **) &d_imageArray, arraySize*sizeof(float)) ) ; 
	GPU_CHECKERROR( cudaMemcpy((void *) d_imageArray, (void *) h_imageArray, arraySize*sizeof(float), 
			cudaMemcpyHostToDevice));

	gpuCalLum<<<grid, threads>>>(d_imageArray,w,h);


    //
    // Your memory copy, & kernel launch code goes here:
    //


//	copies array back from device to host and frees device array
	GPU_CHECKERROR( cudaMemcpy((void *) h_imageArray, (void *) d_imageArray, arraySize*sizeof(float), cudaMemcpyDeviceToHost));
	GPU_CHECKERROR( cudaFree((void *) d_imageArray) );


    // All your work is done. Here we assume that you have copied the 
    // processed image data back, frmm the device to the host, into the
    // original host array h_imageArray. You can do it some other way,
    // this is just a suggestion
    
    printf("writing output image hw1_gpu.exr\n");
    writeOpenEXRFile ("hw1_gpu.exr", h_imageArray, w, h);
    free (h_imageArray);
    timer2 = clock() - timer2;
	printf("This was calculated serially in %f seconds.\n", ((float)timer1)/CLOCKS_PER_SEC);
	printf("This was calculated in parallel in %f seconds.\n", ((float)timer2)/CLOCKS_PER_SEC);

    printf("done.\n");

    return 0;
}


