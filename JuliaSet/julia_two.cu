#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#define DIM 1024

static void cuda_checker(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

//Create the CUDA_CHECK for easy call to check the cuda program
#define CUDA_CHECK(err) (cuda_checker(err, __FILE__, __LINE__ ))


struct cppComplex {
	float r; 
	float i;
	cppComplex( float a, float b ) : r(a), i(b) {}
	float magnitude2( void ) {
		return r * r + i * i;
	}
	cppComplex operator*(const cppComplex& a) {
		return cppComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	cppComplex operator+(const cppComplex& a) {
		return cppComplex(r+a.r, i+a.i);
	}
};

int julia_cpu( int x, int y ) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);
	
	cppComplex c(-0.8, 0.156);
	cppComplex a(jx, jy);

	int i = 0;
	for(i=0; i<200; i++){
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

void julia_set_cpu() {

	unsigned char *pixels = new unsigned char[DIM * DIM];
	//Declear the starting time of start the cpu executing
	clock_t begin = clock();
	for (int x = 0; x < DIM; ++x) {
		for (int y = 0; y < DIM; ++y) {
			pixels[x + y * DIM] = 255 * julia_cpu(x, y);
		}
	}
	//Declear the ending time of end the cpu executing
	clock_t end = clock();
	//Calculate the time spent for the cpu executin
	double time_spent = (double)(end - begin);

	printf("Time to generate:  %3.1f \n", time_spent);

	//print the graph
	FILE *f = fopen("julia_cpu.ppm", "wb");

    fprintf(f, "P6\n%i %i 255\n", DIM, DIM);
    
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            fputc(pixels[(y * DIM + x)], f);
            fputc(0, f);
            fputc(0, f);
      }
    }
    fclose(f);

    delete [] pixels;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
/*Begin the GPU part*/
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

__global__ void kernel( unsigned char *ptr) {
	//Declear the x position of the graph
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	//Declear the y position of the graph
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//Declear the block name index
	int index = x + y * DIM;
	
	float jx = 1.5*(float)(DIM / 2 - x) / (DIM / 2);
	float jy = 1.5*(float)(DIM / 2 - y) / (DIM / 2);

	//set counter i and j for looping
	int i = 0,j = 2;
	
	do {
		float real_p = jx * jx - jy * jy - 0.8;
		float img_p = jx * jy * 2 + 0.156;
		float magnitude = real_p * real_p + img_p * img_p;
		jx = real_p;
		jy = img_p;
		if (magnitude > 1000) {
			ptr[index] = 0;
			j = 0;
		}
	} while (i++ < 200 && j == 2);
	
	if (j == 2) {
		ptr[index] = 1;
	};
}

void julia_set_gpu() {
	int threads = 16;

	dim3 threads_2d = dim3(threads, threads);
	dim3 blocks_2d = dim3(DIM / threads, DIM / threads);

	//Declear two variable pixels for cpu and dev_bitmap for gpu
	unsigned char *pixels = new unsigned char[DIM*DIM];
	unsigned char *dev_bitmap = new unsigned char[DIM*DIM];

	//Allocate and provide memory for dev_bitmap in gpu
	CUDA_CHECK(cudaMalloc(&dev_bitmap, DIM*DIM));

	//Set the timer to calculate the time required for the gpu executing
	clock_t begin = clock();

	//According to <<<No of block, No of Thread>>>, this gpu provided DIM*DIM of blocks and 1 thread per block
	kernel<<<blocks_2d,threads_2d>>>(dev_bitmap);

	//Copy the Data from GPU to CPU as GPU data could not be read in CPU program
	CUDA_CHECK(cudaMemcpy(pixels, dev_bitmap, DIM*DIM, cudaMemcpyDeviceToHost));
	
	//Stop the timer for calculating the gpu executing time
	clock_t end = clock();
	//Calculate the time spent for the gpu executin
	double time_spent = (double)(end - begin);

	printf("Time to generate:  %3.1f \n", time_spent);

	//Free the gpu memory
	CUDA_CHECK(cudaFree(dev_bitmap));

	//Print the graph
	FILE *f = fopen("julia_gpu.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", DIM, DIM);
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
			//The value of pixels should times 255 as the returned value in gpu is 1 and 2
            fputc((pixels[y * DIM + x])*255, f);   
            fputc(0, f);
            fputc(0, f);
      }
    }
	fclose(f);
	
	//Delete the pointer to free the memory
	delete [] pixels; 
}



int main( void ) {
	
	julia_set_cpu();
	julia_set_gpu();

}
