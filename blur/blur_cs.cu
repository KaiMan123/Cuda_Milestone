#include<stdio.h>
#include<stdlib.h>

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

static void cuda_checker(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err) (cuda_checker(err, __FILE__, __LINE__ ))

#define CREATOR "COMP3231"
#define RGB_COMPONENT_COLOR 255


static PPMImage *readPPM(const char *filename)
{
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    //open PPM file for reading
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    //read image format
    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

__constant__ float filter[9];

__device__ unsigned long getindex(unsigned int x, unsigned int y, unsigned int color) {
	return x * 3 + y * 1900 * 3 + color;
}

__global__ void blur_kernel( unsigned char *output_i, unsigned char *input_i, unsigned int length_x, unsigned int length_y, unsigned int startpt_x, unsigned int startpt_y) {
    __shared__ unsigned char *temp;
    temp = input_i;
    float filter[9] = {0.05, 0.1, 0.05, 0.1, 0.4, 0.1, 0.05, 0.1, 0.05};
	int x = threadIdx.x + blockIdx.x * length_x + startpt_x;
	int y = threadIdx.y + blockIdx.y * length_y + startpt_y;
	int c = threadIdx.z;
    //kernel code
	if (x > 0 && y > 0) {
		float target[3][3] = { 0 };
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				target[i][j] = temp[getindex(x - 1 + i, y - 1 + j, c)];
			}
		}
		float result = 0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				result += target[i][j] * filter[i + j * 3];
			}
		}
		output_i[getindex(x, y, c)] = result;
	}
	else {
		output_i[getindex(x, y, c)] = temp[getindex(x, y, c)];
	}

	
}

void your_gaussian_blur_func(PPMImage *img) {
    
    //host code 
	dim3 dimBlock(16, 16, 3);
	dim3 dimGrid((img->x)/16, (img->y)/16);
	unsigned char *dev_bitmap, *dev_blur_bitmap;
    unsigned long dev_size = (3 * img->x * img->y * sizeof(unsigned char));
    
	CUDA_CHECK(cudaMalloc((void **) &dev_bitmap, dev_size));
	CUDA_CHECK(cudaMalloc((void **) &dev_blur_bitmap, dev_size));

	CUDA_CHECK(cudaMemcpy(dev_bitmap, img->data, dev_size, cudaMemcpyHostToDevice));

    //Set the timer to calculate the time required for the gpu executing
	clock_t begin = clock();

	blur_kernel<<<dimGrid, dimBlock>>>(dev_blur_bitmap, dev_bitmap,16,16,0,0);
	//black line on the right hand side
	if (img->x%16 != 0)
	{
		dim3 dimBlock2(img->x % 16, 1, 3);
		dim3 dimGrid2(1, img->y);
		blur_kernel <<<dimGrid2, dimBlock2 >>> (dev_blur_bitmap, dev_bitmap, img->x % 16, 1, ((img->x) / 16) * 16, 0);
	}
	//black line at the bottom
	if (img->y%16 != 0)
	{
		dim3 dimBlock2(1, img->x % 16, 3);
		dim3 dimGrid2(img->x,1);
		blur_kernel <<<dimGrid2, dimBlock2 >>> (dev_blur_bitmap, dev_bitmap, 1, img->y % 16, 0, ((img->y) / 16) * 16);

	}
	CUDA_CHECK(cudaMemcpy(img->data, dev_blur_bitmap, dev_size, cudaMemcpyDeviceToHost));
    
   	//Stop the timer for calculating the gpu executing time
	clock_t end = clock();
	//Calculate the time spent for the gpu executin
	double time_spent = (double)(end - begin);

	printf("Time to generate:  %3.1f \n", time_spent);

    CUDA_CHECK(cudaFree(dev_bitmap)); 
    CUDA_CHECK(cudaFree(dev_blur_bitmap));
}

int main(){
    PPMImage *image;
    image = readPPM("input.ppm");
    your_gaussian_blur_func(image);

    writePPM("output.ppm",image);

}