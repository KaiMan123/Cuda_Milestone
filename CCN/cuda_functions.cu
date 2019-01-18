// CUDA runtime
#include <cuda_runtime.h>
#include <stdio.h>
// Helper functions and utilities to work with CUDA
// #include <helper_functions.h>

/**********************************************
 * Check whether we read back the same input
 * The double check is just for debug purposes.
 * We can comment it out when benchmarking the time.
 **********************************************/
#define GPU_DEBUG


/*
  Define all constant variavle below with a REASONABLE name
*/

#define out_channel_num 6
#define out_y_dim 358
#define out_x_dim 638
#define in_y_dim 720
#define in_x_dim 1280
#define filter_size 36
// filter_side * filter_side = filter_size
#define filter_side 6
// out_channel_num * filter size = shared_weight_size
#define shared_weight_size 216
/****************************************************** 
The original decided shared layer size is  8*8 which cover filter with 6*6 and have 1 column and 1 raw border.
Then we let it time two, that is, (8*2)*(8*2), to reduce the load into two load
Since the padding area for each cell is 5, we changed the size into (8*2 + 5)*(8*2 + 5)
Finally, as we are using uchar4 to transfer the data from global into shared memory,
it is better to the multiple of 4.
I used (8*2 + 5)*(8*2 + 5) because 8*2 + 8 = 24 is the most one closed to (8*2 + 5) = 21
******************************************************/
#define shared_layer_size 504
// (8*2 + 8) = 24
#define shared_layer_x_dim 24

/******************************************
 * Device function declaration
 *****************************************/
 __global__ void layer1_init_bias(float* d_y, float* d_bias);
 __global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight);
 __global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer);
 
 /************************************************************************************
  * Input   : input image, pointer to output result, coefficients bias and weights
  * Output  : neuron outputs of the feature maps represented as an image
  * Procedure: perform feed forward computation through the feature extraction layers
      *******************************************************************************/
 void cuda_convolution_layer1(unsigned char in_layer[], unsigned char out_layer[],
            const float bias[], const float weight[]) {
   /*********************************
    * allocate device memory on GPU
    *********************************/
 
   unsigned int size_y = out_channel_num*out_y_dim*out_x_dim;
   unsigned int mem_size_y = sizeof(float) * size_y;
   float *d_y;
 
   unsigned int size_bias = out_channel_num;
   unsigned int mem_size_bias = sizeof(float) * size_bias;
   float *d_bias;
 
   unsigned int size_weight = out_channel_num*filter_size;
   unsigned int mem_size_weight = sizeof(float) * size_weight;
   float *d_weight;
 
   unsigned int size_in_layer = in_y_dim*in_x_dim;
   unsigned int mem_size_in_layer = sizeof(unsigned char) * size_in_layer;
   unsigned char *d_in_layer;
 
   unsigned int size_out_layer = out_channel_num*out_y_dim*out_x_dim;
   unsigned int mem_size_out_layer = sizeof(unsigned char) * size_out_layer;
   unsigned char *d_out_layer;
 
   cudaError_t error;
 
 
   /********************************
    * Allocate device memory on GPU.
    * Check the first cudaMalloc error,
    * in case GPU is busy.
    ********************************/
   error = cudaMalloc((void **) &d_y, mem_size_y);
   /* Check the error code of the first CUDA API call */
   if (error != cudaSuccess){
     printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
     printf("CUDA error: %s\n", cudaGetErrorString(error));
   }else{
     printf("cudaMalloc success.\n");
   }
   /* if no error for the first cudaMalloc, continue other cudaMalloc */
   error = cudaMalloc((void **) &d_in_layer, mem_size_in_layer);
   error = cudaMalloc((void **) &d_bias, mem_size_bias);
   error = cudaMalloc((void **) &d_weight, mem_size_weight);
   error = cudaMalloc((void **) &d_out_layer, mem_size_out_layer);
 
   /*********************************************
    * copy data from host (CPU) to device (GPU)
    ********************************************/
   error = cudaMemcpy(d_in_layer, in_layer, mem_size_in_layer, cudaMemcpyHostToDevice);
   error = cudaMemcpy(d_bias, bias, mem_size_bias, cudaMemcpyHostToDevice);
   error = cudaMemcpy(d_weight, weight, mem_size_weight, cudaMemcpyHostToDevice);
 
   /* Synchronize all the cudaMemcpy API before doing the actual computation */
   cudaDeviceSynchronize();
 
   /*********************************************
    * Layer 1, Step 1: 
    * init values of feature maps at bias value 
    ********************************************/
   /* (16, 16, z) (choose your z dimension) threads per block */
   /* NOTE: threads per block limit is 1024 for K80 */
   /* NOTE: if you use another GPU, check the deviceQuery */
 
  dim3 grid_y(40, 23);
  dim3 block_y(16, 16, 1);
  layer1_init_bias<<< grid_y, block_y >>>(d_y, d_bias);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /*********************************************
   * Layer 1, Step 2: 
   * loop over output feature maps
   ********************************************/
  /* (8, 8, z) (choose your z dimension) threads per block */
  /***********************************************
   * The layer size is not diviadable by 8 either.
   * Mask out extra threads in the kernel.
   **********************************************/  
  
  dim3 grid_in(80, 45);
  dim3 block_in(8, 8, 1);
  layer1_feature_maps<<<grid_in,block_in>>>(d_y, d_in_layer, d_weight);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /********************************************
   (14, 14, z) (choose your z dimension) threads per block
   ********************************************
   * Layer 1, Step 3: 
   * sigmoid activation function
   ********************************************/
  
  dim3 grid_out(46, 26);
  dim3 block_out(14, 14, 1);
  layer1_sigmoid<<< grid_out, block_out >>>(d_y, d_out_layer);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /* Read back the output from device (GPU) to host (CPU) */
  error = cudaMemcpy(out_layer, d_out_layer, mem_size_out_layer, cudaMemcpyDeviceToHost);


  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /* release device memory */
  cudaFree(d_y);
  cudaFree(d_in_layer);
  cudaFree(d_bias);
  cudaFree(d_weight);
  cudaFree(d_out_layer);

}


/*********************************************
 * GPU kernel
 * Layer 1, Step 1: 
 * init values of feature maps at bias value 
 ********************************************/
__global__ void layer1_init_bias(float* d_y, float* d_bias) {
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if(index_y < out_y_dim && index_x < out_x_dim){
    for(int i = 0; i < 6; i++){
      d_y[i * out_x_dim * out_y_dim + index_y * out_x_dim + index_x] = d_bias[i];
    }
  }
}

/*********************************************
 * GPU kernel
 * Layer 1, Step 2: 
 * loop over output feature maps
 ********************************************/
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight) {
  int block_x = blockIdx.x;   
  int block_y = blockIdx.y;
  int dim_x = blockDim.x;
  int dim_y = blockDim.y;
  int thread_x = threadIdx.x; 
  int thread_y = threadIdx.y; 
  int index_y = block_y * dim_y + thread_y;
  int index_x = block_x * dim_x + thread_x;

  /******************************************************
  out_channel_num * filter size = shared_weight_size

  thread_x && thread_y should be smaller filter_side 
  because the size of each shared memory is 6*6 rather than 8*8
  where 6 is filter_side and 8 is block dim
  *******************************************************
  Both s_weight[i * filter_size + thread_y * filter_side + thread_x]
  and d_weight[i * filter_size + thread_y * filter_side + thread_x],
  i * filter_size is locating the filter in which output channel
  thread_y * filter_side + thread_x is locating the position of the shared memory
  ******************************************************/
  __shared__ float s_weight[shared_weight_size];
  if(thread_x < filter_side && thread_y < filter_side){
    for(int i = 0; i < out_channel_num; i++){
      s_weight[i * filter_size + thread_y * filter_side + thread_x] = d_weight[i * filter_size + thread_y * filter_side + thread_x];
    }
  }

  uchar4 division_container;
  __shared__ unsigned int s_layer[shared_layer_size];
  /*********************************************************************
  thread x could not greater than 6 because the shared_layer_x_dim is 24
  where shared_layer_x_division is 4
  Since 24 /4 = 6, thread x could not greater than 6
  ********************************************************************/
  if(thread_x < 6){
    /*************************************************
    For, ((uchar4*)d_in_layer)[((block_y * in_x_dim + block_x)*8*2 + thread_y*in_x_dim + thread_x*4)/4]
    
    (block_y * in_x_dim + block_x) is locate the block location
    (block_y * in_x_dim + block_x)*8*2 because it has 8 thread size and
    times 2 to jump 2 for each to minimizeto 2 loads
    
    For thread_y*in_x_dim + thread_x*4, thread_y*in_x_dim is locate the shared memory size
    thread_x*4 is jump 4 for uchar4

    Final divide 4 because dividing 4 is memory allocate uchar4.

    thread_y        is first  8 lines in shared layer
    8 + thread_y    is second 8 lines in shared layer
    16 + thread_y   is third  8 lines in shared layer 
    **************************************************
    Next for s_layer[thread_y*shared_layer_x_dim+thread_x*4+0],

    thread_y*shared_layer_x_dim is locating the y position

    thread_x*4 + {0, 1, 2, 3} refers to {x, y, z, w} in uchar4
    **************************************************/
    division_container = ((uchar4*)d_in_layer)[((block_y * in_x_dim + block_x)*8*2 + thread_y*in_x_dim + thread_x*4)/4];
    s_layer[thread_y*shared_layer_x_dim+thread_x*4+0] = division_container.x;
    s_layer[thread_y*shared_layer_x_dim+thread_x*4+1] = division_container.y;
    s_layer[thread_y*shared_layer_x_dim+thread_x*4+2] = division_container.z;
    s_layer[thread_y*shared_layer_x_dim+thread_x*4+3] = division_container.w;
    division_container = ((uchar4*)d_in_layer)[((block_y * in_x_dim + block_x)*8*2 + (8 + thread_y)*in_x_dim + thread_x*4)/4];
    s_layer[(8+thread_y)*shared_layer_x_dim+thread_x*4+0] = division_container.x;
    s_layer[(8+thread_y)*shared_layer_x_dim+thread_x*4+1] = division_container.y;
    s_layer[(8+thread_y)*shared_layer_x_dim+thread_x*4+2] = division_container.z;
    s_layer[(8+thread_y)*shared_layer_x_dim+thread_x*4+3] = division_container.w;
    /*********************************************************************
    to do the following shared memory alloc, thread y could not greater than 5
    because the shared_layer_y_dim (NOT define) is (16 + 5)
    where 8*2 + 5 was said above #define shared_layer_size 
    *********************************************************************/
    if(thread_y < 5){
      division_container = ((uchar4*)d_in_layer)[((block_y * in_x_dim + block_x)*8*2 + (16 + thread_y)*in_x_dim + thread_x*4)/4];
      s_layer[(16+thread_y)*shared_layer_x_dim+thread_x*4+0] = division_container.x;
      s_layer[(16+thread_y)*shared_layer_x_dim+thread_x*4+1] = division_container.y;
      s_layer[(16+thread_y)*shared_layer_x_dim+thread_x*4+2] = division_container.z;
      s_layer[(16+thread_y)*shared_layer_x_dim+thread_x*4+3] = division_container.w;
    }
  }
  __syncthreads();
  if(index_y < out_y_dim && index_x < out_x_dim){
    for(int j = 0; j < out_channel_num; j++){
      for(int k = 0; k < filter_side; k++){
        for(int l = 0; l < filter_side; l++){
          /**************************************
          For d_y[j * out_x_dim * out_y_dim + index_y * out_x_dim + index_x],

          j * out_x_dim * out_y_dim is locating the position in which output channel

          index_y * out_x_dim + index_x is locating the block position in each output channel
          ***************************************
          For s_layer[(thread_y * 2 + k) * 24 + (thread_x * 2 + l)],

          (thread_y * 2 + k) * 24 refers to the y position in the shared memory
          where * 2 means jump 2 to minimize the load into 2
           + k is refers to the filter y position from 0 to 5

          (thread_x * 2 + l) refers to the x position in the shared memory
          where * 2 means jump 2 to minimize the load into 2
           + k is refers to the filter x position from 0 to 5
          **************************************/
          d_y[j * out_x_dim * out_y_dim + index_y * out_x_dim + index_x] += s_layer[(thread_y * 2 + k) * 24 + (thread_x * 2 + l)] * s_weight[j * filter_size + k * filter_side + l];
        }
      }
    }
  }
  __syncthreads();
}

/*********************************************
 * GPU kernel
 * Layer 1, Step 3: 
 * sigmoid activation function
 ********************************************/
 __global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer){
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  if(index_y < out_y_dim && index_x < out_x_dim){
    for(int i = 0; i < 6; i++){
      d_out_layer[i * out_x_dim * out_y_dim + index_y * out_x_dim + index_x] = (unsigned char) (255.999f/(1.0+expf(-1.0*d_y[i * out_x_dim * out_y_dim + index_y * out_x_dim + index_x]/256.0)));
    }
  }
}
