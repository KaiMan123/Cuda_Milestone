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
/****************************************************** 
The original decided shared layer size is  8*8 which cover filter with 6*6 and have 1 column and 1 raw border.
Also, for best usage of shared memory, it will be times 2, that is, (8*2)*(8*2)
As it has padding 5 for each, the size will become (8*2 + 5)*(8*2 + 5)
As we used uchar4 to transfer data from global to shared memory, it is better to be multiple of 4
Thus, the size will be (8*2 + 8)*(8*2 + 5) = 504 for easier handle uchar packing and unpacking
******************************************************/
#define shared_layer_size 504
/******************************************
 * Device function declaration
 *****************************************/
__global__ void layer1_init_bias(float* d_y, float* d_bias);
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight, int i);
__global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer);

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : neuron outputs of the feature maps represented as an image
 * Procedure: perform feed forward computation through the feature extraction layers
     *******************************************************************************/
void cuda_convolution_layer1(unsigned char in_layer[], unsigned char out_layer[],
			     const float bias[], const float weight[]) {

  unsigned int i;

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

#ifdef GPU_DEBUG
  int error_cnt;
  unsigned char cuda_in_layer[720*1280];
  float cuda_weight[6*36];
  float cuda_bias[6];
#endif

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
  /* 16*16*z(choose the correct z dimension) threads per block */
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
  /* 8*8*z(choose the correct z dimension) threads per block */
  /***********************************************
   * The layer size is not diviadable by 8 either.
   * Mask out extra threads in the kernel.
   **********************************************/  
  
  dim3 grid_in(80, 45);
  dim3 block_in(8, 8, 1);
  for(i = 0; i < 6; i++){
    layer1_feature_maps<<<grid_in,block_in>>>(d_y, d_in_layer, d_weight, i);
  }

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /********************************************
   14*14*z(choose the correct z dimension) threads per block
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
  /* For debug purposes, also check whether the input is indeed correct */
#ifdef GPU_DEBUG
  error = cudaMemcpy(cuda_bias, d_bias, mem_size_bias, cudaMemcpyDeviceToHost);
  error = cudaMemcpy(cuda_weight, d_weight, mem_size_weight, cudaMemcpyDeviceToHost);
  error = cudaMemcpy(cuda_in_layer, d_in_layer, mem_size_in_layer, cudaMemcpyDeviceToHost);
#endif

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /* release device memory */
  cudaFree(d_y);
  cudaFree(d_in_layer);
  cudaFree(d_bias);
  cudaFree(d_weight);
  cudaFree(d_out_layer);

  /**********************************************
   * Check whether we read back the same input
   * The double check is just for debug purposes.
   * We can comment it out when benchmarking the time.
   **********************************************/
#ifdef GPU_DEBUG
  /* check bias read back */
  error_cnt = 0;
  for(i = 0; i < size_bias; i++){
    if(cuda_bias[i]!=bias[i]){
      error_cnt++;
      printf("error at %d with mistake %f and correct one is %f \n", i, cuda_bias[i], bias[i]);
    }
  }
  if(error_cnt == 0){
    printf("Bias read back: passed.\n");
  }else{
    printf("Bias read back has eror. Number of error: %d\n", error_cnt);
  }
  /* check weight read back */
  error_cnt = 0;
  for(i = 0; i < size_weight; i++){
    if(cuda_weight[i]!=weight[i]){
      error_cnt++;
    }
  }
  if(error_cnt == 0){
    printf("Weight read back: passed.\n");
  }else{
    printf("Weight read back has error. Number of error: %d\n", error_cnt);
  }
  /* check in_layer read back */
  error_cnt = 0;
  for(i = 0; i < 720*128; i++){
    if( cuda_in_layer[i] != in_layer[i] ){
      error_cnt++;
    }
  }
  if(error_cnt == 0){
    printf("In_layer read back: passed.\n");
  }else{
    printf("In_layer read back has error. Number of error: %d\n", error_cnt);
  }
#endif

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
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight, int r) {
  int block_x = blockIdx.x;   
  int block_y = blockIdx.y;   
//  int block_z = blockIdx.z;
  int thread_x = threadIdx.x; 
  int thread_y = threadIdx.y; 
//  int thread_z = threadIdx.z; 
  int index_y = block_y * blockDim.y + thread_y;
  int index_x = block_x * blockDim.x + thread_x;

  /****************************************************************************
  The in_layer meomry transfer will be divided into 3 part
  where 24 is the whole size of 8 side length times 3 part
  where each uchar4 include 4 part and jump 2 and the copy of the filter is each cell 2 times
  so it will be thread_x*4+? 
  *****************************************************************************/
  // The memory from global d_in_layer will be stored into shared memory for first part
  int s1_in_layer = block_y*8*2*in_x_dim+block_x*8*2 + thread_y*in_x_dim + thread_x*4;
  // The memory from global d_in_layer will be stored into shared memory for second part
  int s2_in_layer = block_y*8*2*in_x_dim+block_x*8*2 + (8+thread_y)*in_x_dim + thread_x*4;
  // The memory from global d_in_layer will be stored into shared memory for third part
  int s3_in_layer = block_y*8*2*in_x_dim+block_x*8*2 + (16+thread_y)*in_x_dim + thread_x*4;

  uchar4 uchar4_tmp;
  __shared__ float s_weight[filter_size];
  __shared__ unsigned int s_layer[shared_layer_size];
  if(thread_x < filter_side && thread_y < filter_side){
    s_weight[thread_y * filter_side + thread_x] = d_weight[r * filter_size + thread_y * filter_side + thread_x];
  }
  if(thread_x < 6){
    uchar4_tmp = ((uchar4*)d_in_layer)[s1_in_layer / 4];
    s_layer[thread_y*24+thread_x*4+0] = uchar4_tmp.x;
    s_layer[thread_y*24+thread_x*4+1] = uchar4_tmp.y;
    s_layer[thread_y*24+thread_x*4+2] = uchar4_tmp.z;
    s_layer[thread_y*24+thread_x*4+3] = uchar4_tmp.w;
    uchar4_tmp = ((uchar4*)d_in_layer)[s2_in_layer / 4];
    s_layer[(8+thread_y)*24+thread_x*4+0] = uchar4_tmp.x;
    s_layer[(8+thread_y)*24+thread_x*4+1] = uchar4_tmp.y;
    s_layer[(8+thread_y)*24+thread_x*4+2] = uchar4_tmp.z;
    s_layer[(8+thread_y)*24+thread_x*4+3] = uchar4_tmp.w;
    if(thread_y < 5){
      uchar4_tmp = ((uchar4*)d_in_layer)[s3_in_layer / 4];
      s_layer[(16+thread_y)*24+thread_x*4+0] = uchar4_tmp.x;
      s_layer[(16+thread_y)*24+thread_x*4+1] = uchar4_tmp.y;
      s_layer[(16+thread_y)*24+thread_x*4+2] = uchar4_tmp.z;
      s_layer[(16+thread_y)*24+thread_x*4+3] = uchar4_tmp.w;
    }
  }
  __syncthreads();
  if(index_y < out_y_dim && index_x < out_x_dim){
    for(int k = 0; k < filter_side; k++){
      for(int l = 0; l < filter_side; l++){
        d_y[r * out_x_dim * out_y_dim + index_y * out_x_dim + index_x] += s_layer[(thread_y * 2 + k) * 24 + (thread_x * 2 + l)] * s_weight[k * filter_side + l];
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
