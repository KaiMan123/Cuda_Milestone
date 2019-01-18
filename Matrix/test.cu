#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 16

const bool TransA = false;
const bool TransB = true;
const int M = 3;
const int N = 4;
const int K = 2;
const float alpha = 1;
const float A[6] = {  8, 1, 6,
                      5, 2, 3};
const float B[8] = {  1, 2, 3, 4,
                       5, 6, 7, 8};
const float beta = 1;
float C[12] = { 1, 1, 1, 1,
                1, 1, 1, 1, 
                1, 1, 1, 1};
float a[6];
float b[8];
float c[12];

__global__ void transpose(int row, int col, float *d_in, float *d_out){
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  int pos = 0, trans_pos = 0;
  if (index_y < row && index_x < col) {
      pos = index_y * col + index_x;
      trans_pos = index_x * row + index_y;
      d_out[pos] = d_in[trans_pos];
  }
  __syncthreads();

}
  
__global__ void multiplication(float a, int a_row, int a_col, int b_row, int b_col, float *d_out_a, float *d_out_b, float *d_out_c){
  int index_y = blockIdx.y*blockDim.y+threadIdx.y;
  int index_x = blockIdx.x*blockDim.x+threadIdx.x;
  if (index_y < a_row && index_x < b_col) {
    float temp = 0;
    for (int i = 0; i < a_col; i++) {
      temp += d_out_a[index_y * a_col + i] * d_out_b[i * b_col + index_x];
      //printf("%f and %f with %d + %d and ans is %f at %d\n",d_out_a[index_y * a_col + i], d_out_b[i * b_col + index_x], index_y * a_col + i, i * b_col + index_x, temp, index_y * a_col + index_x);
    }
    d_out_c[index_y * b_col + index_x] = a * temp;
  }
  __syncthreads();
}

__global__ void addition(float b, int row, int col, float *d_out_c, float *d_in_c, float *d_result){
  int index_y = blockIdx.y*blockDim.y+threadIdx.y;
  int index_x = blockIdx.x*blockDim.x+threadIdx.x;
  if(index_y < row && index_x < col){
    int pos = index_y * col + index_x;
    d_result[pos] = d_out_c[pos] + b * d_in_c[pos];
  }
  __syncthreads();
}


int main(){
  int a_row = (TransA) ? K : M;
  int a_col = (TransA) ? M : K;
  int b_row = (TransB) ? K : N;
  int b_col = (TransB) ? N : K;

  bool cuTransA = (TransA) ? true : false;
  bool cuTransB = (TransB) ? true : false;
  /*
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  */
  unsigned int size_A = M * K;
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int size_B = K * N;
  unsigned int mem_size_B = sizeof(float) * size_B;
  unsigned int size_C = M * N;
  unsigned int mem_size_C = sizeof(float) * size_C;

  float *d_in_A;
  cudaMalloc((void **) &d_in_A, mem_size_A);
  float *d_out_A;
  cudaMalloc((void **) &d_out_A, mem_size_A);

  float *d_in_B;
  cudaMalloc((void **) &d_in_B, mem_size_B);
  float *d_out_B;
  cudaMalloc((void **) &d_out_B, mem_size_B);

  float *d_in_C;
  cudaMalloc((void **) &d_in_C, mem_size_C);
  float *d_out_C;
  cudaMalloc((void **) &d_out_C, mem_size_C);

  float *d_result;
  cudaMalloc((void **) &d_result, mem_size_C);
  
  cudaMemcpy(d_in_A, A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_B, B, mem_size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_C, C, mem_size_C, cudaMemcpyHostToDevice);

  dim3 grid(1,1);
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);

  cudaDeviceSynchronize();

  if(cuTransA){
    cudaMemcpy(d_out_A, A, mem_size_A, cudaMemcpyHostToDevice);
  }else{
    grid.x = (a_col + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid.y = (a_row + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose<<<grid, block>>>(a_row, a_col, d_in_A, d_out_A);
  }
  cudaDeviceSynchronize();

  /**********       check output of a         *******************/
  /***************************************************************/
  cudaMemcpy(a, d_out_A, mem_size_A, cudaMemcpyDeviceToHost);
  for(int j = 0; j < a_row; j++){
    for(int i = 0; i < a_col; i++){
      cout << a[j * a_col + i] << " ";
    }
    cout << endl;
  }
  /***************************************************************/

  if(cuTransB){
    cudaMemcpy(d_out_B, B, mem_size_B, cudaMemcpyHostToDevice);
  }else{
    grid.x = (b_col + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid.y = (b_row + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose<<<grid, block>>>(b_row, b_col, d_in_B, d_out_B);
  }
  cudaDeviceSynchronize();

  /**********       check output of b         *******************/
  /***************************************************************
  cudaMemcpy(b, d_out_B, mem_size_B, cudaMemcpyDeviceToHost);
  for(int j = 0; j < b_row; j++){
    for(int i = 0; i < b_col; i++){
      cout << b[j * b_col + i] << " ";
    }
    cout << endl;
  }
  /***************************************************************/

  grid.x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  grid.y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  multiplication<<<grid, block>>>(alpha, M, K, K, N, d_out_A, d_out_B, d_out_C);
  cudaDeviceSynchronize();

  /**********       check output of c         *******************/
  /***************************************************************
  cudaMemcpy(c, d_out_C, mem_size_C, cudaMemcpyDeviceToHost);
  for(int j = 0; j < M; j++){
    for(int i = 0; i < N; i++){
      cout << c[j * N + i] << " ";
    }
    cout << endl;
  }
  /***************************************************************/

  grid.x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  grid.y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  addition<<<grid, block>>>(beta, M, N, d_out_C, d_in_C, d_result);
  cudaDeviceSynchronize();

  cudaMemcpy(C, d_result, mem_size_C, cudaMemcpyDeviceToHost);


  /**********       check output of C         *******************/
  /***************************************************************
  for(int j = 0; j < M; j++){
    for(int i = 0; i < N; i++){
      cout << C[j * N + i] << " ";
    }
    cout << endl;
  }
  /***************************************************************/

  cudaFree(d_in_A);
  cudaFree(d_in_B);
  cudaFree(d_in_C);
  cudaFree(d_out_A);
  cudaFree(d_out_B);
  cudaFree(d_out_C);
  cudaFree(d_result);
}