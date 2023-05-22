// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>      // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cublas_v2.h>
#include <cusparse.h>

namespace cg = cooperative_groups;

#define ENABLE_CPU_DEBUG_CODE 0
#define THREADS_PER_BLOCK 512

__device__ double grid_dot_result = 0.0;

__device__ double atomicAdd_(double *address, double val)
{
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand() / RAND_MAX + 10.0f;
    val[1] = (float)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i - 1] + 3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i - 1) * 3 + 2;
        J[start] = i - 1;
        J[start + 1] = i;

        if (i < N - 1)
        {
            J[start + 2] = i + 1;
        }

        val[start] = val[start - 1];
        val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

        if (i < N - 1)
        {
            val[start + 2] = (float)rand() / RAND_MAX;
        }
    }

    I[N] = nz;
}

void genSparseMat()
{
    
}

// I - contains location of the given non-zero element in the row of the matrix
// J - contains location of the given non-zero element in the column of the
// matrix val - contains values of the given non-zero elements of the matrix
// inputVecX - input vector to be multiplied
// outputVecY - resultant vector
void cpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha,
             float *inputVecX, float *outputVecY)
{
    for (int i = 0; i < num_rows; i++)
    {
        int num_elems_this_row = I[i + 1] - I[i];

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++)
        {
            output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
        }
        outputVecY[i] = output;
    }

    return;
}

double dotProduct(float *vecA, float *vecB, int size)
{
    double result = 0.0;

    for (int i = 0; i < size; i++)
    {
        result = result + (vecA[i] * vecB[i]);
    }

    return result;
}

void scaleVector(float *vec, float alpha, int size)
{
    for (int i = 0; i < size; i++)
    {
        vec[i] = alpha * vec[i];
    }
}

void saxpy(float *x, float *y, float a, int size)
{
    for (int i = 0; i < size; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

void cpuConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p,
                      float *r, int nnz, int N, float tol)
{
    int max_iter = 10000;

    float alpha = 1.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;

    cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
    saxpy(Ax, r, alpham1, N);

    float r1 = dotProduct(r, r, N);

    int k = 1;

    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            scaleVector(p, b, N);

            saxpy(r, p, alpha, N);
        }
        else
        {
            for (int i = 0; i < N; i++)
                p[i] = r[i];
        }

        cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

        float dot = dotProduct(p, Ax, N);
        a = r1 / dot;

        saxpy(p, x, a, N);
        na = -a;
        saxpy(Ax, r, na, N);

        r0 = r1;
        r1 = dotProduct(r, r, N);

        printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
}

__global__ void gpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, int startIdx,
                        float *inputVecX, float *outputVecY)
{
    cg::grid_group grid = cg::this_grid();

    for (int i = grid.thread_rank(); i < num_rows; i += grid.size())
    {
        int globalIdx = startIdx + i;
        int row_elem = I[globalIdx];
        int next_row_elem = I[globalIdx + 1];
        int num_elems_this_row = next_row_elem - row_elem;

        float output = 0.0;
        for (int j = 0; j < num_elems_this_row; j++)
        {
            output += val[row_elem + j] * inputVecX[J[row_elem + j]];
        }

        outputVecY[globalIdx] = output;
        // printf("%d Ax[%d] = %f\n", globalIdx, globalIdx, output);
    }
}

__global__ void gpuSaxpy(float *x, float *y, float a, int size, int startIdx)
{

    cg::grid_group grid = cg::this_grid();
    for (int i = grid.thread_rank(); i < size; i += grid.size())
    {
        int globalIdx = startIdx + i;
        y[globalIdx] = a * x[globalIdx] + y[globalIdx];
    }
}

__device__ void gpuScaleVectorAndSaxpy(float *x, float *y, float a, float scale, int size, int startIdx)
{
    cg::grid_group grid = cg::this_grid();
    for (int i = grid.thread_rank(); i < size; i += grid.size())
    {
        int globalIdx = startIdx + i;
        y[i] = a * x[i] + scale * y[i];
    }
}

// __global__ void gpuDotProduct(double* result, float* vecA, float* vecB, int size, int startIdx) {
//     cg::grid_group grid = cg::this_grid();

//     for (int i = grid.thread_rank(); i < size; i += grid.size()) {
//         int globalIdx = startIdx + i;
//         double tmp = static_cast<double>(vecA[globalIdx] * vecB[globalIdx]);
//         atomicAdd_(result, tmp);
//     }
// }

__global__ void gpuDotProduct(double *result, float *vecA, float *vecB, int size, int startIdx)
{
    __shared__ double tmp[THREADS_PER_BLOCK];

    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    double temp_sum = 0.0;
    for (int i = grid.thread_rank(); i < size; i += grid.size())
    {
        int globalIdx = startIdx + i;
        temp_sum += static_cast<double>(vecA[globalIdx] * vecB[globalIdx]);
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    double beta = temp_sum;
    double temp;

    for (int i = tile32.size() / 2; i > 0; i >>= 1)
    {
        if (tile32.thread_rank() < i)
        {
            temp = tmp[cta.thread_rank() + i];
            beta += temp;
            tmp[cta.thread_rank()] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0)
    {
        beta = 0.0;
        for (int i = 0; i < cta.size(); i += tile32.size())
        {
            beta += tmp[i];
        }
        atomicAdd_(result, beta);
    }
}

class SPMat
{
public:
    int *I;
    int *J;
    int N;
    int nz;
    float *val;

public:
    void init(int size, int noneZero)
    {
        N = size;
        nz = noneZero;
        mallocGpuMem();
    }

    void createSparseMat(cusparseSpMatDescr_t& mat)
    {
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreateCsr(&mat, N, N, nz, I, J, val,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        // checkCudaErrors(cusparseStatus);
    }

    void mallocGpuMem()
    {
        cudaMalloc(&I, (N + 1) * sizeof(int));
        cudaMalloc(&J, nz * sizeof(int));
        cudaMalloc(&val, nz * sizeof(float));
    }

    void destroy()
    {
        cudaFree(I);
        cudaFree(J);
        cudaFree(val);
    }
};

class CGData
{
public:
    float *r;
    float *p;
    float *Ax;
    float *x;
    float *dotProduct;
    float *alpha;
    float *alpham1;
    float *beta;
    float *b;
    float *a;
    float *na;

    int size;
    int localNumRow;
    int deviceId;
    int global_begin_idx;
    cudaStream_t stream;

    cublasHandle_t cublasHandle = 0;
    cusparseHandle_t cusparseHandle = 0;

    
    cusparseSpMatDescr_t matA = NULL;
    cusparseDnVecDescr_t vecx = NULL;
    cusparseDnVecDescr_t vecp = NULL;
    cusparseDnVecDescr_t vecAx = NULL;

    void *cuSparseWorkBuffer = NULL;

    SPMat spMat;

public:
    void init(int id, int size_, int nnz, int gbi, int numRow)
    {
        deviceId = id;
        size = size_;
        global_begin_idx = gbi;
        localNumRow = numRow;
        spMat.init(size_, nnz);
        mallocGpuMem();

        cusparseStatus_t cusparseStatus;
        cublasStatus_t cublasStatus;
        cublasStatus = cublasCreate(&cublasHandle);
        // checkCudaErrors(cublasStatus);

        /* Get handle to the CUSPARSE context */
        cusparseStatus = cusparseCreate(&cusparseHandle);
        // checkCudaErrors(cusparseStatus);

        cusparseMatDescr_t descr = 0;
        cusparseStatus = cusparseCreateMatDescr(&descr);
        // checkCudaErrors(cusparseStatus);

        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    }

    void createSparseMatVec()
    {
        spMat.createSparseMat(matA);
        
        cusparseCreateDnVec(&vecx, size, x, CUDA_R_32F);
        cusparseCreateDnVec(&vecp, size, p, CUDA_R_32F);
        cusparseCreateDnVec(&vecAx, size, Ax, CUDA_R_32F);
    }

    void createSpMVBuffer(const void* alpha, const void* beta)
    {
        checkCudaErrors(cudaSetDevice(deviceId));
        /* Allocate workspace for cuSPARSE */
        size_t bufferSize = 0;
        cusparseSpMV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecx,
            beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        
        checkCudaErrors(cudaMalloc(&cuSparseWorkBuffer, bufferSize));
    }

    void createSpMVBuffer()
    {
        checkCudaErrors(cudaSetDevice(deviceId));
        /* Allocate workspace for cuSPARSE */
        size_t bufferSize = 0;
        cusparseSpMV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, matA, vecx,
            beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        
        checkCudaErrors(cudaMalloc(&cuSparseWorkBuffer, bufferSize));
    }

    void spMV(const void* alpha, const void* beta)
    {
        checkCudaErrors(cudaSetDevice(deviceId));
        cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            alpha, matA, vecx, beta, vecAx, CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, cuSparseWorkBuffer);
    }

    void spMV()
    {
        checkCudaErrors(cudaSetDevice(deviceId));
        cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            alpha, matA, vecx, beta, vecAx, CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, cuSparseWorkBuffer);
    }

    void destroySpMVBuffer()
    {
        checkCudaErrors(cudaFree(cuSparseWorkBuffer));
    }

    // void setDeviceId(int id) { deviceId = id; }
    // void setSize(int size_) { size = size_; }
    // void setGlobalBeginIdx(int idx) { global_begin_idx = idx; }

    void mallocGpuMem()
    {
        checkCudaErrors(cudaSetDevice(deviceId));
        cudaMalloc(&r, size * sizeof(float));
        cudaMalloc(&p, size * sizeof(float));
        cudaMalloc(&Ax, size * sizeof(float));
        cudaMalloc(&x, size * sizeof(float));
        cudaMalloc(&dotProduct, sizeof(float));
        cudaMalloc(&alpha, sizeof(float));
        cudaMalloc(&alpham1, sizeof(float));
        cudaMalloc(&beta, sizeof(float));
        cudaMalloc(&b, sizeof(float));
        cudaMalloc(&a, sizeof(float));
        cudaMalloc(&na, sizeof(float));
        spMat.mallocGpuMem();
    }

    void destroy()
    {
        cudaFree(r);
        cudaFree(p);
        cudaFree(Ax);
        cudaFree(x);
        cudaFree(dotProduct);
        cudaFree(alpha);
        cudaFree(alpham1);
        cudaFree(beta);
        cudaFree(b);
        cudaFree(a);
        cudaFree(na);
        destroySpMVBuffer();
        spMat.destroy();
    }
};

class MultiCGData
{
public:
    int numDevice;
    int numRow;
    CGData *cgData;
    cudaStream_t *stream;

    MultiCGData(const int nd, const int N, const int nnz)
    {
        numDevice = nd;
        numRow = N;
        cgData = new CGData[numDevice];

        int rowPerDevice = N / nd;
        int rowLastDevice = N - rowPerDevice * (nd - 1);

        for (int i = 0; i < numDevice; i++)
        {
            int rowDevice = i == numDevice - 1 ? rowLastDevice : rowPerDevice;
            cgData[i].init(i, N, nnz, i * rowPerDevice, rowDevice);
        }

        stream = new cudaStream_t[numDevice];
    }

    void createSparseMatVec()
    {
        for(int i = 0; i < numDevice; i++)
        {
            cgData[i].createSparseMatVec();
        }
    }

    void createSpMVBuffer(const void* alpha, const void* beta)
    {
        for(int i = 0; i < numDevice; i++)
        {
            cgData[i].createSpMVBuffer(alpha, beta);
        }
    }

    void createSpMVBuffer()
    {
        for(int i = 0; i < numDevice; i++)
        {
            cgData[i].createSpMVBuffer();
        }
    }

    void spMV(const void* alpha, const void* beta)
    {
        for(int i = 0; i < numDevice; i++)
        {
            cgData[i].spMV(alpha, beta);
        }
    }

    void spMV()
    {
        for(int i = 0; i < numDevice; i++)
        {
            cgData[i].spMV();
        }
    }

    void syncDevice(int deviceId)
    {
        checkCudaErrors(cudaSetDevice(deviceId));
        cudaDeviceSynchronize();
    }

    void syncDevice()
    {
        for(int i = 0; i < numDevice; i++)
        {
            checkCudaErrors(cudaSetDevice(i));
            cudaDeviceSynchronize();
        }
    }

    void createCudaStream()
    {
        for (int i = 0; i < numDevice; i++)
        {
            cudaSetDevice(i);
            checkCudaErrors(cudaStreamCreate(&stream[i]));
        }
    }

    void destroyCudaStream()
    {
        for (int i = 0; i < numDevice; i++)
        {
            checkCudaErrors(cudaStreamDestroy(stream[i]));
        }
    }

    ~MultiCGData()
    {
        for (int i = 0; i < numDevice; i++)
        {
            cgData[i].destroy();
            // delete[] cgData;
            // delete[] stream;
        }
    }
};

void multiGpuSpMV(MultiCGData &multiCGData)
{
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData &data = multiCGData.cgData[i];

        checkCudaErrors(cudaSetDevice(data.deviceId));

        void *kernelArgs[] = {
            (void *)&data.spMat.I,
            (void *)&data.spMat.J,
            (void *)&data.spMat.val,
            (void *)&data.spMat.nz,
            (void *)&data.localNumRow,
            (void *)&data.global_begin_idx,
            (void *)&data.x,
            (void *)&data.Ax};

        std::cout << "device " << data.deviceId << std::endl;

        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)gpuSpMV, dimGrid, dimBlock, kernelArgs,
            0, multiCGData.stream[data.deviceId]));
        getLastCudaError("gpuSpMV execution failed");
    }
}

void multiGpuSaxpy(MultiCGData &multiCGData)
{
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData &data = multiCGData.cgData[i];

        checkCudaErrors(cudaSetDevice(data.deviceId));

        float alpha = -1.0;
        void *sapyArgs[] = {
            &data.Ax,
            &data.r,
            &alpha,
            &data.localNumRow,
            &data.global_begin_idx};

        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)gpuSaxpy, dimGrid, dimBlock, sapyArgs,
            0, multiCGData.stream[data.deviceId]));
        getLastCudaError("gpuSaxpy execution failed");
    }
}

void multiGpuSaxpyCuSparse(MultiCGData &multiCGData)
{

}

double multiGpuDotProductRR(MultiCGData &multiCGData)
{
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    std::vector<double> dotResult(multiCGData.numDevice);
    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData &data = multiCGData.cgData[i];

        checkCudaErrors(cudaSetDevice(data.deviceId));
        checkCudaErrors(cudaMemsetAsync(data.dotProduct, 0, sizeof(double), multiCGData.stream[data.deviceId]));

        void *dotProductArgs[] = {
            &data.dotProduct,
            &data.r,
            &data.r,
            &data.localNumRow,
            &data.global_begin_idx};

        int sMemSize = sizeof(double) * THREADS_PER_BLOCK;
        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)gpuDotProduct, dimGrid, dimBlock, dotProductArgs,
            sMemSize, multiCGData.stream[data.deviceId]));
        getLastCudaError("gpuDotProduct execution failed");

        checkCudaErrors(cudaMemcpyAsync(dotResult.data() + data.deviceId, data.dotProduct, sizeof(double), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
    }

    double sum = 0.0;
    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceSynchronize();
        sum += dotResult[i];
    }
    return sum;
}

float multiGpuDotProductRRCuSparse(MultiCGData &multiCGData)
{
    std::vector<float> dotResult(multiCGData.numDevice);
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData& data = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(data.deviceId));
        cublasSdot(data.cublasHandle, data.localNumRow, data.r + data.global_begin_idx, 1,
                    data.r + data.global_begin_idx , 1, data.dotProduct);
        checkCudaErrors(cudaMemcpyAsync(&dotResult[i], data.dotProduct, sizeof(float), cudaMemcpyDefault,multiCGData.stream[i]));
    }

    float sum = 0.0;
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData& data = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(data.deviceId));
        checkCudaErrors(cudaDeviceSynchronize());
        sum += dotResult[i];
    }
    return sum;
}

double multiGpuDotProductPAx(MultiCGData &multiCGData)
{
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    std::vector<double> dotResult(multiCGData.numDevice);
    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData &data = multiCGData.cgData[i];

        checkCudaErrors(cudaSetDevice(data.deviceId));
        checkCudaErrors(cudaMemsetAsync(data.dotProduct, 0, sizeof(double), multiCGData.stream[data.deviceId]));

        void *dotProductArgs[] = {
            &data.dotProduct,
            &data.p,
            &data.Ax,
            &data.localNumRow,
            &data.global_begin_idx};

        int sMemSize = sizeof(double) * THREADS_PER_BLOCK;
        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)gpuDotProduct, dimGrid, dimBlock, dotProductArgs,
            sMemSize, multiCGData.stream[data.deviceId]));
        getLastCudaError("gpuDotProduct execution failed");

        checkCudaErrors(cudaMemcpyAsync(dotResult.data() + data.deviceId, data.dotProduct, sizeof(double), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
    }

    double sum = 0.0;
    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceSynchronize();
        sum += dotResult[i];
    }
    return sum;
}

float multiGpuDotProductPAxCuSparse(MultiCGData& multiCGData)
{
    std::vector<float> dotResult(multiCGData.numDevice);
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData& data = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(data.deviceId));
        cublasSdot(data.cublasHandle, data.localNumRow, data.p + data.global_begin_idx, 1,
                    data.Ax + data.global_begin_idx , 1, data.dotProduct);
        checkCudaErrors(cudaMemcpyAsync(&dotResult[i], data.dotProduct, sizeof(float), cudaMemcpyDefault,multiCGData.stream[i]));
    }

    float sum = 0.0;
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData& data = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(data.deviceId));
        checkCudaErrors(cudaDeviceSynchronize());
        sum += dotResult[i];
    }
    return sum;
}


__global__ void setValue(float *x, float val, int count)
{
    cg::grid_group grid = cg::this_grid();
    for (int i = grid.thread_rank(); i < count; i += grid.size())
    {
        x[i] = val;
    }
}

__global__ void gpuScaleVector(float* vec, float alpha, int size, int startIdx) {
    const cg::grid_group& grid = cg::this_grid();
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        int globalIdx = i + startIdx;
        vec[globalIdx] = alpha * vec[globalIdx];
    }
}

void multiScaleP(MultiCGData& multiCGData, float s)
{
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData &data = multiCGData.cgData[i];

        checkCudaErrors(cudaSetDevice(data.deviceId));

        void *args[] = {
            &data.p,
            &s,
            &data.localNumRow,
            &data.global_begin_idx};

        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)gpuScaleVector, dimGrid, dimBlock, args,
            0, multiCGData.stream[data.deviceId]));
    }    
}

void multiGpuLaunch(void* kernel, void** args[], int numDevice, cudaStream_t* streams, int sMem = 0)
{
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    for(int i = 0; i < numDevice; i++)
    {
        checkCudaErrors(cudaSetDevice(i));

        checkCudaErrors(cudaLaunchCooperativeKernel(
            kernel, dimGrid, dimBlock, args[i],
            sMem, streams[i]));        
    }

}

void initGpuData(MultiCGData &multiCGData)
{
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        CGData &data = multiCGData.cgData[i];

        checkCudaErrors(cudaSetDevice(data.deviceId));
        cudaMemset(data.x, 0, multiCGData.numRow * sizeof(float));

        float val = 1.0;
        void *args[] = {&data.r, &val, &multiCGData.numRow};
        checkCudaErrors(cudaLaunchCooperativeKernel(
            (void *)setValue, dimGrid, dimBlock, args,
            0, multiCGData.stream[data.deviceId]));
        getLastCudaError("gpuSaxpy execution failed");
    }
}



void ConjugateGrad1(int *I, int *J, float *val, float *x, float *Ax, float *p,
                   float *r, int nnz, int N, float tol, MultiCGData &multiCGData)
{
    int max_iter = 10000;

    float alpha = 1.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;

    initGpuData(multiCGData);


    cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
    multiGpuSpMV(multiCGData);

    saxpy(Ax, r, alpham1, N);
    multiGpuSaxpy(multiCGData);

    float r1 = dotProduct(r, r, N);

    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    auto r1_ = multiGpuDotProductRR(multiCGData);


    int k = 1;

    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            scaleVector(p, b, N);
            // gpu
            multiScaleP(multiCGData, b);

            saxpy(r, p, alpha, N);
            
            // multi gpu saxpy
            for (int i = 0; i < multiCGData.numDevice; i++)
            {
                CGData &data = multiCGData.cgData[i];

                checkCudaErrors(cudaSetDevice(data.deviceId));

                void *args[] = {&data.r, &data.p, &alpha, &data.localNumRow, &data.global_begin_idx};

                checkCudaErrors(cudaLaunchCooperativeKernel(
                    (void *)gpuSaxpy, dimGrid, dimBlock, args,
                    0, multiCGData.stream[data.deviceId]));
            }
            // end gpu saxpy            
        }
        else
        {
            for (int i = 0; i < N; i++)
                p[i] = r[i];
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                CGData &data = multiCGData.cgData[i];
                checkCudaErrors(cudaMemcpyAsync(data.p + data.global_begin_idx, data.r, sizeof(float) * data.localNumRow, cudaMemcpyDefault, multiCGData.stream[i]));
            }
        }

        // sync p
        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];
            // transfer data to other device
            for (int j = 1; j < multiCGData.numDevice; j++)
            {
                auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
                checkCudaErrors(cudaMemcpyAsync(nextData.p + data.global_begin_idx, data.p + data.global_begin_idx,
                                                data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
            }
        }
        multiCGData.syncDevice();

        std::vector<float> x_tmp(multiCGData.numRow, 2);
        std::vector<float> p_tmp(multiCGData.numRow, 2);
        std::vector<float> r_tmp(multiCGData.numRow, 2);
        
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            auto &data = multiCGData.cgData[i];
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(x_tmp.data() + data.global_begin_idx, data.x + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
            checkCudaErrors(cudaMemcpy(p_tmp.data() + data.global_begin_idx, data.p + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
            checkCudaErrors(cudaMemcpy(r_tmp.data() + data.global_begin_idx, data.r + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
        }

        cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

        // gpu spmv
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];

            checkCudaErrors(cudaSetDevice(data.deviceId));

            void *kernelArgs[] = {
                (void *)&data.spMat.I,
                (void *)&data.spMat.J,
                (void *)&data.spMat.val,
                (void *)&data.spMat.nz,
                (void *)&data.localNumRow,
                (void *)&data.global_begin_idx,
                (void *)&data.p,
                (void *)&data.Ax};

            checkCudaErrors(cudaLaunchCooperativeKernel(
                (void *)gpuSpMV, dimGrid, dimBlock, kernelArgs,
                0, multiCGData.stream[data.deviceId]));
            getLastCudaError("gpuSpMV execution failed");
        }
        // end gpu spmv

        // // sync Ax
        // for(int i = 0; i < multiCGData.numDevice; i++)
        // {
        //     CGData &data = multiCGData.cgData[i];
        //     // transfer data to other device
        //     for (int j = 1; j < multiCGData.numDevice; j++)
        //     {
        //         auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
        //         checkCudaErrors(cudaMemcpyAsync(nextData.Ax + data.global_begin_idx, data.Ax + data.global_begin_idx,
        //                                         data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
        //     }
        // }
        // multiCGData.syncDevice();

        std::vector<float> tmp(multiCGData.numRow);
        std::vector<float> tmp_cpu(multiCGData.numRow);

        memcpy(tmp_cpu.data(), Ax, multiCGData.numRow * sizeof(float));
        cudaMemcpy(tmp.data(), multiCGData.cgData[0].Ax, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(tmp.data(), multiCGData.cgData[1].Ax, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);

        memcpy(tmp_cpu.data(), p, multiCGData.numRow * sizeof(float));

        cudaMemcpy(tmp.data(), multiCGData.cgData[0].p, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(tmp.data(), multiCGData.cgData[1].p, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);        
        

        float dot = dotProduct(p, Ax, N);
        a = r1 / dot;
        
        auto dot_ = multiGpuDotProductPAx(multiCGData);

        saxpy(p, x, a, N);

        // gpu saxpy
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];

            checkCudaErrors(cudaSetDevice(data.deviceId));

            void *sapyArgs[] = {
                &data.p,
                &data.x,
                &a,
                &data.localNumRow,
                &data.global_begin_idx};

            checkCudaErrors(cudaLaunchCooperativeKernel(
                (void *)gpuSaxpy, dimGrid, dimBlock, sapyArgs,
                0, multiCGData.stream[data.deviceId]));
            getLastCudaError("gpuSaxpy execution failed");
        }
        // end gpu saxpy

        na = -a;
        saxpy(Ax, r, na, N);

        // gpu saxpy
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];

            checkCudaErrors(cudaSetDevice(data.deviceId));

            void *sapyArgs[] = {
                &data.Ax,
                &data.r,
                &na,
                &data.localNumRow,
                &data.global_begin_idx};

            checkCudaErrors(cudaLaunchCooperativeKernel(
                (void *)gpuSaxpy, dimGrid, dimBlock, sapyArgs,
                0, multiCGData.stream[data.deviceId]));
            getLastCudaError("gpuSaxpy execution failed");
        }
        // end gpu saxpy

        r0 = r1;
        r1 = dotProduct(r, r, N);

        r1_ = multiGpuDotProductRR(multiCGData);

        printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
}



double ConjugateGradGPU(MultiCGData &multiCGData, float tol)
{
    int max_iter = 10000;

    float alpha = 1.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;

    initGpuData(multiCGData);


    //cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
    multiGpuSpMV(multiCGData);

    //saxpy(Ax, r, alpham1, N);
    multiGpuSaxpy(multiCGData);

    //float r1 = dotProduct(r, r, N);

    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    auto r1 = multiGpuDotProductRR(multiCGData);


    int k = 1;

    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            //scaleVector(p, b, N);
            // gpu
            multiScaleP(multiCGData, b);

            //saxpy(r, p, alpha, N);
            
            // multi gpu saxpy
            for (int i = 0; i < multiCGData.numDevice; i++)
            {
                CGData &data = multiCGData.cgData[i];

                checkCudaErrors(cudaSetDevice(data.deviceId));

                void *args[] = {&data.r, &data.p, &alpha, &data.localNumRow, &data.global_begin_idx};

                checkCudaErrors(cudaLaunchCooperativeKernel(
                    (void *)gpuSaxpy, dimGrid, dimBlock, args,
                    0, multiCGData.stream[data.deviceId]));
            }
            // end gpu saxpy            
        }
        else
        {
            // for (int i = 0; i < N; i++)
            //     p[i] = r[i];
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                CGData &data = multiCGData.cgData[i];
                checkCudaErrors(cudaMemcpyAsync(data.p, data.r, sizeof(float) * multiCGData.numRow, cudaMemcpyDefault, multiCGData.stream[i]));
            }
        }

        // sync p
        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];
            // transfer data to other device
            for (int j = 1; j < multiCGData.numDevice; j++)
            {
                auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
                checkCudaErrors(cudaMemcpyAsync(nextData.p + data.global_begin_idx, data.p + data.global_begin_idx,
                                                data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
            }
        }
        multiCGData.syncDevice();


        //cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

        // gpu spmv
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];

            checkCudaErrors(cudaSetDevice(data.deviceId));

            void *kernelArgs[] = {
                (void *)&data.spMat.I,
                (void *)&data.spMat.J,
                (void *)&data.spMat.val,
                (void *)&data.spMat.nz,
                (void *)&data.localNumRow,
                (void *)&data.global_begin_idx,
                (void *)&data.p,
                (void *)&data.Ax};

            checkCudaErrors(cudaLaunchCooperativeKernel(
                (void *)gpuSpMV, dimGrid, dimBlock, kernelArgs,
                0, multiCGData.stream[data.deviceId]));
            getLastCudaError("gpuSpMV execution failed");
        }
        // end gpu spmv

        // // sync Ax
        // for(int i = 0; i < multiCGData.numDevice; i++)
        // {
        //     CGData &data = multiCGData.cgData[i];
        //     // transfer data to other device
        //     for (int j = 1; j < multiCGData.numDevice; j++)
        //     {
        //         auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
        //         checkCudaErrors(cudaMemcpyAsync(nextData.Ax + data.global_begin_idx, data.Ax + data.global_begin_idx,
        //                                         data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
        //     }
        // }
        // multiCGData.syncDevice();

        //float dot = dotProduct(p, Ax, N);

        auto dot = multiGpuDotProductPAx(multiCGData);
        a = r1 / dot;
        
        //saxpy(p, x, a, N);

        // gpu saxpy
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];

            checkCudaErrors(cudaSetDevice(data.deviceId));

            void *sapyArgs[] = {
                &data.p,
                &data.x,
                &a,
                &data.localNumRow,
                &data.global_begin_idx};

            checkCudaErrors(cudaLaunchCooperativeKernel(
                (void *)gpuSaxpy, dimGrid, dimBlock, sapyArgs,
                0, multiCGData.stream[data.deviceId]));
            getLastCudaError("gpuSaxpy execution failed");
        }
        // end gpu saxpy

        std::vector<float> x_tmp(multiCGData.numRow);
        std::vector<float> p_tmp(multiCGData.numRow);
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            auto &data = multiCGData.cgData[i];
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(x_tmp.data() + data.deviceId, data.x + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
            checkCudaErrors(cudaMemcpy(p_tmp.data() + data.deviceId, data.p + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
        }


        na = -a;
        //saxpy(Ax, r, na, N);

        // gpu saxpy
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];

            checkCudaErrors(cudaSetDevice(data.deviceId));

            void *sapyArgs[] = {
                &data.Ax,
                &data.r,
                &na,
                &data.localNumRow,
                &data.global_begin_idx};

            checkCudaErrors(cudaLaunchCooperativeKernel(
                (void *)gpuSaxpy, dimGrid, dimBlock, sapyArgs,
                0, multiCGData.stream[data.deviceId]));
            getLastCudaError("gpuSaxpy execution failed");
        }
        // end gpu saxpy

        r0 = r1;
        //r1 = dotProduct(r, r, N);

        r1 = multiGpuDotProductRR(multiCGData);

        printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    return r1;
}


void ConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p,
                   float *r, int nnz, int N, float tol, MultiCGData &multiCGData)
{

    int max_iter = 10000;

    float alpha = 1.0;
    float beta = 0.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;
    

    cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
    // multiGpuSpMV(multiCGData);

    saxpy(Ax, r, alpham1, N);
    // multiGpuSaxpy(multiCGData);

    float r1 = dotProduct(r, r, N);


    initGpuData(multiCGData);
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& data = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(data.deviceId));

        cusparseSetPointerMode(data.cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);
        cublasSetPointerMode(data.cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
        cusparseSetStream(data.cusparseHandle, multiCGData.stream[i]);
        cublasSetStream(data.cublasHandle, multiCGData.stream[i]);

        checkCudaErrors(cudaMemcpyAsync(data.alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.alpham1, &alpham1, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.beta, &beta, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.b, &b, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
    }


    multiCGData.createSparseMatVec();
    multiCGData.createSpMVBuffer();
    multiCGData.spMV();

    // checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                             &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
    //                             CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& cgData = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(cgData.deviceId));
        cublasSaxpy(cgData.cublasHandle, cgData.localNumRow, cgData.alpham1, cgData.Ax + cgData.global_begin_idx, 1, cgData.r + cgData.global_begin_idx, 1);
    }

    // cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);

    multiCGData.syncDevice();

    std::vector<float> Ax_cpu(Ax, Ax + multiCGData.numRow);
    std::vector<float> r_cpu(r, r + multiCGData.numRow);
    std::vector<float> Ax_gpu(multiCGData.numRow);
    std::vector<float> r_gpu(multiCGData.numRow);
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& data = multiCGData.cgData[i];
        checkCudaErrors(cudaMemcpy(Ax_gpu.data(), data.Ax, sizeof(float) * multiCGData.numRow, cudaMemcpyDefault));
        checkCudaErrors(cudaMemcpy(r_gpu.data(), data.r, sizeof(float) * multiCGData.numRow, cudaMemcpyDefault));

    }


    float r1_ = multiGpuDotProductRRCuSparse(multiCGData);

    // for(int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     checkCudaErrors(cudaSetDevice(i));
    //     checkCudaErrors(cudaDeviceSynchronize());
    // }
    // auto r1_ = multiGpuDotProductRR(multiCGData);


    int k = 1;

    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            scaleVector(p, b, N);
            // gpu
            // multiScaleP(multiCGData, b);

            saxpy(r, p, alpha, N);
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                checkCudaErrors(cudaMemcpyAsync(data.b, &b, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
                cublasSscal(data.cublasHandle, data.localNumRow, data.b, data.p + data.global_begin_idx, 1);
                cublasSaxpy(data.cublasHandle, data.localNumRow, data.alpha, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
            // end gpu saxpy            
        }
        else
        {
            for (int i = 0; i < N; i++)
                p[i] = r[i];
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                cublasScopy(data.cublasHandle, data.localNumRow, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
        }

        // sync p
        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];
            // transfer data to other device
            for (int j = 1; j < multiCGData.numDevice; j++)
            {
                auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
                checkCudaErrors(cudaMemcpyAsync(nextData.p + data.global_begin_idx, data.p + data.global_begin_idx,
                                                data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
            }
        }
        multiCGData.syncDevice();

        std::vector<float> x_tmp(multiCGData.numRow, 2);
        std::vector<float> p_tmp(multiCGData.numRow, 2);
        std::vector<float> r_tmp(multiCGData.numRow, 2);
        
        for (int i = 0; i < multiCGData.numDevice; i++)
        {
            auto &data = multiCGData.cgData[i];
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(x_tmp.data() + data.global_begin_idx, data.x + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
            checkCudaErrors(cudaMemcpy(p_tmp.data() + data.global_begin_idx, data.p + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
            checkCudaErrors(cudaMemcpy(r_tmp.data() + data.global_begin_idx, data.r + data.deviceId, data.localNumRow * sizeof(float), cudaMemcpyDefault));
        }

        cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);

        // gpu spmv

        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto &data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            cusparseSpMV(
                data.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, data.alpha, data.matA, data.vecp,
                data.beta, data.vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, data.cuSparseWorkBuffer);            
        }

        std::vector<float> tmp(multiCGData.numRow);
        std::vector<float> tmp_cpu(multiCGData.numRow);

        memcpy(tmp_cpu.data(), Ax, multiCGData.numRow * sizeof(float));
        cudaMemcpy(tmp.data(), multiCGData.cgData[0].Ax, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(tmp.data(), multiCGData.cgData[1].Ax, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);

        memcpy(tmp_cpu.data(), p, multiCGData.numRow * sizeof(float));

        cudaMemcpy(tmp.data(), multiCGData.cgData[0].p, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(tmp.data(), multiCGData.cgData[1].p, multiCGData.numRow * sizeof(float), cudaMemcpyDefault);        
        

        float dot = dotProduct(p, Ax, N);
        a = r1 / dot;
        
        float dot_ = multiGpuDotProductPAxCuSparse(multiCGData);

        saxpy(p, x, a, N);

        na = -a;
        saxpy(Ax, r, na, N);

        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto& data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            checkCudaErrors(cudaMemcpyAsync(data.a, &a, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
            checkCudaErrors(cudaMemcpyAsync(data.na, &na, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
            cublasSaxpy(data.cublasHandle, data.localNumRow, data.a, data.p + data.global_begin_idx, 1, data.x + data.global_begin_idx, 1);
            cublasSaxpy(data.cublasHandle, data.localNumRow, data.na, data.Ax + data.global_begin_idx, 1, data.r + data.global_begin_idx, 1);
        }

        r0 = r1;
        r1 = dotProduct(r, r, N);

        r1_ = multiGpuDotProductRRCuSparse(multiCGData);

        printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
}


float ConjugateGradCuSparse(MultiCGData &multiCGData, float tol)
{

    // cusparseLoggerSetLevel(4);
    float alpha = 1.0;
    float alpham1 = -1.0;
    float beta = 0.0;
    float r0 = 0.0;
    float r1 = 0.0;
    int max_iter = 1000;

    initGpuData(multiCGData);

    multiCGData.createSparseMatVec();
    multiCGData.createSpMVBuffer(&alpha, &beta);
    multiCGData.spMV(&alpha, &beta);

    // checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                             &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
    //                             CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& cgData = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(cgData.deviceId));
        cublasSaxpy(cgData.cublasHandle, cgData.localNumRow, &alpham1, cgData.Ax + cgData.global_begin_idx, 1, cgData.r + cgData.global_begin_idx, 1);
    }

    // cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);


    r1 = multiGpuDotProductRRCuSparse(multiCGData);
    // cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);

    int k = 1;

    while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
            float b = r1 / r0;
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                cublasSscal(data.cublasHandle, data.localNumRow, &b, data.p + data.global_begin_idx, 1);
                cublasSaxpy(data.cublasHandle, data.localNumRow, &alpha, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
            // cublasStatus = cublasSscal(cublasHandle, N, &b, p, 1);
            // cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, r, 1, p, 1);
        } else {
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                cublasScopy(data.cublasHandle, data.localNumRow, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
            // cublasStatus = cublasScopy(cublasHandle, N, r, 1, p, 1);
        }


        // sync p
        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];
            // transfer data to other device
            for (int j = 1; j < multiCGData.numDevice; j++)
            {
                auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
                checkCudaErrors(cudaMemcpyAsync(nextData.p + data.global_begin_idx, data.p + data.global_begin_idx,
                                                data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
            }
        }
        multiCGData.syncDevice();
        
        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto &data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            cusparseSpMV(
                data.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, data.matA, data.vecp,
                &beta, data.vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, data.cuSparseWorkBuffer);            
        }
        // checkCudaErrors(cusparseSpMV(
        //     cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        //     &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        float dot = multiGpuDotProductPAxCuSparse(multiCGData);

        // cublasStatus = cublasSdot(cublasHandle, N, p, 1, Ax, 1, &dot);
        float a = r1 / dot;
        float na = -a;

        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto& data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            cublasSaxpy(data.cublasHandle, data.localNumRow, &a, data.p + data.global_begin_idx, 1, data.x + data.global_begin_idx, 1);
            cublasSaxpy(data.cublasHandle, data.localNumRow, &na, data.Ax + data.global_begin_idx, 1, data.r + data.global_begin_idx, 1);
        }

        // cublasStatus = cublasSaxpy(cublasHandle, N, &a, p, 1, x, 1);
        
        // cublasStatus = cublasSaxpy(cublasHandle, N, &na, Ax, 1, r, 1);

        r0 = r1;
        
        r1 = multiGpuDotProductRRCuSparse(multiCGData);
        // cublasStatus = cublasSdot(cublasHandle, N, r, 1, r, 1, &r1);
        // multiCGData.syncDevice();
        // cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
    return r1;
}

float ConjugateGradCuSparse2(MultiCGData &multiCGData, float tol)
{
    int max_iter = 100;

    float alpha = 1.0;
    float beta = 0.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;
 
    initGpuData(multiCGData);
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& data = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(data.deviceId));

        cusparseSetPointerMode(data.cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);
        cublasSetPointerMode(data.cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
        cusparseSetStream(data.cusparseHandle, multiCGData.stream[i]);
        cublasSetStream(data.cublasHandle, multiCGData.stream[i]);

        checkCudaErrors(cudaMemcpyAsync(data.alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.alpham1, &alpham1, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.beta, &beta, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.b, &b, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
    }


    multiCGData.createSparseMatVec();
    multiCGData.createSpMVBuffer();


    auto startTime = std::chrono::high_resolution_clock::now();

    multiCGData.spMV();

    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& cgData = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(cgData.deviceId));
        cublasSaxpy(cgData.cublasHandle, cgData.localNumRow, cgData.alpham1, cgData.Ax + cgData.global_begin_idx, 1, cgData.r + cgData.global_begin_idx, 1);
    }

    // cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);

    // multiCGData.syncDevice();

    // std::vector<float> Ax_cpu(Ax, Ax + multiCGData.numRow);
    // std::vector<float> r_cpu(r, r + multiCGData.numRow);
    // std::vector<float> Ax_gpu(multiCGData.numRow);
    // std::vector<float> r_gpu(multiCGData.numRow);
    // for(int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     auto& data = multiCGData.cgData[i];
    //     checkCudaErrors(cudaMemcpy(Ax_gpu.data(), data.Ax, sizeof(float) * multiCGData.numRow, cudaMemcpyDefault));
    //     checkCudaErrors(cudaMemcpy(r_gpu.data(), data.r, sizeof(float) * multiCGData.numRow, cudaMemcpyDefault));

    // }


    float r1 = multiGpuDotProductRRCuSparse(multiCGData);

    // for(int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     checkCudaErrors(cudaSetDevice(i));
    //     checkCudaErrors(cudaDeviceSynchronize());
    // }
    // auto r1_ = multiGpuDotProductRR(multiCGData);


    int k = 1;

    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                checkCudaErrors(cudaMemcpyAsync(data.b, &b, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
                cublasSscal(data.cublasHandle, data.localNumRow, data.b, data.p + data.global_begin_idx, 1);
                cublasSaxpy(data.cublasHandle, data.localNumRow, data.alpha, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
            // end gpu saxpy            
        }
        else
        {
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                cublasScopy(data.cublasHandle, data.localNumRow, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
        }

        // sync p
        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];
            // transfer data to other device
            for (int j = 1; j < multiCGData.numDevice; j++)
            {
                auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
                checkCudaErrors(cudaMemcpyAsync(nextData.p + data.global_begin_idx, data.p + data.global_begin_idx,
                                                data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
            }
        }
        multiCGData.syncDevice();

        // gpu spmv

        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto &data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            cusparseSpMV(
                data.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, data.alpha, data.matA, data.vecp,
                data.beta, data.vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, data.cuSparseWorkBuffer);            
        }
        

        float dot = multiGpuDotProductPAxCuSparse(multiCGData);
        a = r1 / dot;
        na = -a;

        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto& data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            checkCudaErrors(cudaMemcpyAsync(data.a, &a, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
            checkCudaErrors(cudaMemcpyAsync(data.na, &na, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
            cublasSaxpy(data.cublasHandle, data.localNumRow, data.a, data.p + data.global_begin_idx, 1, data.x + data.global_begin_idx, 1);
            cublasSaxpy(data.cublasHandle, data.localNumRow, data.na, data.Ax + data.global_begin_idx, 1, data.r + data.global_begin_idx, 1);
        }

        r0 = r1;

        r1 = multiGpuDotProductRRCuSparse(multiCGData);

        // printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "cg time: " << delta.count() / 1000.f << std::endl;
    std::cout << "iteration: " << k << std::endl;
    return r1;
}


float ConjugateGradCuSparse3(MultiCGData &multiCGData, float tol)
{
    int max_iter = 100;

    float alpha = 1.0;
    float beta = 0.0;
    float alpham1 = -1.0;
    float r0 = 0.0, b, a, na;
 
    initGpuData(multiCGData);
    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& data = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(data.deviceId));

        cusparseSetPointerMode(data.cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);
        cublasSetPointerMode(data.cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
        cusparseSetStream(data.cusparseHandle, multiCGData.stream[i]);
        cublasSetStream(data.cublasHandle, multiCGData.stream[i]);

        checkCudaErrors(cudaMemcpyAsync(data.alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.alpham1, &alpham1, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.beta, &beta, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
        checkCudaErrors(cudaMemcpyAsync(data.b, &b, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
    }


    multiCGData.createSparseMatVec();
    multiCGData.createSpMVBuffer();


    auto startTime = std::chrono::high_resolution_clock::now();

    multiCGData.spMV();

    for(int i = 0; i < multiCGData.numDevice; i++)
    {
        auto& cgData = multiCGData.cgData[i];
        checkCudaErrors(cudaSetDevice(cgData.deviceId));
        cublasSaxpy(cgData.cublasHandle, cgData.localNumRow, cgData.alpham1, cgData.Ax + cgData.global_begin_idx, 1, cgData.r + cgData.global_begin_idx, 1);
    }

    // cublasSaxpy(cublasHandle, N, &alpham1, Ax, 1, r, 1);

    // multiCGData.syncDevice();

    // std::vector<float> Ax_cpu(Ax, Ax + multiCGData.numRow);
    // std::vector<float> r_cpu(r, r + multiCGData.numRow);
    // std::vector<float> Ax_gpu(multiCGData.numRow);
    // std::vector<float> r_gpu(multiCGData.numRow);
    // for(int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     auto& data = multiCGData.cgData[i];
    //     checkCudaErrors(cudaMemcpy(Ax_gpu.data(), data.Ax, sizeof(float) * multiCGData.numRow, cudaMemcpyDefault));
    //     checkCudaErrors(cudaMemcpy(r_gpu.data(), data.r, sizeof(float) * multiCGData.numRow, cudaMemcpyDefault));

    // }


    float r1 = multiGpuDotProductRRCuSparse(multiCGData);

    // for(int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     checkCudaErrors(cudaSetDevice(i));
    //     checkCudaErrors(cudaDeviceSynchronize());
    // }
    // auto r1_ = multiGpuDotProductRR(multiCGData);


    int k = 1;

    while (r1 > tol * tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                checkCudaErrors(cudaMemcpyAsync(data.b, &b, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
                cublasSscal(data.cublasHandle, data.localNumRow, data.b, data.p + data.global_begin_idx, 1);
                cublasSaxpy(data.cublasHandle, data.localNumRow, data.alpha, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
            // end gpu saxpy            
        }
        else
        {
            
            for(int i = 0; i < multiCGData.numDevice; i++)
            {
                auto& data = multiCGData.cgData[i];
                checkCudaErrors(cudaSetDevice(data.deviceId));
                cublasScopy(data.cublasHandle, data.localNumRow, data.r + data.global_begin_idx, 1, data.p + data.global_begin_idx, 1);
            }
        }

        // sync p
        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            CGData &data = multiCGData.cgData[i];
            // transfer data to other device
            for (int j = 1; j < multiCGData.numDevice; j++)
            {
                auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
                checkCudaErrors(cudaMemcpyAsync(nextData.p + data.global_begin_idx, data.p + data.global_begin_idx,
                                                data.localNumRow * sizeof(float), cudaMemcpyDefault, multiCGData.stream[data.deviceId]));
            }
        }
        multiCGData.syncDevice();

        // gpu spmv

        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto &data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            cusparseSpMV(
                data.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, data.alpha, data.matA, data.vecp,
                data.beta, data.vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, data.cuSparseWorkBuffer);            
        }
        

        float dot = multiGpuDotProductPAxCuSparse(multiCGData);
        a = r1 / dot;
        na = -a;

        for(int i = 0; i < multiCGData.numDevice; i++)
        {
            auto& data = multiCGData.cgData[i];
            checkCudaErrors(cudaSetDevice(data.deviceId));
            checkCudaErrors(cudaMemcpyAsync(data.a, &a, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
            checkCudaErrors(cudaMemcpyAsync(data.na, &na, sizeof(float), cudaMemcpyDefault, multiCGData.stream[i]));
            cublasSaxpy(data.cublasHandle, data.localNumRow, data.a, data.p + data.global_begin_idx, 1, data.x + data.global_begin_idx, 1);
            cublasSaxpy(data.cublasHandle, data.localNumRow, data.na, data.Ax + data.global_begin_idx, 1, data.r + data.global_begin_idx, 1);
        }

        r0 = r1;

        r1 = multiGpuDotProductRRCuSparse(multiCGData);

        // printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "cg time: " << delta.count() / 1000.f << std::endl;
    std::cout << "iteration: " << k << std::endl;
    return r1;
}

void test(MultiCGData& multiCGData)
{
    // cusparseLoggerSetLevel(4);
    float alpha = 1.0;
    float alpham1 = -1.0;
    float beta = 0.0;
    float r0 = 0.0;
    float r1 = 0.0;
    int max_iter = 1000;

    multiCGData.createSpMVBuffer(&alpha, &beta);
    multiCGData.spMV(&alpha, &beta);

}

// Map of device version to device number
std::multimap<std::pair<int, int>, int> getIdenticalGPUs()
{
    int numGpus = 0;
    checkCudaErrors(cudaGetDeviceCount(&numGpus));

    std::multimap<std::pair<int, int>, int> identicalGpus;

    for (int i = 0; i < numGpus; i++)
    {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));

        // Filter unsupported devices
        if (deviceProp.cooperativeLaunch)
        {
            identicalGpus.emplace(std::make_pair(deviceProp.major, deviceProp.minor),
                                  i);
        }
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", i,
               deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    return identicalGpus;
}

void initDevice(const int kNumGpusRequired)
{
    auto gpusByArch = getIdenticalGPUs();

    auto it = gpusByArch.begin();
    auto end = gpusByArch.end();

    auto bestFit = std::make_pair(it, it);
    // use std::distance to find the largest number of GPUs amongst architectures
    auto distance = [](decltype(bestFit) p)
    {
        return std::distance(p.first, p.second);
    };

    // Read each unique key/pair element in order
    for (; it != end; it = gpusByArch.upper_bound(it->first))
    {
        // first and second are iterators bounded within the architecture group
        auto testFit = gpusByArch.equal_range(it->first);
        // Always use devices with highest architecture version or whichever has the
        // most devices available
        if (distance(bestFit) <= distance(testFit))
            bestFit = testFit;
    }

    if (distance(bestFit) < kNumGpusRequired)
    {
        printf(
            "No two or more GPUs with same architecture capable of "
            "concurrentManagedAccess found. "
            "\nWaiving the sample\n");
        exit(EXIT_WAIVED);
    }

    std::set<int> bestFitDeviceIds;

    // Check & select peer-to-peer access capable GPU devices as enabling p2p
    // access between participating GPUs gives better performance.
    for (auto itr = bestFit.first; itr != bestFit.second; itr++)
    {
        int deviceId = itr->second;
        checkCudaErrors(cudaSetDevice(deviceId));

        std::for_each(
            itr, bestFit.second,
            [&deviceId, &bestFitDeviceIds,
             &kNumGpusRequired](decltype(*itr) mapPair)
            {
                if (deviceId != mapPair.second)
                {
                    int access = 0;
                    checkCudaErrors(
                        cudaDeviceCanAccessPeer(&access, deviceId, mapPair.second));
                    printf("Device=%d %s Access Peer Device=%d\n", deviceId,
                           access ? "CAN" : "CANNOT", mapPair.second);
                    if (access && bestFitDeviceIds.size() < kNumGpusRequired)
                    {
                        bestFitDeviceIds.emplace(deviceId);
                        bestFitDeviceIds.emplace(mapPair.second);
                    }
                    else
                    {
                        printf("Ignoring device %i (max devices exceeded)\n",
                               mapPair.second);
                    }
                }
            });

        if (bestFitDeviceIds.size() >= kNumGpusRequired)
        {
            printf("Selected p2p capable devices - ");
            for (auto devicesItr = bestFitDeviceIds.begin();
                 devicesItr != bestFitDeviceIds.end(); devicesItr++)
            {
                printf("deviceId = %d  ", *devicesItr);
            }
            printf("\n");
            break;
        }
    }

    // if bestFitDeviceIds.size() == 0 it means the GPUs in system are not p2p
    // capable, hence we add it without p2p capability check.
    if (!bestFitDeviceIds.size())
    {
        printf("Devices involved are not p2p capable.. selecting %zu of them\n",
               kNumGpusRequired);
        std::for_each(bestFit.first, bestFit.second,
                      [&bestFitDeviceIds,
                       &kNumGpusRequired](decltype(*bestFit.first) mapPair)
                      {
                          if (bestFitDeviceIds.size() < kNumGpusRequired)
                          {
                              bestFitDeviceIds.emplace(mapPair.second);
                          }
                          else
                          {
                              printf("Ignoring device %i (max devices exceeded)\n",
                                     mapPair.second);
                          }
                          // Insert the sequence into the deviceIds set
                      });
    }
    else
    {
        // perform cudaDeviceEnablePeerAccess in both directions for all
        // participating devices.
        for (auto p1_itr = bestFitDeviceIds.begin();
             p1_itr != bestFitDeviceIds.end(); p1_itr++)
        {
            checkCudaErrors(cudaSetDevice(*p1_itr));
            for (auto p2_itr = bestFitDeviceIds.begin();
                 p2_itr != bestFitDeviceIds.end(); p2_itr++)
            {
                if (*p1_itr != *p2_itr)
                {
                    checkCudaErrors(cudaDeviceEnablePeerAccess(*p2_itr, 0));
                    checkCudaErrors(cudaSetDevice(*p1_itr));
                }
            }
        }
    }
}

int main(int, char **)
{
    constexpr size_t kNumGpusRequired = 1;
    initDevice(kNumGpusRequired);

    int N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-5f;
    float *x;
    float rhs = 1.0;
    float r1;
    float *r, *p, *Ax;

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    N = 1000000;
    nz = (N - 2) * 3 + 4;

    I = new int[N + 1];
    J = new int[nz];
    val = new float[nz];

    genTridiag(I, J, val, N, nz);

    ////////////////////////////////////////////////////////

    std::vector<float> invec(N, 1.0);
    std::vector<float> out(N);
    std::vector<double> dotResult(kNumGpusRequired);

    MultiCGData multiCGData(kNumGpusRequired, N, nz);
    multiCGData.createCudaStream();

    for (int i = 0; i < multiCGData.numDevice; i++)
    {
        cudaSetDevice(multiCGData.cgData[i].deviceId);
        checkCudaErrors(cudaMemcpy(multiCGData.cgData[i].spMat.I, I, sizeof(int) * (N + 1), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(multiCGData.cgData[i].spMat.J, J, sizeof(int) * nz, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(multiCGData.cgData[i].spMat.val, val, sizeof(float) * nz, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(multiCGData.cgData[i].x, invec.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    }

#if 1
    float *Ax_cpu = (float *)malloc(sizeof(float) * N);
    float *r_cpu = (float *)malloc(sizeof(float) * N);
    float *p_cpu = (float *)malloc(sizeof(float) * N);
    float *x_cpu = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        r_cpu[i] = 1.0;
        Ax_cpu[i] = x_cpu[i] = 0.0;
    }
#endif

    // initGpuData(multiCGData);

    // multiCGData.createSparseMatVec();

    // test(multiCGData);

    auto residual = ConjugateGradCuSparse2(multiCGData, tol);
    std::cout << "residual: " << residual << std::endl;
    // auto residual = ConjugateGradCuSparse(multiCGData, tol);

    // std::vector<float> x_gpu(multiCGData.numRow);
    // for (int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     auto &data = multiCGData.cgData[i];
    //     checkCudaErrors(cudaMemcpy(x_gpu.data() + data.global_begin_idx, data.x + data.global_begin_idx, data.localNumRow * sizeof(float), cudaMemcpyDefault));
    // }
    
    // float rsum, diff, err = 0.0;

    // for (int i = 0; i < N; i++) {
    //     rsum = 0.0;

    //     for (int j = I[i]; j < I[i + 1]; j++) {
    //         rsum += val[j] * x_gpu[J[j]];
    //     }

    //     diff = fabs(rsum - rhs);

    //     if (diff > err) {
    //         err = diff;
    //     }
    // }

    // std::cout << "residual: " << residual << std::endl;
    // std::cout << "err: " << err << std::endl;

    // std::vector<float> x_tmp(multiCGData.numRow);
    // memcpy(x_tmp.data(), x_cpu, N * sizeof(float));
    // for (int i = 0; i < N; i++) {
    //     rsum = 0.0;

    //     for (int j = I[i]; j < I[i + 1]; j++) {
    //         rsum += val[j] * x_gpu[J[j]];
    //     }

    //     diff = fabs(rsum - rhs);

    //     if (diff > err) {
    //         err = diff;
    //     }
    // }

    // cpuSpMV(I, J, val, nz, N, 1, invec.data(), out.data());
    // saxpy(invec.data(), out.data(), 1, out.size());
    // float dotRes = dotProduct(invec.data(), out.data(), out.size());

    // cudaStream_t nStreams[kNumGpusRequired];
    // for (int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     cudaSetDevice(i);
    //     checkCudaErrors(cudaStreamCreate(&nStreams[i]));
    // }

    // dim3 dimGrid(32, 1, 1);
    // dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    // printf("Launching kernel\n");

    // for (int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     CGData &data = multiCGData.cgData[i];

    //     checkCudaErrors(cudaSetDevice(data.deviceId));

    //     void *kernelArgs[] = {
    //         (void *)&data.spMat.I,
    //         (void *)&data.spMat.J,
    //         (void *)&data.spMat.val,
    //         (void *)&data.spMat.nz,
    //         (void *)&data.localNumRow,
    //         (void *)&data.global_begin_idx,
    //         (void *)&data.x,
    //         (void *)&data.Ax};

    //     std::cout << "device " << data.deviceId << std::endl;

    //     checkCudaErrors(cudaLaunchCooperativeKernel(
    //         (void *)gpuSpMV, dimGrid, dimBlock, kernelArgs,
    //         0, nStreams[data.deviceId]));
    //     getLastCudaError("gpuSpMV execution failed");

    //     float alpha = 1.0;
    //     void *sapyArgs[] = {
    //         &data.x,
    //         &data.Ax,
    //         &alpha,
    //         &data.localNumRow,
    //         &data.global_begin_idx};

    //     checkCudaErrors(cudaLaunchCooperativeKernel(
    //         (void *)gpuSaxpy, dimGrid, dimBlock, sapyArgs,
    //         0, nStreams[data.deviceId]));
    //     getLastCudaError("gpuSaxpy execution failed");

    //     void *dotProductArgs[] = {
    //         &data.r,
    //         &data.Ax,
    //         &data.x,
    //         &data.localNumRow,
    //         &data.global_begin_idx};

    //     int sMemSize = sizeof(double) * THREADS_PER_BLOCK;
    //     checkCudaErrors(cudaLaunchCooperativeKernel(
    //         (void *)gpuDotProduct, dimGrid, dimBlock, dotProductArgs,
    //         0, nStreams[data.deviceId]));
    //     getLastCudaError("gpuDotProduct execution failed");

    //     // transfer data to other device
    //     for (int j = 1; j < multiCGData.numDevice; j++)
    //     {
    //         auto &nextData = multiCGData.cgData[(i + j) % multiCGData.numDevice];
    //         checkCudaErrors(cudaMemcpyAsync(nextData.Ax + data.global_begin_idx, data.Ax + data.global_begin_idx,
    //                                         data.localNumRow * sizeof(float), cudaMemcpyDefault, nStreams[data.deviceId]));
    //     }

    //     checkCudaErrors(cudaMemcpyAsync(dotResult.data() + data.deviceId, data.r, sizeof(double), cudaMemcpyDefault, nStreams[data.deviceId]));
    // }

    // std::cout << "I" << std::endl;
    // for (int i = 0; i < N + 1; i++)
    // {
    //     std::cout << I[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "j" << std::endl;
    // for (int i = 0; i < nz; i++)
    // {
    //     std::cout << J[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "val" << std::endl;
    // for (int i = 0; i < nz; i++)
    // {
    //     std::cout << val[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "MV output" << std::endl;
    // for (int i = 0; i < N; i++)
    // {
    //     std::cout << out[i] << " ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < kNumGpusRequired; i++)
    // {
    //     checkCudaErrors(cudaStreamSynchronize(nStreams[i]));
    // }

    // std::vector<float> Ax_cpu_(N);
    // for (int i = 0; i < multiCGData.numDevice; i++)
    // {
    //     auto &data = multiCGData.cgData[i];
    //     checkCudaErrors(cudaMemcpy(Ax_cpu_.data(), data.Ax, Ax_cpu_.size() * sizeof(float), cudaMemcpyDefault));

    //     std::cout << "device " << i << std::endl;
    //     for (int idx = 0; idx < Ax_cpu_.size(); idx++)
    //     {
    //         std::cout << Ax_cpu_[idx] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "------------dot result--------------" << std::endl;
    // std::cout << "cpu dotproduct: " << dotRes << std::endl;

    // double gpuDotSum = 0.0;
    // for (int i = 0; i < dotResult.size(); i++)
    // {
    //     gpuDotSum += dotResult[i];
    //     std::cout << dotResult[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "gpu dotproduct: " << gpuDotSum << std::endl;
    return 0;
}
