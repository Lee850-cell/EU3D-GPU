#include <cuda_runtime.h>
#include <cstddef>
#include "TimeAdvCuda.cuh"

__device__ inline size_t index4(
    int i, int j, int k, int s,
    int ny, int nz, int nvar)
{
    return ((static_cast<size_t>(i) * ny + j) * nz + k)
            * nvar + s; 
}

__global__ void ee_kernel(
    const double *xnode,
    const double *ynode,
    const double *znode,
    const double *F,
    const double *G,
    const double *Q,
    double *RHS,
    int nx, int ny, int nz,
    int nvar, int bc,
    double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + bc;
    int j = blockIdx.y * blockDim.y + threadIdx.y + bc;
    int k = blockIdx.z * blockDim.z + threadIdx.z + bc;

    if (i >= nx - bc ||
        j >= ny - bc ||
        k >= nz - bc) {
        return;
    }

    double dx = xnode[i] - xnode[i - 1];
    double dy = ynode[j] - ynode[j - 1];
    double dz = znode[k] - znode[k - 1];

    for (int s = 0; s < nvar; ++s) {
        size_t idx = index4(i, j, k, s, ny, nz, nvar);

        size_t idx_x = index4(i - 1, j, k, s, ny, nz, nvar);
        size_t idx_y = index4(i, j - 1, k, s, ny, nz, nvar);
        size_t idx_z = index4(i, j, k - 1, s, ny, nz, nvar);

        RHS[idx] =
            -dt / dx * (F[idx] - F[idx_x])
            -dt / dy * (G[idx] - G[idx_y])
            -dt / dz * (Q[idx] - Q[idx_z]);
    }

}

cudaError_t launch_ee_cuda(
    const double *xnode,
    const double *ynode,
    const double *znode,
    const double *F,
    const double *G,
    const double *Q,
    double *RHS,
    int nx, int ny, int nz,
    int nvar, int bc,
    double dt)
{
    dim3 threads(8, 4, 4);

    dim3 blocks(
        (nx - 2 * bc + threads.x - 1) / threads.x,
        (ny - 2 * bc + threads.y - 1) / threads.y,
        (nz - 2 * bc + threads.z - 1) / threads.z
    );

    ee_kernel<<<blocks, threads>>>(
        xnode, ynode, znode,
        F, G, Q, RHS,
        nx, ny, nz,
        nvar, bc, dt
    );

    return cudaGetLastError();
}