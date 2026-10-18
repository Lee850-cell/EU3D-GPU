#pragma once

#include <cuda_runtime.h>

// All array pointers must refer to GPU memory.
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
    double dt);