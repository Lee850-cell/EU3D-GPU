#include "FlowfieldCuda.cuh"
#include "ReactionCuda.cuh"
#include "ResidentMusclState.hpp"
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

__device__ __forceinline__
int index3(int i, int j, int k, int ny, int nz)
{
    return (i * ny + j) * nz + k;
}

__device__ __forceinline__
int index4(int i, int j, int k, int s,
           int ny, int nz, int ns)
{
    return ((i * ny + j) * nz + k) * ns + s;
}

__device__ __forceinline__
double minmod_cuda(double r)
{
    if (r > 0.0)
        return r < 1.0 ? r : 1.0;
    
    return 0.0;
}
__device__ __forceinline__
double sound_speed_cuda(double gamma, double pressure, double density)
{
    return sqrt(gamma * pressure / density); // sound speed
}

__global__ void muscl2_kernel(
    const double* fi,
    double* flr,
    int nx,
    int ny,
    int nz,
    int bc,
    int direction)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + bc - 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + bc - 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + bc - 1;

    if (i >= nx - bc ||
        j >= ny - bc ||
        k >= nz - bc)
    {
        return;
    }

    double fm1;
    double f0;
    double fp1;
    double fp2;

    if (direction == 1)
    {
        fm1 = fi[index3(i - 1, j, k, ny, nz)];
        f0  = fi[index3(i,     j, k, ny, nz)];
        fp1 = fi[index3(i + 1, j, k, ny, nz)];
        fp2 = fi[index3(i + 2, j, k, ny, nz)];
    }
    else if (direction == 2)
    {
        fm1 = fi[index3(i, j - 1, k, ny, nz)];
        f0  = fi[index3(i, j,     k, ny, nz)];
        fp1 = fi[index3(i, j + 1, k, ny, nz)];
        fp2 = fi[index3(i, j + 2, k, ny, nz)];
    }
    else
    {
        fm1 = fi[index3(i, j, k - 1, ny, nz)];
        f0  = fi[index3(i, j, k,     ny, nz)];
        fp1 = fi[index3(i, j, k + 1, ny, nz)];
        fp2 = fi[index3(i, j, k + 2, ny, nz)];
    }

    double A0 = f0 - fm1;
    double A1 = fp1 - f0;
    double A2 = fp2 - fp1;

    double L = A1 / (A0 + 1e-12);
    double R = A1 / (A2 + 1e-12);

    double L1 = minmod_cuda(L);
    double L2 = minmod_cuda(1.0 / L);
    double R1 = minmod_cuda(R);
    double R2 = minmod_cuda(1.0 / R);

    const double kk = 1.0 / 3.0;

    double left = f0
        + 0.25 * ((1.0 - kk) * L1
                + (1.0 + kk) * L2 * L) * A0;

    double right = fp1
        - 0.25 * ((1.0 - kk) * R1
                + (1.0 + kk) * R2 * R) * A2;

    flr[index4(i, j, k, 0, ny, nz, 2)] = left;
    flr[index4(i, j, k, 1, ny, nz, 2)] = right;
}

void launch_muscl2_cuda(
    const double* fi,
    double* flr,
    int nx,
    int ny,
    int nz,
    int bc,
    int direction)
{
    dim3 threads(8, 4, 4);

    dim3 blocks(
        (nx - 2 * bc + 1 + threads.x - 1) / threads.x,
        (ny - 2 * bc + 1 + threads.y - 1) / threads.y,
        (nz - 2 * bc + 1 + threads.z - 1) / threads.z
    );

    muscl2_kernel<<<blocks, threads>>>(
        fi, flr, nx, ny, nz, bc, direction);
}

void muscl2_host(
    const double* h_fi,
    double* h_flr,
    int nx,
    int ny,
    int nz,
    int bc,
    int direction)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "MUSCL CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    size_t count = static_cast<size_t>(nx) * ny * nz;
    size_t input_bytes = count * sizeof(double);
    size_t output_bytes = count * 2 * sizeof(double);

    double* d_fi = nullptr;
    double* d_flr = nullptr;

    check(cudaMalloc(&d_fi, input_bytes));
    check(cudaMalloc(&d_flr, output_bytes));

    check(cudaMemcpy(
        d_fi, h_fi, input_bytes,
        cudaMemcpyHostToDevice));

    // 保留 kernel 不会更新的边界区域
    check(cudaMemcpy(
        d_flr, h_flr, output_bytes,
        cudaMemcpyHostToDevice));

    launch_muscl2_cuda(
        d_fi, d_flr, nx, ny, nz, bc, direction);

    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    check(cudaMemcpy(
        h_flr, d_flr, output_bytes,
        cudaMemcpyDeviceToHost));

    check(cudaFree(d_fi));
    check(cudaFree(d_flr));
}

__global__ void ausm_kernel(
    const double* plr,
    const double* dlr,
    const double* ulr,
    const double* vlr,
    const double* wlr,
    const double* hlr,
    const double* glr,
    const double* yl,
    const double* yr,
    double* flux,
    int nx, int ny, int nz,
    int ns,
    int bc,
    int direction)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i >= nx - bc ||
        j >= ny - bc ||
        k >= nz - bc)
    {
        return;
    }

    int left_idx  = index4(i, j, k, 0, ny, nz, 2);
    int right_idx = index4(i, j, k, 1, ny, nz, 2);

    double pL = plr[left_idx];
    double pR = plr[right_idx];

    double rhoL = dlr[left_idx];
    double rhoR = dlr[right_idx];

    double uL = ulr[left_idx];
    double uR = ulr[right_idx];

    double vL = vlr[left_idx];
    double vR = vlr[right_idx];

    double wL = wlr[left_idx];
    double wR = wlr[right_idx];

    double hL = hlr[left_idx];
    double hR = hlr[right_idx];

    double gammaL = glr[left_idx];
    double gammaR = glr[right_idx];

    double cL = sound_speed_cuda(gammaL, pL, rhoL);
    double cR = sound_speed_cuda(gammaR, pR, rhoR);

    double cI = 0.5 * (cL + cR);

    double velocityL;
    double velocityR;

    if (direction == 1)
    {
        velocityL = uL;
        velocityR = uR;
    }
    else if (direction == 2)
    {
        velocityL = vL;
        velocityR = vR;
    }
    else
    {
        velocityL = wL;
        velocityR = wR;
    }

    double machL = velocityL / cI;
    double machR = velocityR / cI;

    double B1;
    double G1;

    if (fabs(machL) < 1.0)
    {
        double plus = machL + 1.0;
        double square_minus_one = machL * machL - 1.0;

        B1 = 0.25 * plus * plus
        + 0.125 * square_minus_one * square_minus_one;

        G1 = 0.25 * plus * plus * (2.0 - machL)
        + (3.0 / 16.0) * machL
        * square_minus_one * square_minus_one;
    }
    else
    {
        B1 = 0.5 * (machL + fabs(machL));
        G1 = B1 / machL;
    }

    double B2;
    double G2;

    if (fabs(machR) < 1.0)
    {
        double minus = machR - 1.0;
        double square_minus_one = machR * machR - 1.0;

        B2 = -0.25 * minus * minus
            -0.125 * square_minus_one * square_minus_one;

        G2 = 0.25 * minus * minus * (2.0 + machR)
            -(3.0 / 16.0) * machR
            * square_minus_one * square_minus_one;
    }
    else
    {
        B2 = 0.5 * (machR - fabs(machR));
        G2 = B2 / machR;
    }

    double machI = B1 + B2;
    double velocityI = machI * cI;

    // 根据流动方向选择上游一侧
    bool from_left = velocityI >= 0.0;

    double rho = from_left ? rhoL : rhoR;
    const double* species = from_left ? yl : yr;

    // 总质量通量
    double mass_flux = velocityI * rho;

    // 各组分质量通量
    for (int s = 0; s < ns; ++s)
    {
        int species_idx = index4(i, j, k, s, ny, nz, ns);
        int flux_idx = index4(i, j, k, s, ny, nz, ns + 4);

        flux[flux_idx] = mass_flux * species[species_idx];
    }

    // 选择上游一侧的速度和焓
    double u = from_left ? uL : uR;
    double v = from_left ? vL : vR;
    double w = from_left ? wL : wR;
    double h = from_left ? hL : hR;

    // 面上的压力项
    double pressureI = G1 * pL + G2 * pR;

    // 三个方向的动量通量
    flux[index4(i, j, k, ns, ny, nz, ns + 4)] =
        mass_flux * u + (direction == 1 ? pressureI : 0.0);

    flux[index4(i, j, k, ns + 1, ny, nz, ns + 4)] =
        mass_flux * v + (direction == 2 ? pressureI : 0.0);

    flux[index4(i, j, k, ns + 2, ny, nz, ns + 4)] =
        mass_flux * w + (direction == 3 ? pressureI : 0.0);

    // 能量通量：按原 CPU 公式计算
    flux[index4(i, j, k, ns + 3, ny, nz, ns + 4)] =
        velocityI * (rho * h
            + 0.5 * rho * (u * u + v * v + w * w));
}

void launch_ausm_cuda(
    const double* plr,
    const double* dlr,
    const double* ulr,
    const double* vlr,
    const double* wlr,
    const double* hlr,
    const double* glr,
    const double* yl,
    const double* yr,
    double* flux,
    int nx, int ny, int nz,
    int ns,
    int bc,
    int direction)
{
    dim3 threads(8, 4, 4);

    dim3 blocks(
        (nx - bc - 1 + threads.x - 1) / threads.x,
        (ny - bc - 1 + threads.y - 1) / threads.y,
        (nz - bc - 1 + threads.z - 1) / threads.z
    );

    ausm_kernel<<<blocks, threads>>>(
        plr, dlr, ulr, vlr, wlr, hlr, glr,
        yl, yr, flux,
        nx, ny, nz, ns, bc, direction);
}

void ausm_host(
    const double* h_plr,
    const double* h_dlr,
    const double* h_ulr,
    const double* h_vlr,
    const double* h_wlr,
    const double* h_hlr,
    const double* h_glr,
    const double* h_yl,
    const double* h_yr,
    double* h_flux,
    int nx, int ny, int nz,
    int ns,
    int bc,
    int direction)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "AUSM CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    size_t count = static_cast<size_t>(nx) * ny * nz;

    size_t lr_bytes = count * 2 * sizeof(double);
    size_t species_bytes = count * ns * sizeof(double);
    size_t flux_bytes = count * (ns + 4) * sizeof(double);

    // 七个左右状态数组，顺序与 kernel 参数一致
    const double* host_lr[7] = {
        h_plr, h_dlr, h_ulr, h_vlr,
        h_wlr, h_hlr, h_glr
    };

    double* device_lr[7] = {};

    for (int n = 0; n < 7; ++n)
    {
        check(cudaMalloc(&device_lr[n], lr_bytes));

        check(cudaMemcpy(
            device_lr[n], host_lr[n], lr_bytes,
            cudaMemcpyHostToDevice));
    }

    // 左右两侧的组分数组
    double* d_yl = nullptr;
    double* d_yr = nullptr;

    check(cudaMalloc(&d_yl, species_bytes));
    check(cudaMalloc(&d_yr, species_bytes));

    check(cudaMemcpy(
        d_yl, h_yl, species_bytes,
        cudaMemcpyHostToDevice));

    check(cudaMemcpy(
        d_yr, h_yr, species_bytes,
        cudaMemcpyHostToDevice));

    // 通量输出数组
    double* d_flux = nullptr;
    check(cudaMalloc(&d_flux, flux_bytes));

    // 保留 kernel 不更新的边界位置
    check(cudaMemcpy(
        d_flux, h_flux, flux_bytes,
        cudaMemcpyHostToDevice));
    
    // 启动 GPU 通量计算
    launch_ausm_cuda(
        device_lr[0], device_lr[1],
        device_lr[2], device_lr[3], device_lr[4],
        device_lr[5], device_lr[6],
        d_yl, d_yr, d_flux,
        nx, ny, nz, ns, bc, direction);

    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    // 将通量结果复制回 CPU
    check(cudaMemcpy(
        h_flux, d_flux, flux_bytes,
        cudaMemcpyDeviceToHost));

    // 释放本次分配的显存
    for (int n = 0; n < 7; ++n)
    {
        check(cudaFree(device_lr[n]));
    }

    check(cudaFree(d_yl));
    check(cudaFree(d_yr));
    check(cudaFree(d_flux));
}

__global__ void extract_species_kernel(
    const double* yi,
    double* scalar,
    size_t count,
    int ns,
    int s)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (idx >= count)
        return;

    scalar[idx] = yi[idx * ns + s];
}

__global__ void store_species_kernel(
    const double* species_lr,
    double* yl,
    double* yr,
    size_t count,
    int ns,
    int s)
{
    size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (idx >= count)
        return;

    yl[idx * ns + s] = species_lr[idx * 2];
    yr[idx * ns + s] = species_lr[idx * 2 + 1];
}

struct MusclAusmBuffers
{
    double* cached_fields = nullptr;
    double* front_host = nullptr;
    double* front_device = nullptr;
    size_t front_capacity = 0;
    bool upload_scope = false;
    bool uploaded = false;
    const double* host_identity[8] = {};
    double* input = nullptr;
    double* lr[7] = {};

    double* yi = nullptr;
    double* species_lr = nullptr;
    double* yl = nullptr;
    double* yr = nullptr;
    double* flux[3] = {};

    int nx = 0;
    int ny = 0;
    int nz = 0;
    int ns = 0;

    MusclAusmBuffers() = default;

    // 禁止复制，避免两个对象释放同一块显存
    MusclAusmBuffers(const MusclAusmBuffers&) = delete;
    MusclAusmBuffers& operator=(const MusclAusmBuffers&) = delete;

    void release()
    {
        cudaFreeHost(front_host); front_host = nullptr;
        cudaFree(front_device); front_device = nullptr; front_capacity = 0;
        if (cached_fields) cudaFree(cached_fields);
        cached_fields = nullptr;
        uploaded = false;
        if (input) cudaFree(input);

        for (int n = 0; n < 7; ++n)
        {
            if (lr[n]) cudaFree(lr[n]);
            lr[n] = nullptr;
        }

        if (yi) cudaFree(yi);
        if (species_lr) cudaFree(species_lr);
        if (yl) cudaFree(yl);
        if (yr) cudaFree(yr);
        for (int d = 0; d < 3; ++d)
        {
            if (flux[d]) cudaFree(flux[d]);
            flux[d] = nullptr;
        }

        input = nullptr;
        yi = nullptr;
        species_lr = nullptr;
        yl = nullptr;
        yr = nullptr;

        nx = ny = nz = ns = 0;
    }

    ~MusclAusmBuffers()
    {
        release();
    }
};

static MusclAusmBuffers& get_muscl_ausm_buffers()
{
    static MusclAusmBuffers buffers;
    return buffers;
}


namespace {
__global__ void resident_muscl_interior(const ReactionFlowState* states, double* fields,
    double* yi, int nx, int ny, int nz, int bc, int ns)
{
    const size_t q = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    const int ni=nx-2*bc, nj=ny-2*bc, nk=nz-2*bc;
    if (q >= static_cast<size_t>(ni)*nj*nk) return;
    const size_t grid=((q/(nj*nk)+bc)*ny+(q/nk)%nj+bc)*nz+q%nk+bc;
    const size_t full=static_cast<size_t>(nx)*ny*nz;
    const auto& st=states[q];
    fields[grid]=st.pressure;
    fields[full+grid]=st.density;
    fields[2*full+grid]=st.u;
    fields[3*full+grid]=st.v;
    fields[4*full+grid]=st.w;
    fields[5*full+grid]=st.enthalpy;
    fields[6*full+grid]=st.gamma;
    for(int n=0;n<ns;++n) yi[grid*ns+n]=st.mass_fractions[n];
}

__device__ void copy_muscl_boundary(double* fields, double* yi, size_t dst, size_t src,
                                  size_t full, int ns, bool reflect_w)
{
    for(int n=0;n<7;++n) {
        const double value=fields[n*full+src];
        fields[n*full+dst]=(reflect_w && n==4)?-value:value;
    }
    for(int n=0;n<ns;++n) yi[dst*ns+n]=yi[src*ns+n];
}

// Match the CPU RSBI ordering exactly: x faces, y faces, then z faces.
// y-front thermodynamics are still computed by the CPU and uploaded separately.
__global__ void resident_muscl_x(double* fields,double* yi,int nx,int ny,int nz,int bc,int ns)
{
    const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const int nj=ny-2*bc,nk=nz-2*bc;
    if(q>=static_cast<size_t>(2*bc)*nj*nk) return;
    const int layer=q/(nj*nk),j=(q/nk)%nj+bc,k=q%nk+bc;
    const int i=layer<bc?layer:nx-2*bc+layer;
    const int source=layer<bc?2*bc-1-i:2*(nx-bc)-1-i;
    copy_muscl_boundary(fields,yi,(static_cast<size_t>(i)*ny+j)*nz+k,
        (static_cast<size_t>(source)*ny+j)*nz+k,static_cast<size_t>(nx)*ny*nz,ns,false);
}

__global__ void resident_muscl_y(double* fields,double* yi,const double* front,
                               int nx,int ny,int nz,int bc,int ns)
{
    const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const int nk=nz-2*bc;
    if(q>=static_cast<size_t>(nx)*bc*nk) return;
    const int i=q/(bc*nk),j=(q/nk)%bc,k=q%nk+bc;
    const size_t full=static_cast<size_t>(nx)*ny*nz;
    const size_t dst=(static_cast<size_t>(i)*ny+j)*nz+k;
    for(int n=0;n<7;++n) fields[n*full+dst]=front[q*(7+ns)+n];
    for(int n=0;n<ns;++n) yi[dst*ns+n]=front[q*(7+ns)+7+n];
    const int back=ny-bc+j,source=2*(ny-bc)-1-back;
    copy_muscl_boundary(fields,yi,(static_cast<size_t>(i)*ny+back)*nz+k,
        (static_cast<size_t>(i)*ny+source)*nz+k,full,ns,false);
}

__global__ void resident_muscl_z(double* fields,double* yi,int nx,int ny,int nz,int bc,int ns)
{
    const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(q>=static_cast<size_t>(nx)*ny*2*bc) return;
    const int i=q/(ny*2*bc),j=(q/(2*bc))%ny,layer=q%(2*bc);
    const int k=layer<bc?layer:nz-2*bc+layer;
    const int source=layer<bc?2*bc-1-k:2*(nz-bc)-1-k;
    copy_muscl_boundary(fields,yi,(static_cast<size_t>(i)*ny+j)*nz+k,
        (static_cast<size_t>(i)*ny+j)*nz+source,static_cast<size_t>(nx)*ny*nz,ns,true);
}

size_t build_resident_muscl(MusclAusmBuffers& b,const double* const* host,
                            const double* host_yi,int nx,int ny,int nz,int bc,int ns)
{
    auto check=[](cudaError_t e){if(e!=cudaSuccess){
        std::fprintf(stderr,"Resident MUSCL CUDA error: %s\n",cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);}};
    const auto& view=resident_muscl_state;
    const int ni=nx-2*bc,nj=ny-2*bc,nk=nz-2*bc;
    if(!view.states || view.ni!=ni || view.nj!=nj || view.nk!=nk
        || view.bc!=bc || view.ns!=ns || ni<bc || nj<bc || nk<bc){
        std::fprintf(stderr,"Resident MUSCL error: invalid state shape\n");std::exit(EXIT_FAILURE);
    }
    const size_t front_count=static_cast<size_t>(nx)*bc*nk;
    const size_t elements=front_count*(7+ns);
    if(b.front_capacity<elements){
        check(cudaFreeHost(b.front_host));b.front_host=nullptr;
        check(cudaFree(b.front_device));b.front_device=nullptr;b.front_capacity=0;
        check(cudaMallocHost(&b.front_host,elements*sizeof(double)));
        check(cudaMalloc(&b.front_device,elements*sizeof(double)));
        b.front_capacity=elements;
    }
    size_t q=0;
    for(int i=0;i<nx;++i) for(int j=0;j<bc;++j) for(int k=bc;k<nz-bc;++k,++q){
        const size_t grid=(static_cast<size_t>(i)*ny+j)*nz+k;
        for(int n=0;n<7;++n) b.front_host[q*(7+ns)+n]=host[n][grid];
        std::memcpy(b.front_host+q*(7+ns)+7,host_yi+grid*ns,ns*sizeof(double));
    }
    check(cudaMemcpy(b.front_device,b.front_host,elements*sizeof(double),cudaMemcpyHostToDevice));
    const size_t interior=static_cast<size_t>(ni)*nj*nk;
    resident_muscl_interior<<<(interior+255)/256,256>>>(view.states,b.cached_fields,b.yi,nx,ny,nz,bc,ns);
    check(cudaGetLastError());
    const size_t x_count=static_cast<size_t>(2*bc)*nj*nk;
    resident_muscl_x<<<(x_count+255)/256,256>>>(b.cached_fields,b.yi,nx,ny,nz,bc,ns);
    check(cudaGetLastError());
    resident_muscl_y<<<(front_count+255)/256,256>>>(b.cached_fields,b.yi,b.front_device,nx,ny,nz,bc,ns);
    check(cudaGetLastError());
    const size_t z_count=static_cast<size_t>(nx)*ny*2*bc;
    resident_muscl_z<<<(z_count+255)/256,256>>>(b.cached_fields,b.yi,nx,ny,nz,bc,ns);
    check(cudaGetLastError());
    if(view.validate){
        const size_t full=static_cast<size_t>(nx)*ny*nz;
        std::vector<double> actual(full*ns);
        size_t differences=0;
        for(int n=0;n<8;++n){
            const size_t length=n==7?full*ns:full;
            const double* expected=n==7?host_yi:host[n];
            check(cudaMemcpy(actual.data(),n==7?b.yi:b.cached_fields+n*full,
                             length*sizeof(double),cudaMemcpyDeviceToHost));
            for(size_t k=0;k<length;++k) if(std::memcmp(&actual[k],&expected[k],sizeof(double))!=0){
                if(differences<8) std::fprintf(stderr,"Resident MUSCL mismatch: field=%d index=%zu host=%.17g device=%.17g\n",n,k,expected[k],actual[k]);
                ++differences;
            }
        }
        std::printf("Resident MUSCL validation: %s, differences=%zu, halos=checked, comparison=bitwise\n",differences?"Failed":"Passed",differences);
        if(differences) std::exit(EXIT_FAILURE);
    }
    return elements*sizeof(double);
}
}

void begin_muscl_upload_scope(bool enabled)
{
    auto& b = get_muscl_ausm_buffers();
    b.upload_scope = enabled;
    b.uploaded = false;
}
void end_muscl_upload_scope()
{
    auto& b = get_muscl_ausm_buffers();
    b.upload_scope = false;
    b.uploaded = false;
}

void muscl_ausm_host(
    const double* h_p,
    const double* h_density,
    const double* h_u,
    const double* h_v,
    const double* h_w,
    const double* h_h,
    const double* h_gamma,
    const double* h_yi,
    double* h_flux,
    int nx, int ny, int nz,
    int ns,
    int bc,
    int direction,
    bool copy_flux_to_host)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "MUSCL-AUSM CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    size_t count = static_cast<size_t>(nx) * ny * nz;
    size_t scalar_bytes = count * sizeof(double);
    size_t lr_bytes = count * 2 * sizeof(double);

    MusclAusmBuffers& buffers = get_muscl_ausm_buffers();

    bool size_changed =
        buffers.nx != nx ||
        buffers.ny != ny ||
        buffers.nz != nz ||
        buffers.ns != ns;

    if (size_changed)
    {
        buffers.release();

        check(cudaMalloc(&buffers.input, scalar_bytes));
    
        for (int n = 0; n < 7; ++n)
            check(cudaMalloc(&buffers.lr[n], lr_bytes));

        check(cudaMalloc(
            &buffers.yi, count * ns * sizeof(double)));

        check(cudaMalloc(&buffers.species_lr, lr_bytes));

        check(cudaMalloc(
            &buffers.yl, count * ns * sizeof(double)));

        check(cudaMalloc(
            &buffers.yr, count * ns * sizeof(double)));

        for (int d = 0; d < 3; ++d)
        {
            check(cudaMalloc(
                &buffers.flux[d],
                count * (ns + 4) * sizeof(double)));
        }

        buffers.nx = nx;
        buffers.ny = ny;
        buffers.nz = nz;
        buffers.ns = ns;
    }

    const double* host_fields[7] = {
        h_p, h_density, h_u, h_v, h_w, h_h, h_gamma
    };

    const bool reuse_upload = buffers.upload_scope;
    const bool upload_now = !reuse_upload || !buffers.uploaded;
    if (reuse_upload && !buffers.cached_fields)
        check(cudaMalloc(&buffers.cached_fields, 7 * scalar_bytes));
    if (reuse_upload && buffers.uploaded) {
        bool same = buffers.host_identity[7] == h_yi;
        for (int n=0;n<7;++n) same = same && buffers.host_identity[n] == host_fields[n];
        if (!same) {
            std::fprintf(stderr,"MUSCL upload scope error: input pointers changed inside sweep\n");
            std::exit(EXIT_FAILURE);
        }
    }

    double* d_input = buffers.input;
    double** d_lr = buffers.lr;

    double upload_ms = 0.0;
    size_t resident_bytes = 0;
    const bool resident = resident_muscl_state.enabled && resident_muscl_state.ready;
    if (resident && !reuse_upload) {
        std::fprintf(stderr,"Resident MUSCL error: upload scope required\n");
        std::exit(EXIT_FAILURE);
    }
    if (resident && upload_now) {
        const auto start=std::chrono::steady_clock::now();
        resident_bytes=build_resident_muscl(buffers,host_fields,h_yi,nx,ny,nz,bc,ns);
        upload_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::printf("Resident MUSCL: reused interior, boundary_upload_bytes=%zu\n",resident_bytes);
    }

    for (int n = 0; n < 7; ++n)
    {
        check(cudaMemset(d_lr[n], 0, lr_bytes));

        auto upload_start = std::chrono::steady_clock::now();

        double* field_input = reuse_upload ? buffers.cached_fields + count*n : d_input;
        if (upload_now && !resident) {
            check(cudaMemcpy(field_input, host_fields[n], scalar_bytes, cudaMemcpyHostToDevice));
        }

        auto upload_end = std::chrono::steady_clock::now();

        upload_ms +=
            std::chrono::duration<double, std::milli>(
                upload_end - upload_start).count();

        launch_muscl2_cuda(
            field_input, d_lr[n],
            nx, ny, nz, bc, direction);

        check(cudaGetLastError());
    }

    std::printf(
    "Field upload direction=%d: %.3f ms\n",
    direction, upload_ms);

    size_t species_bytes = count * ns * sizeof(double);

    double* d_yi = buffers.yi;
    double* d_species_lr = buffers.species_lr;

    if (upload_now && !resident) {
        check(cudaMemcpy(d_yi, h_yi, species_bytes, cudaMemcpyHostToDevice));
    }
    if (reuse_upload) {
        for (int n=0;n<7;++n) buffers.host_identity[n] = host_fields[n];
        buffers.host_identity[7] = h_yi;
        buffers.uploaded = true;
        std::printf("MUSCL upload reuse: direction=%d, host_to_device_bytes=%zu\n",
            direction, upload_now ? (resident ? resident_bytes : 7*scalar_bytes+species_bytes) : size_t(0));
    }

    check(cudaMemset(d_species_lr, 0, lr_bytes));

    int threads = 256;
    int blocks = static_cast<int>(
        (count + threads - 1) / threads);

    double* d_yl = buffers.yl;
    double* d_yr = buffers.yr;

    for (int s = 0; s < ns; ++s)
    {
        // 取出第 s 个组分，放入复用的输入数组
        extract_species_kernel<<<blocks, threads>>>(
            d_yi, d_input, count, ns, s);

        check(cudaGetLastError());

        // 对这个组分进行面左右重构
        launch_muscl2_cuda(
            d_input, d_species_lr,
            nx, ny, nz, bc, direction);

        check(cudaGetLastError());
        

        store_species_kernel<<<blocks, threads>>>(
        d_species_lr, d_yl, d_yr, count, ns, s);

        check(cudaGetLastError());
        
    }

    size_t flux_bytes = count * (ns + 4) * sizeof(double);

    double* d_flux = buffers.flux[direction - 1];
    // 保留 AUSM 不更新的边界位置
    if (copy_flux_to_host)
    {
        // 验证路径：保留 CPU 数组中未更新的位置
        check(cudaMemcpy(
            d_flux, h_flux, flux_bytes,
            cudaMemcpyHostToDevice));
    }
    else
    {
        // 当前性能路径：对应 Advection 中 F/G/Q 的清零
        check(cudaMemset(d_flux, 0, flux_bytes));
    }

    // 直接使用留在 GPU 上的重构结果
    launch_ausm_cuda(
        d_lr[0], d_lr[1],
        d_lr[2], d_lr[3], d_lr[4],
        d_lr[5], d_lr[6],
        d_yl, d_yr, d_flux,
        nx, ny, nz, ns, bc, direction);

    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    // 只把最终通量复制回 CPU
    if (copy_flux_to_host)
    {
        check(cudaMemcpy(
            h_flux, d_flux, flux_bytes,
            cudaMemcpyDeviceToHost));
    }
}

const double* get_ausm_device_flux(int direction)
{
    if (direction < 1 || direction > 3)
        return nullptr;

    return get_muscl_ausm_buffers().flux[direction - 1];
}
