#include "ReactionCuda.cuh"
/*=================================================================================
 * The functions of TimeAdv
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <iostream>

#include "TimeAdvCuda.cuh"
#include <cstdio>
#include <cstdlib>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "TimeAdv.hpp"

/*---------------------------------------------------------------------------------
	Name:		TimeAdv::EE

	Input(8):	The Number of species; The time step;
				The mesh nodes and boundary nodes;
				The Flux; The current residual value

	Function:	Use the 1st-order Euler Explicit Scheme

	Return:		The residual value after dt
*---------------------------------------------------------------------------------*/
static void checkCuda(cudaError_t err, const char *message)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n",
                     message, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void TimeAdv::EE(int NS, double dt, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, int bc, Array<double, 4> &F, Array<double, 4> &G, Array<double, 4> &Q, Array<double, 4> &RHS)
{

	int ni = xnode.GetSize() - 2 * bc;
	int nj = ynode.GetSize() - 2 * bc;
	int nk = znode.GetSize() - 2 * bc;

	for (int i = bc; i < ni + bc; i++)
		for (int j = bc; j < nj + bc; j++)
			for (int k = bc; k < nk + bc; k++)
				for (int s = 0; s < NS + 4; s++)
					RHS(i, j, k, s) = -dt / (xnode(i) - xnode(i - 1)) * (F(i, j, k, s) - F(i - 1, j, k, s)) -
									  dt / (ynode(j) - ynode(j - 1)) * (G(i, j, k, s) - G(i, j - 1, k, s)) -
									  dt / (znode(k) - znode(k - 1)) * (Q(i, j, k, s) - Q(i, j, k - 1, s));
}

void TimeAdv::EE_GPU(
    int NS, double dt,
    Array<double, 1> &xnode,
    Array<double, 1> &ynode,
    Array<double, 1> &znode,
    int bc,
    Array<double, 4> &F,
    Array<double, 4> &G,
    Array<double, 4> &Q,
    Array<double, 4> &RHS,
	const double* device_F,
	const double* device_G,
	const double* device_Q)
{
    int nx = static_cast<int>(xnode.GetSize());
    int ny = static_cast<int>(ynode.GetSize());
    int nz = static_cast<int>(znode.GetSize());
    int nvar = NS + 4;

    const std::size_t bytes = F.GetLen() * sizeof(double);


	bool size_changed =
		device_bytes_ != bytes ||
		device_nx_ != nx ||
		device_ny_ != ny ||
		device_nz_ != nz;

	if (size_changed) {
		if (d_x)   checkCuda(cudaFree(d_x),   "Resize: free x");
		if (d_y)   checkCuda(cudaFree(d_y),   "Resize: free y");
		if (d_z)   checkCuda(cudaFree(d_z),   "Resize: free z");
		if (d_F)   checkCuda(cudaFree(d_F),   "Resize: free F");
		if (d_G)   checkCuda(cudaFree(d_G),   "Resize: free G");
		if (d_Q)   checkCuda(cudaFree(d_Q),   "Resize: free Q");
		if (d_rhs) checkCuda(cudaFree(d_rhs), "Resize: free RHS");

		checkCuda(cudaMalloc((void **)&d_x,
							nx * sizeof(double)), "Allocate x");
		checkCuda(cudaMalloc((void **)&d_y,
							ny * sizeof(double)), "Allocate y");
		checkCuda(cudaMalloc((void **)&d_z,
							nz * sizeof(double)), "Allocate z");

		checkCuda(cudaMalloc((void **)&d_F, bytes), "Allocate F");
		checkCuda(cudaMalloc((void **)&d_G, bytes), "Allocate G");
		checkCuda(cudaMalloc((void **)&d_Q, bytes), "Allocate Q");
		checkCuda(cudaMalloc((void **)&d_rhs, bytes), "Allocate RHS");

		device_bytes_ = bytes;
		device_nx_ = nx;
		device_ny_ = ny;
		device_nz_ = nz;
	}
	checkCuda(cudaMemcpy(d_x, xnode.Getbuf(),
						nx * sizeof(double),
						cudaMemcpyHostToDevice),
			"Copy x");

	checkCuda(cudaMemcpy(d_y, ynode.Getbuf(),
						ny * sizeof(double),
						cudaMemcpyHostToDevice),
			"Copy y");

	checkCuda(cudaMemcpy(d_z, znode.Getbuf(),
						nz * sizeof(double),
						cudaMemcpyHostToDevice),
			"Copy z");

	if (device_F == nullptr)
	{
		checkCuda(cudaMemcpy(
			d_F, F.Getbuf(), bytes, cudaMemcpyHostToDevice),
			"Copy F");
	}

	if (device_G == nullptr)
	{
		checkCuda(cudaMemcpy(
			d_G, G.Getbuf(), bytes, cudaMemcpyHostToDevice),
			"Copy G");
	}

	if (device_Q == nullptr)
	{
		checkCuda(cudaMemcpy(
			d_Q, Q.Getbuf(), bytes, cudaMemcpyHostToDevice),
			"Copy Q");
	}

	const double* flux_F = device_F ? device_F : d_F;
	const double* flux_G = device_G ? device_G : d_G;
	const double* flux_Q = device_Q ? device_Q : d_Q;

	if (ee_state_chain_enabled()) {
        checkCuda(cudaMemset(d_rhs,0,bytes), "Clear device-only RHS");
    } else {
	checkCuda(cudaMemcpy(d_rhs, RHS.Getbuf(), bytes,
                     cudaMemcpyHostToDevice),
          "Copy initial RHS");
	}

	cudaEvent_t start, stop;
	checkCuda(cudaEventCreate(&start), "Create start event");
	checkCuda(cudaEventCreate(&stop), "Create stop event");

	checkCuda(cudaEventRecord(start), "Record kernel start");

	checkCuda(
		launch_ee_cuda(
			d_x, d_y, d_z,
			flux_F, flux_G, flux_Q, d_rhs,
			nx, ny, nz,
			nvar, bc, dt
		),
		"Launch EE"
	);

	checkCuda(cudaEventRecord(stop), "Record kernel stop");
	checkCuda(cudaEventSynchronize(stop), "Wait for kernel");

	float kernel_ms = 0.0f;
	checkCuda(cudaEventElapsedTime(&kernel_ms, start, stop),
			"Measure kernel");

	std::cout << "EE GPU kernel time: "
			<< kernel_ms << " ms" << std::endl;

	checkCuda(cudaEventDestroy(start), "Destroy start event");
	checkCuda(cudaEventDestroy(stop), "Destroy stop event");

    if(ee_state_chain_enabled()){
        publish_ee_state_rhs(d_rhs,nx,ny,nz,nvar,bc);
    } else {
	checkCuda(cudaMemcpy(RHS.Getbuf(), d_rhs, bytes,
						cudaMemcpyDeviceToHost),
			"Copy RHS to CPU");
    }

}
/*---------------------------------------------------------------------------------
	Name:		TimeAdv::TVD_RK3

	Input(8):	The Number of species; The time step;
				The mesh nodes and boundary nodes;
				The Flux; The current residual value

	Function:	Use the 3rd-order TVD Runge-Kutta Scheme

	Return:		The flow field value after dt
 *---------------------------------------------------------------------------------*/

void TimeAdv::TVD_RK3(int NS, double dt, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, int bc, Array<double, 4> &F, Array<double, 4> &G, Array<double, 4> &Q, Array<double, 4> &RHS)
{
	int ni = xnode.GetSize();
	int nj = ynode.GetSize();
	int nk = znode.GetSize() - 2 * bc;

	for (int i = bc; i < ni + bc; i++)
		for (int j = bc; j < nj + bc; j++)
			for (int k = bc; k < nk + bc; k++)
				for (int s = 0; s < NS + 4; s++)
					RHS(i, j, k, s) = RHS(i, j, k, s);
}
TimeAdv::~TimeAdv()
{
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_z) cudaFree(d_z);
    if (d_F) cudaFree(d_F);
    if (d_G) cudaFree(d_G);
    if (d_Q) cudaFree(d_Q);
    if (d_rhs) cudaFree(d_rhs);
}
