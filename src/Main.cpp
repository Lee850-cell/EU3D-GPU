/*=================================================================================
 * This program is to solve the 2D multi-species Euler equation.
 *
 * For User:
 * Open the ctrl.txt:	Set the ctrl parameters
 * Open the input.txt:	Set the initial parameters and chemical reaction model
 * Open the grid.txt:	Set the grid parameters
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <iostream>
#include <mpi.h>
// #include <omp.h>
// #include <vector>
// #include <tuple>
// #include <random>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Euler.hpp"
#include "StopWatch.hpp"
#include "TimeBase.hpp"
#include "WallClock.hpp"
#include <cuda_runtime.h>
#include "ComputeConfig.hpp"
// #include <Array.hpp>
// #include "LoadBalance.hpp"
/*---------------------------------------------------------------------------------
 * Main function
 *---------------------------------------------------------------------------------*/
using namespace std;
int main()
{
	/* Record the running time */
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	cout << "Total process: " << numprocs << endl;

	double t1 = wall_seconds();

	if (use_cuda)
	{
		// 将同一台机器上的进程分到一个通信组。
		MPI_Comm local_comm;
		MPI_Comm_split_type(
			MPI_COMM_WORLD,
			MPI_COMM_TYPE_SHARED,
			myid,
			MPI_INFO_NULL,
			&local_comm);

		int local_rank, local_size;
		MPI_Comm_rank(local_comm, &local_rank);
		MPI_Comm_size(local_comm, &local_size);

		auto check_cuda = [&](cudaError_t err, const char* operation)
		{
			if (err != cudaSuccess)
			{
				std::cerr << "rank " << myid << ": "
						<< operation << ": "
						<< cudaGetErrorString(err) << std::endl;

				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		};

		int device_count = 0;
		check_cuda(cudaGetDeviceCount(&device_count), "Get GPU count");

		// 这一版要求同一台机器上的进程看见相同的完整 GPU 列表。
		if (device_count < local_size)
		{
			std::cerr << "rank " << myid
					<< ": local processes=" << local_size
					<< ", visible GPUs=" << device_count
					<< "; one GPU per process is required"
					<< std::endl;

			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		const int device = local_rank;
		check_cuda(cudaSetDevice(device), "Select GPU");

		cudaDeviceProp properties{};
		check_cuda(
			cudaGetDeviceProperties(&properties, device),
			"Get GPU properties");

		std::cout << "world rank=" << myid
				<< ", local rank=" << local_rank
				<< ", GPU=" << device
				<< ", name=" << properties.name
				<< std::endl;

		MPI_Comm_free(&local_comm);
	}
	/* Solve the equation */
	Euler value;
	double t2 = wall_seconds();
	if (myid == 0)
	{
		cout << "Total Time: " << t2 - t1 << endl;
	}

	MPI_Finalize();

	return 0;
}