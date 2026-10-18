/*=================================================================================
 * Class TimeAdv:	The time advance scheme:
 *					1st-order Euler Explicit Scheme
 *					3st-order TVD Runge-Kutta Scheme
 *=================================================================================*/
#include <cstddef>

#ifndef TIME
#define TIME

#include "Array.hpp"
using namespace ARRAY;



class TimeAdv
{
private:
    double *d_x = nullptr;
    double *d_y = nullptr;
    double *d_z = nullptr;

    double *d_F = nullptr;
    double *d_G = nullptr;
    double *d_Q = nullptr;
    double *d_rhs = nullptr;

    std::size_t device_bytes_ = 0;
    int device_nx_ = 0;
    int device_ny_ = 0;
    int device_nz_ = 0;

public:
	/* Constructor */
	TimeAdv() = default;

	/* 1st-order Euler Explicit Scheme */
	void EE(int, double, Array<double, 1>&, Array<double, 1>&,Array<double, 1>&, int, Array<double, 4>&, Array<double, 4>&,Array<double, 4>&, Array<double, 4> &);

	void EE_GPU(
    int NS, double dt,
    Array<double, 1> &xnode,
    Array<double, 1> &ynode,
    Array<double, 1> &znode,
    int bc,
    Array<double, 4> &F,
    Array<double, 4> &G,
    Array<double, 4> &Q,
    Array<double, 4> &RHS,
    const double* device_F = nullptr,
    const double* device_G = nullptr,
    const double* device_Q = nullptr);
    /* 3st-order TVD Runge-Kutta Scheme */
	void TVD_RK3(int, double, Array<double, 1>&, Array<double, 1>&,Array<double, 1>&, int, Array<double, 4>&, Array<double, 4>&,Array<double, 4>&, Array<double, 4> &);
	
	/* Destructor */
	~TimeAdv();
};

#endif
