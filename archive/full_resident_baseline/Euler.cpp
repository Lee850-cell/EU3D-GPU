#include <iomanip>
#include <cstdint>
#include "PackedReactionCopy.hpp"
/*=================================================================================
 * The functions of Euler
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <numeric>
#include <filesystem>
#include <cmath>
#include <cstdio>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Euler.hpp"
#include "_string_switch.hpp"
#include "FileReader.hpp"

#include "ReactionCuda.cuh"
#include <cstdlib>
#include "WallClock.hpp"
#include "ComputeConfig.hpp"

namespace
{
constexpr bool validate_reaction_gpu = false;
}

using namespace std;

/*---------------------------------------------------------------------------------
	Name:		Euler::Euler

	Input(0):	None

	Function:	Read files which identify the physical problem;
				Initialization;
				Compute the flow field and output in real time;
				Output the final result

	Return:		None
 *---------------------------------------------------------------------------------*/

Euler::Euler()
{
	/* Print the result of pre-condition */
	cout << "Input ctrl file is:" << this->ctrlfile << endl;

	/* Read the ctrl file */
	double t1 = wall_seconds();
	FileRead();
	double t2 = wall_seconds();
	/* Mpi initialization: calculate neighbor process id*/
	Mpi_Initial();
	double t3 = wall_seconds();
	/* Read the mesh file and process the mesh */
	Mymesh.MeshProcess(meshFile);
	std::cout << "Partition: rank=" << myid
			<< ", coords=("
			<< myid_x << "," << myid_y << "," << myid_z << ")"
			<< ", local_grid=("
			<< Mymesh.ni << "," << Mymesh.nj << "," << Mymesh.nk << ")"
			<< ", halo=" << Mymesh.bc
			<< ", left=" << m_left
			<< ", right=" << m_right
			<< std::endl;
	double t4 = wall_seconds();
	/* Read the input file */
	TwoDim.InputRead(inputFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	ExtraFlow.InputRead(inputFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	double t5 = wall_seconds();
	/* Read the reaction model and thermo file */
	React.ReactionRead(Reaction_Model, thermoFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	ExtraReact.ReactionRead(Reaction_Model, thermoFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	double t6 = wall_seconds();
	/* Intitalize the flow field */
	TwoDim.FieldInitial(React.Ri, React.Mw, React.Mi, React.Yi, React.Coeff0, React.Coeff1);
	ExtraFlow.FieldInitial(React.Ri, React.Mw, React.Mi, React.Yi, React.Coeff0, React.Coeff1);
	double t7 = wall_seconds();
	MPI_Barrier(MPI_COMM_WORLD);
	double t8 = wall_seconds();
	/* Compute the flow field and output in real time */
	Computing(count);
	double t9 = wall_seconds();
	/* Output the final result in a file */
	Output(count);
	double t10 = wall_seconds();
	Output_Total_Time(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);
	return;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::FileRead

	Input(0):	None

	Function:	Read the ctrl file to get the global informartion

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::FileRead()
{
	FileReader Ctrl;
	Ctrl.readFile(ctrlfile);
 
	/* Read number */
	Final_Time = Ctrl.getdoubleParameter("Final-Time");
	cfl = Ctrl.getdoubleParameter("cfl");
	if (cfl == 0)
		dt = Ctrl.getdoubleParameter("dt");
	count = Ctrl.getIntParameter("count");

	/* Read string */
	strcpy(meshFile, Ctrl.getStringParameter("gridname").c_str());
	strcpy(inputFile, Ctrl.getStringParameter("initialization").c_str());
	strcpy(Reaction_Model, Ctrl.getStringParameter("reaction-model").c_str());
	strcpy(thermoFile, Ctrl.getStringParameter("Thermofile").c_str());

	/* Judge the reaction scheme */
	strcpy(react, Ctrl.getStringParameter("reaction").c_str());
	switch (hash_(react))
	{
	case "Trapezoid"_hash:
		Reaction_Sch = Global::Trapezoid;
		type = 0;
		break;
	case "Trapezoid-Adaptive"_hash:
		Reaction_Sch = Global::Trapezoid;
		type = 1;
		break;
	case "IMEX"_hash:
		Reaction_Sch = Global::IMEX;
		break;
	case "Cantera"_hash:
		Reaction_Sch = Global::DNN;
		type = 0;
		break;
	case "DNN"_hash:
		Reaction_Sch = Global::DNN;
		type = 1;
		break;
	default:
		cout << "Invalid reaction scheme" << endl;
		abort();
	}

	/* Judge the difference scheme */
	strcpy(diff, Ctrl.getStringParameter("difference").c_str());
	switch (hash_(diff))
	{
	case "MUSCL_1"_hash:
		Diff_Sch = Global::MUSCL_1;
		break;
	case "MUSCL_2"_hash:
		Diff_Sch = Global::MUSCL_2;
		break;
	case "WENO"_hash:
		Diff_Sch = Global::WENO;
		break;
	default:
		cout << "Invalid difference scheme" << endl;
	}

	/* Judge the time advance scheme */
	strcpy(timeadv, Ctrl.getStringParameter("time-adv").c_str());
	switch (hash_(timeadv))
	{
	case "EE"_hash:
		TimeAdv_Sch = Global::EE;
		break;
	case "TVD_RK3"_hash:
		TimeAdv_Sch = Global::TVD_RK3;
		break;
	default:
		cout << "Invalid time-advance scheme" << endl;
	}

	/* Read mpi block number */
	m_block_x = Ctrl.getIntParameter("m_block_x");
	m_block_y = Ctrl.getIntParameter("m_block_y");
	m_block_z = Ctrl.getIntParameter("m_block_z");

	/* Read openmp thread number */
	num_thread = Ctrl.getIntParameter("num_thread");

	/* Read DLB parameters */
	DLB_step = Ctrl.getIntParameter("DLB_step");
	DLB_tol = Ctrl.getdoubleParameter("DLB_tol");

	/* Create output directory based on process number */
	if (myid == 0)
	{
		std::string outputFolder = "./output_" + std::to_string(numprocs);
		std::filesystem::create_directory(outputFolder);

		std::filesystem::create_directory(outputFolder + "/grid");
		std::filesystem::create_directory(outputFolder + "/load");
		std::filesystem::create_directory(outputFolder + "/time_record");
		std::filesystem::create_directory(outputFolder + "/time_record/all");
		std::filesystem::create_directory(outputFolder + "/time_record/compare");
		std::filesystem::create_directory(outputFolder + "/time_record/compute");
		std::filesystem::create_directory(outputFolder + "/time_record/total");
		std::filesystem::create_directory(outputFolder + "/results");

		std::cout << "目录结构创建成功: " << outputFolder << std::endl;
	}

	return;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::Mpi_Initial

	Input(0):	None

	Function:	Initialize mpi information

	Return:		None
 *---------------------------------------------------------------------------------*/
void Euler::Mpi_Initial()
{	
	if (m_block_x <= 0 || m_block_y <= 0 || m_block_z <= 0)
	{
		if (myid == 0)
			std::cerr << "MPI block counts must be positive\n";

		MPI_Abort(MPI_COMM_WORLD, 1);
		return;
	}

	const long long required_processes =
		1LL * m_block_x * m_block_y * m_block_z;

	if (required_processes != numprocs)
	{
		if (myid == 0)
		{
			std::cerr << "MPI layout requires "
					<< required_processes
					<< " processes, but started "
					<< numprocs << std::endl;
		}

		MPI_Abort(MPI_COMM_WORLD, 1);
		return;
	}
	myid_x = int(myid % m_block_x);
	myid_z = int(floor(myid / (m_block_x * m_block_y)));
	myid_y = int(floor(myid % (m_block_x * m_block_y) / m_block_x));
	if (myid_x > 0)
		m_left = myid - 1;
	else
		m_left = MPI_PROC_NULL;
	if (myid_x < m_block_x - 1)
		m_right = myid + 1;
	else
		m_right = MPI_PROC_NULL;

	if (myid_y > 0)
		m_front = myid - m_block_x;
	else
		m_front = MPI_PROC_NULL;
	if (myid_y < m_block_y - 1)
		m_back = myid + m_block_x;
	else
		m_back = MPI_PROC_NULL;

	if (myid_z > 0)
		m_down = myid - m_block_x * m_block_y;
	else
		m_down = MPI_PROC_NULL;
	if (myid_z < m_block_z - 1)
		m_up = myid + m_block_x * m_block_y;
	else
		m_up = MPI_PROC_NULL;
	// cout << myid << " " << myid_x << " " << myid_y << " " << myid_z << " " << m_left << " " << m_right << " " << m_front << " " << m_back << " " << m_down << " " << m_up << endl;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::Computing

	Input(1):	The number of output files

	Function:	Solve and update the flow field;
				Output the result in real time

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::Computing(int count)
{
    // Reset cross-step device-state validity for every independent solve.
    configure_resident_ee_state(use_cuda && numprocs == 1 && !enable_dlb && cfl == 0
        && TimeAdv_Sch == 0 && Diff_Sch == 1 && Reaction_Sch == 0
        && type == 1 && !validate_reaction_gpu);
	double time = 0.0;
	int num = 0;
	iteration = 0;
	double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
	double t12 = 0, t23 = 0, t34 = 0, t45 = 0, t67 = 0;
	double tt1 = 0, tt2 = 0;
	ofstream loadfile("./output_" + to_string(numprocs) + "/load/LoadDegree.dat");
	tt1 = wall_seconds();

	const double time_tol = 1e-12 * std::abs(Final_Time);
	while (Final_Time - time > time_tol)
	{
		const double profile_step_start = wall_seconds();
		double profile_output_ms = 0.0;
		/* Calculate the time step */
		if (cfl == 0)
			TwoDim.dt = dt;
		else
			TwoDim.CFLcondition(cfl, Final_Time);
		// 最后一步不能超过剩余时间
		if (TwoDim.dt > Final_Time - time)
			TwoDim.dt = Final_Time - time;
		/* Ouput the real-time result unifomrly */
		if (time <= num * Final_Time / count && time + TwoDim.dt > num * Final_Time / count)
		{
			t6 = wall_seconds();
			Output(num);
			num = num + 1;
			t7 = wall_seconds();
			t67 += t7 - t6;
			profile_output_ms = (t7 - t6) * 1000.0;
		}
		t1 = wall_seconds();
		/* Update the boundary */
		//Output(10);
		TwoDim.FieldBoundary_3dRSBI();
		//Output(11);
		t2 = wall_seconds();
		t12 += t2 - t1;
		TwoDim.Mpi_Boundary();
		//Output(12);
		t3 = wall_seconds();
		t23 += t3 - t2;
		/* Solve the advection term */
        configure_ee_state_chain(use_cuda && (!enable_dlb || numprocs == 1) && TimeAdv_Sch == 0
            && Diff_Sch == 1 && Reaction_Sch == 0 && type == 1 && !validate_reaction_gpu);
		TwoDim.Advection(TimeAdv_Sch, Diff_Sch);
		//Output(13);
		// TwoDim.Update_after_Adv();
		t4 = wall_seconds();
		t34 += t4 - t3;
		/* Identify the reaction scheme */
		switch (Reaction_Sch)
		{
		case 0:
			Trapezoid(type);
			break;
		case 1:
			IMEX();
			break;
		case 2:
			DNN(type);
			break;
		}
		t5 = wall_seconds();
		//Output(14);
		t45 += t5 - t4;
		/* Monitor the iteration */
		time += TwoDim.dt;
		iteration += 1;
		if (myid == 0)
			cout << "Process " << myid << ":  Iteration:  " << iteration << "  " << TwoDim.dt << "  " << time << endl;

		/* Judge if there exists nan value */
		if (TwoDim.D.IsNan())
		{
			cout << "D is nan\n";
			Output(999);
			abort();
		}

		if (TwoDim.T.IsNan())
		{
			cout << "T is nan\n";
			Output(999);
			abort();
		}
		loadfile << iteration << '\t' << DLB.LoadDegree << '\t' << DLB.LoadDegreeBalance << '\n';
        const double profile_end = wall_seconds();
        if (myid == 0)
            std::cout << "Step profile: boundary=" << (t2-t1)*1000
                << " mpi=" << (t3-t2)*1000 << " advection=" << (t4-t3)*1000
                << " reaction_with_update=" << (t5-t4)*1000
                << " checks_and_log=" << (profile_end-t5)*1000
                << " output=" << profile_output_ms
                << " setup=" << (t1-profile_step_start)*1000-profile_output_ms
                << " total=" << (profile_end-profile_step_start)*1000 << " ms\n";
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tt2 = wall_seconds();
	trackt = tt2 - tt1;
	Output_Compute_Time(t12, t23, t34, t45, t67);

	/* For adaptive-trapezoid method, output NchemMax */
	if (Reaction_Sch == 0 && type == 1)
		React.GetNchemMax();

	return;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::Trapezoid

	Input(1):	The type of the Trapezoid formula

	Function:	Solve the reaction term by trapezoid formula
				(Adaptive (1) or not (0))

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::Trapezoid(int type)
{
	int ni = Mymesh.ni;
	int nj = Mymesh.nj;
	int nk = Mymesh.nk;
	int bc = Mymesh.bc;

	double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
	double tw1 = 0, tw2 = 0, tw3 = 0, tw4 = 0, tw5 = 0, tw6 = 0, tw7 = 0; // wait time record
	/* Update the flow field after solving the advection term */
	t1 = wall_seconds();

	configure_post_advection_chain(use_cuda && (!enable_dlb || numprocs == 1) && type == 1 && !validate_reaction_gpu);
	TwoDim.Update_after_Adv();
	t2 = wall_seconds();
	tr1 += t2 - t1;
	std::cout << "Post-advection state update: " << (t2-t1)*1000 << " ms\n";
	if (type == 0)
	{
		/* Solve the reaction source term */
		TwoDim.Di = React.Trapezoid(TwoDim.Mc, TwoDim.Di, TwoDim.T, TwoDim.dt);

		/* Update the flow field */
		TwoDim.Explicit();
	}
	else if (type == 1) /* Adaptive */
	{
        const bool local_gpu_reaction = use_cuda && (!enable_dlb || numprocs == 1);
		    if (!use_cuda && (!enable_dlb || numprocs == 1))
			{
				// 预测每个网格需要多少个反应子步
				React.TrapezoidPrediction(
					TwoDim.Mc, TwoDim.Di,
					TwoDim.Yi, TwoDim.T, TwoDim.dt);

				// 依次计算所有内部网格
				for (int i = bc; i < ni + bc; ++i)
				{
					for (int j = bc; j < nj + bc; ++j)
					{
						for (int k = bc; k < nk + bc; ++k)
						{
							for (int step = 0;
								step < React.Nchem(i, j, k);
								++step)
							{
								React.Trapezoid(
									TwoDim.Mc, TwoDim.Di,
									TwoDim.Yi, TwoDim.T,
									TwoDim.dt, step, i, j, k);

								TwoDim.Explicit(step, i, j, k);
							}
						}
					}
				}

				React.PushNchemMax();
				return;
			}
		// 单进程性能模式使用 GPU 预测。
		// 验证模式及原多进程路径仍需要 CPU 预测。
		double ttp1 = wall_seconds();
		double ttp2 = ttp1;

		if (!local_gpu_reaction || validate_reaction_gpu)
		{
			React.TrapezoidPrediction(
				TwoDim.Mc, TwoDim.Di,
				TwoDim.Yi, TwoDim.T, TwoDim.dt);

			ttp2 = wall_seconds();
			dt4 += ttp2 - ttp1;
		}
		if (local_gpu_reaction) {
			double reaction_start = wall_seconds();

			static int full_substep_checks = 0;

			ReactionFlowState gpu_state{};
			std::vector<double> gpu_sources(React.NS);
			
			// 打包前检查固定数组容量
			if (React.NS <= 0 || React.NS > REACTION_MAX_SPECIES ||
				React.NR <= 0 || React.NR > REACTION_MAX_REACTIONS)
			{
				std::cerr << "Reaction model exceeds GPU capacity\n";
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}

			int cell_count = ni * nj * nk;

			static std::vector<ReactionFlowState> batch_states;
			static std::vector<int> batch_nchem;
			static std::vector<double> batch_sources;
			batch_states.resize(cell_count);
			batch_nchem.resize(cell_count);
			batch_sources.resize(static_cast<size_t>(cell_count) * React.NS);

			int cell_idx = 0;

			if (!post_advection_chain_pending()) {
			for (int i = bc; i < ni + bc; ++i)
			{
				for (int j = bc; j < nj + bc; ++j)
				{
					for (int k = bc; k < nk + bc; ++k)
					{
						ReactionFlowState& state = batch_states[cell_idx];

						state.temperature = TwoDim.T(i, j, k);
						state.u = TwoDim.U(i, j, k);
						state.v = TwoDim.V(i, j, k);
						state.w = TwoDim.W(i, j, k);
						state.internal_energy = TwoDim.E(i, j, k);

						for (int s = 0; s < React.NS; ++s)
						{
							state.species_density[s] =
								TwoDim.Di(i, j, k, s);

							state.concentration[s] =
								TwoDim.Mc(i, j, k, s);

							state.mass_fractions[s] =
        						TwoDim.Yi(i, j, k, s);
						}

						batch_nchem[cell_idx] = React.Nchem(i, j, k);

						++cell_idx;
					}
				}
			}

			} // Otherwise reaction consumes the retained device state directly.

			double batch_gpu_start = wall_seconds();
			std::cout << "Reaction allocation_and_pack: " << (batch_gpu_start-reaction_start)*1000 << " ms\n";

			reaction_cells_host(
				batch_states.data(),
				batch_nchem.data(),
				cell_count,
				TwoDim.dt,
				React.R, React.Ru, React.P0,
				React.Mw.Getbuf(),
				TwoDim.Ri.Getbuf(),
				React.Coeff0.Getbuf(),
				React.Coeff1.Getbuf(),
				React.Af.Getbuf(),
				React.Bf.Getbuf(),
				React.Eaf.Getbuf(),
				React.Stoi_F.Getbuf(),
				React.Stoi_B.Getbuf(),
				React.React_TB.Getbuf(),
				React.NS, React.NR,
				batch_sources.data(),
				validate_reaction_gpu);

			double batch_gpu_ms =
				(wall_seconds() - batch_gpu_start) * 1000.0;

			std::cout << "Reaction batch GPU total: "
					<< batch_gpu_ms << " ms"
					<< std::endl;
			if (validate_reaction_gpu)
			{
				for (int i = bc; i < ni + bc; ++i) {
					for (int j =bc; j < nj + bc; ++j) {
						for (int k = bc; k < nk + bc; ++k) {
							for (int step = 0;
								step < React.Nchem(i, j, k);
								++step) {
									bool selected_cell =
										(i == bc && j == bc && k == bc) ||
										(i == bc + ni / 2 &&
										j == bc + nj / 2 &&
										k == bc + nk / 2) ||
										(i == bc + ni - 1 &&
										j == bc + nj - 1 &&
										k == bc + nk - 1);

									bool selected_substep =
										step == 0 ||
										step == React.Nchem(i, j, k) - 1;

									bool check_this_substep =
										full_substep_checks < 18 &&
										selected_cell &&
										selected_substep;

									if (check_this_substep)
									{
										double substep_dt =
											TwoDim.dt / React.Nchem(i, j, k);

										full_reaction_substep_test_host(
											TwoDim.T(i, j, k),
											TwoDim.U(i, j, k),
											TwoDim.V(i, j, k),
											TwoDim.W(i, j, k),
											TwoDim.E(i, j, k),
											substep_dt,
											React.R, React.Ru, React.P0,
											&TwoDim.Mc(i, j, k, 0),
											&TwoDim.Di(i, j, k, 0),
											React.Mw.Getbuf(),
											TwoDim.Ri.Getbuf(),
											React.Coeff0.Getbuf(),
											React.Coeff1.Getbuf(),
											React.Af.Getbuf(),
											React.Bf.Getbuf(),
											React.Eaf.Getbuf(),
											React.Stoi_F.Getbuf(),
											React.Stoi_B.Getbuf(),
											React.React_TB.Getbuf(),
											React.NS, React.NR,
											gpu_state,
											gpu_sources.data());
									}
									React.Trapezoid(
										TwoDim.Mc, TwoDim.Di,
										TwoDim.Yi, TwoDim.T,
										TwoDim.dt, step, i, j, k
									);

									TwoDim.Explicit(step, i, j, k);

									if (check_this_substep)
									{
										bool passed = true;

										auto compare = [&](const char* name, int component,
														double expected, double actual)
										{
											bool finite =
												std::isfinite(expected) && std::isfinite(actual);

											double error = std::abs(actual - expected);
											double tolerance = 1e-10 * (1.0 + std::abs(expected));

											if (!finite || error > tolerance)
											{
												passed = false;

												std::cout << "Full substep mismatch: " << name
														<< ", component=" << component
														<< ", CPU=" << expected
														<< ", GPU=" << actual
														<< std::endl;
											}
										};

										for (int s = 0; s < React.NS; ++s)
										{
											compare("Di", s, TwoDim.Di(i, j, k, s),
													gpu_state.species_density[s]);

											compare("Yi", s, TwoDim.Yi(i, j, k, s),
													gpu_state.mass_fractions[s]);

											compare("Mc", s, TwoDim.Mc(i, j, k, s),
													gpu_state.concentration[s]);

											compare("CMS", s, React.CMS(i, j, k, s),
													gpu_sources[s]);
										}

										for (int s = 0; s < React.NS + 4; ++s)
										{
											compare("CS", s, TwoDim.CS(i, j, k, s),
													gpu_state.conserved[s]);
										}

										compare("D", -1, TwoDim.D(i, j, k), gpu_state.density);
										compare("T", -1, TwoDim.T(i, j, k), gpu_state.temperature);

										compare("U", -1, TwoDim.U(i, j, k), gpu_state.u);
										compare("V", -1, TwoDim.V(i, j, k), gpu_state.v);
										compare("W", -1, TwoDim.W(i, j, k), gpu_state.w);

										compare("E", -1, TwoDim.E(i, j, k),
												gpu_state.internal_energy);

										compare("Wav", -1, TwoDim.Wav(i, j, k),
												gpu_state.mean_molecular_weight);

										compare("Rgas", -1, TwoDim.Rgas(i, j, k),
												gpu_state.gas_constant);

										compare("Cp", -1, TwoDim.Cp(i, j, k),
												gpu_state.specific_heat_cp);

										compare("P", -1, TwoDim.P(i, j, k), gpu_state.pressure);
										compare("H", -1, TwoDim.H(i, j, k), gpu_state.enthalpy);
										compare("Gamma", -1, TwoDim.Gamma(i, j, k), gpu_state.gamma);
										compare("C", -1, TwoDim.C(i, j, k), gpu_state.sound_speed);
										compare("Ma", -1, TwoDim.Ma(i, j, k), gpu_state.mach);

										std::cout << "Full reaction substep validation: "
												<< (passed ? "Passed" : "Failed")
												<< ", cell=(" << i << "," << j << "," << k << ")"
												<< ", step=" << step
												<< std::endl;

										if (!passed)
											MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

										++full_substep_checks;
									}
							}
						}
					}
				}

				bool batch_passed = true;
				size_t mismatch_count = 0;
				size_t boundary_difference_count = 0;

				auto compare_batch = [&](const char* name, int cell, int component,
										double expected, double actual)
				{
					double error = std::abs(actual - expected);
					double tolerance = 1e-10 * (1.0 + std::abs(expected));

					// 只对分段边界两侧、温差极小的派生量进行额外核验
					int ii = bc + cell / (nj * nk);
					int jj = bc + (cell / nk) % nj;
					int kk = bc + cell % nk;

					double cpu_t = TwoDim.T(ii, jj, kk);
					double gpu_t = batch_states[cell].temperature;

					bool crossed_boundary =
						(cpu_t < 1000.0) != (gpu_t < 1000.0);

					bool affected_quantity =
						std::strcmp(name, "Cp") == 0 ||
						std::strcmp(name, "Gamma") == 0 ||
						std::strcmp(name, "C") == 0 ||
						std::strcmp(name, "Ma") == 0;

					if (error > tolerance &&
						crossed_boundary &&
						std::abs(cpu_t - gpu_t) <= 1e-11 &&
						affected_quantity)
					{
						// 用 CPU 实现，在 GPU 温度对应的分支重新求派生量
						double cp = 0.0;
						double inverse_weight = 0.0;

						for (int s = 0; s < React.NS; ++s)
						{
							double yi = TwoDim.Yi(ii, jj, kk, s);

							cp += yi * React.GetCpi(
								gpu_t, TwoDim.Ri(s), s,
								TwoDim.Coeff0, TwoDim.Coeff1);

							inverse_weight += yi / TwoDim.Mw(s);
						}

						cp *= 1000.0;

						double gamma =
							cp / (cp - TwoDim.R * inverse_weight * 1000.0);

						double sound = std::sqrt(
							gamma * TwoDim.P(ii, jj, kk) / TwoDim.D(ii, jj, kk));

						double u = TwoDim.U(ii, jj, kk);
						double v = TwoDim.V(ii, jj, kk);
						double w = TwoDim.W(ii, jj, kk);

						double alternate = cp;

						if (std::strcmp(name, "Gamma") == 0)
							alternate = gamma;
						else if (std::strcmp(name, "C") == 0)
							alternate = sound;
						else if (std::strcmp(name, "Ma") == 0)
							alternate = std::sqrt(u*u + v*v + w*w) / sound;

						if (std::isfinite(alternate) &&
							std::isfinite(actual) &&
							std::abs(actual - alternate) <=
								1e-10 * (1.0 + std::abs(alternate)))
						{
							++boundary_difference_count;
							return;
						}
					}

					if (!std::isfinite(expected) ||
						!std::isfinite(actual) ||
						error > tolerance)
					{
						batch_passed = false;

						// 只打印前十处，避免日志过多
						if (mismatch_count < 10)
						{
							int ii = bc + cell / (nj * nk);
							int jj = bc + (cell / nk) % nj;
							int kk = bc + cell % nk;

							std::printf(
								"Mismatch %s cell=(%d,%d,%d) component=%d\n"
								"  CPU=%.17g GPU=%.17g\n"
								"  error=%.17g tolerance=%.17g\n"
								"  CPU_T=%.17g GPU_T=%.17g Nchem=%d\n",
								name, ii, jj, kk, component,
								expected, actual,
								error, tolerance,
								TwoDim.T(ii, jj, kk),
								batch_states[cell].temperature,
								batch_nchem[cell]);
						}

						++mismatch_count;
					}
				};

				cell_idx = 0;

				for (int i = bc; i < ni + bc; ++i)
				{
					for (int j = bc; j < nj + bc; ++j)
					{
						for (int k = bc; k < nk + bc; ++k)
						{
							const ReactionFlowState& state = batch_states[cell_idx];

							for (int s = 0; s < React.NS; ++s)
							{
								compare_batch("Di", cell_idx, s,
									TwoDim.Di(i, j, k, s), state.species_density[s]);

								compare_batch("Yi", cell_idx, s,
									TwoDim.Yi(i, j, k, s), state.mass_fractions[s]);

								compare_batch("Mc", cell_idx, s,
									TwoDim.Mc(i, j, k, s), state.concentration[s]);

								compare_batch("CMS", cell_idx, s,
									React.CMS(i, j, k, s),
									batch_sources[
										static_cast<size_t>(cell_idx) * React.NS + s]);
							}

							for (int s = 0; s < React.NS + 4; ++s)
							{
								compare_batch("CS", cell_idx, s,
									TwoDim.CS(i, j, k, s), state.conserved[s]);
							}

							const char* names[14] = {
								"D", "T", "U", "V", "W", "E", "Wav",
								"Rgas", "Cp", "P", "H", "Gamma", "C", "Ma"
							};

							double cpu_values[14] = {
								TwoDim.D(i,j,k), TwoDim.T(i,j,k),
								TwoDim.U(i,j,k), TwoDim.V(i,j,k), TwoDim.W(i,j,k),
								TwoDim.E(i,j,k), TwoDim.Wav(i,j,k),
								TwoDim.Rgas(i,j,k), TwoDim.Cp(i,j,k),
								TwoDim.P(i,j,k), TwoDim.H(i,j,k),
								TwoDim.Gamma(i,j,k), TwoDim.C(i,j,k), TwoDim.Ma(i,j,k)
							};

							double gpu_values[14] = {
								state.density, state.temperature,
								state.u, state.v, state.w,
								state.internal_energy, state.mean_molecular_weight,
								state.gas_constant, state.specific_heat_cp,
								state.pressure, state.enthalpy,
								state.gamma, state.sound_speed, state.mach
							};

							for (int n = 0; n < 14; ++n)
							{
								compare_batch(names[n], cell_idx, -1,
											cpu_values[n], gpu_values[n]);
							}

							++cell_idx;
						}
					}
				}

				std::cout << "Reaction batch validation: "
						<< (batch_passed ? "Passed" : "Failed")
						<< ", cells=" << cell_count
						<< ", mismatches=" << mismatch_count
						<< ", boundary differences=" << boundary_difference_count
						<< std::endl;

				if (!batch_passed)
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
			const double profile_unpack_start = wall_seconds();
			// 验证通过后，将 GPU 结果写回主流场
            bool validate_packed=false;
            const double* packed_output=reaction_packed_output(cell_count,React.NS,validate_packed);
            if(!packed_output || validate_packed){
			cell_idx = 0;

			for (int i = bc; i < ni + bc; ++i)
			{
				for (int j = bc; j < nj + bc; ++j)
				{
					for (int k = bc; k < nk + bc; ++k)
					{
						const ReactionFlowState& state = batch_states[cell_idx];
						React.Nchem(i, j, k) = batch_nchem[cell_idx];

						for (int s = 0; s < React.NS; ++s)
						{
							TwoDim.Di(i,j,k,s) = state.species_density[s];
							TwoDim.Yi(i,j,k,s) = state.mass_fractions[s];
							TwoDim.Mc(i,j,k,s) = state.concentration[s];

							React.CMS(i,j,k,s) =
								batch_sources[
									static_cast<size_t>(cell_idx) * React.NS + s];
						}

						for (int s = 0; s < React.NS + 4; ++s)
							TwoDim.CS(i,j,k,s) = state.conserved[s];

						TwoDim.D(i,j,k) = state.density;
						TwoDim.T(i,j,k) = state.temperature;

						TwoDim.U(i,j,k) = state.u;
						TwoDim.V(i,j,k) = state.v;
						TwoDim.W(i,j,k) = state.w;
						TwoDim.E(i,j,k) = state.internal_energy;

						TwoDim.Wav(i,j,k) = state.mean_molecular_weight;
						TwoDim.Rgas(i,j,k) = state.gas_constant;
						TwoDim.Cp(i,j,k) = state.specific_heat_cp;

						TwoDim.P(i,j,k) = state.pressure;
						TwoDim.H(i,j,k) = state.enthalpy;
						TwoDim.Gamma(i,j,k) = state.gamma;
						TwoDim.C(i,j,k) = state.sound_speed;
						TwoDim.Ma(i,j,k) = state.mach;

						++cell_idx;
					}
				}
			}

            }
            if(packed_output){
                struct Target {const char* name; double* data; size_t length; int width;};
                const Target targets[]={
                    {"Di",TwoDim.Di.Getbuf(),TwoDim.Di.GetLen(),React.NS},
                    {"Yi",TwoDim.Yi.Getbuf(),TwoDim.Yi.GetLen(),React.NS},
                    {"Mc",TwoDim.Mc.Getbuf(),TwoDim.Mc.GetLen(),React.NS},
                    {"CMS",React.CMS.Getbuf(),React.CMS.GetLen(),React.NS},
                    {"CS",TwoDim.CS.Getbuf(),TwoDim.CS.GetLen(),React.NS+4},
                    {"D",TwoDim.D.Getbuf(),TwoDim.D.GetLen(),1},
                    {"T",TwoDim.T.Getbuf(),TwoDim.T.GetLen(),1},
                    {"U",TwoDim.U.Getbuf(),TwoDim.U.GetLen(),1},
                    {"V",TwoDim.V.Getbuf(),TwoDim.V.GetLen(),1},
                    {"W",TwoDim.W.Getbuf(),TwoDim.W.GetLen(),1},
                    {"E",TwoDim.E.Getbuf(),TwoDim.E.GetLen(),1},
                    {"Wav",TwoDim.Wav.Getbuf(),TwoDim.Wav.GetLen(),1},
                    {"Rgas",TwoDim.Rgas.Getbuf(),TwoDim.Rgas.GetLen(),1},
                    {"Cp",TwoDim.Cp.Getbuf(),TwoDim.Cp.GetLen(),1},
                    {"P",TwoDim.P.Getbuf(),TwoDim.P.GetLen(),1},
                    {"H",TwoDim.H.Getbuf(),TwoDim.H.GetLen(),1},
                    {"Gamma",TwoDim.Gamma.Getbuf(),TwoDim.Gamma.GetLen(),1},
                    {"C",TwoDim.C.Getbuf(),TwoDim.C.GetLen(),1},
                    {"Ma",TwoDim.Ma.Getbuf(),TwoDim.Ma.GetLen(),1},
                };
                const size_t full_cells=static_cast<size_t>(ni+2*bc)*(nj+2*bc)*(nk+2*bc);
                // Confirm the grid layout before bypassing Array's index operator.
                for(const auto& target:targets)
                    if(target.length!=full_cells*target.width){
                        std::cerr<<"Packed output shape error: "<<target.name<<std::endl;
                        MPI_Abort(MPI_COMM_WORLD,1);
                    }
                if(React.Nchem.GetLen()!=full_cells){
                    std::cerr<<"Packed output shape error: Nchem\n";
                    MPI_Abort(MPI_COMM_WORLD,1);
                }
                std::vector<std::vector<unsigned char>> reference;
                std::vector<unsigned char> reference_nchem;
                if(validate_packed){
                    // Old scatter ran above. Snapshot the entire arrays, including halos.
                    // Poison interiors so missing writes cannot pass accidentally.
                    for(const auto& target:targets){
                        reference.emplace_back(target.length*sizeof(double));
                        std::memcpy(reference.back().data(),target.data,reference.back().size());
                        for(int i=bc;i<ni+bc;++i) for(int j=bc;j<nj+bc;++j){
                            const size_t row=((static_cast<size_t>(i)*(nj+2*bc)+j)*(nk+2*bc)+bc)*target.width;
                            std::memset(target.data+row,0xA5,static_cast<size_t>(nk)*target.width*sizeof(double));
                        }
                    }
                    reference_nchem.resize(full_cells*sizeof(int));
                    std::memcpy(reference_nchem.data(),React.Nchem.Getbuf(),reference_nchem.size());
                    for(int i=bc;i<ni+bc;++i) for(int j=bc;j<nj+bc;++j){
                        const size_t row=(static_cast<size_t>(i)*(nj+2*bc)+j)*(nk+2*bc)+bc;
                        std::memset(React.Nchem.Getbuf()+row,0xA5,static_cast<size_t>(nk)*sizeof(int));
                    }
                }
                size_t offset=0;
                const char* fast_unpack_env=std::getenv("EU3D_FAST_UNPACK");
                const bool fast_unpack=fast_unpack_env && std::strcmp(fast_unpack_env,"1")==0;
                for(const auto& target:targets){
                    copy_packed_interior(target.data,packed_output+offset,ni,nj,nk,bc,target.width,fast_unpack);
                    offset+=static_cast<size_t>(cell_count)*target.width;
                }
                copy_packed_interior(React.Nchem.Getbuf(),batch_nchem.data(),ni,nj,nk,bc,1,fast_unpack);
                if(validate_packed){
                    size_t different_fields=0, f=0;
                    for(const auto& target:targets){
                        if(std::memcmp(reference[f].data(),target.data,reference[f].size())!=0){
                            std::cerr<<"Packed output mismatch: field="<<target.name<<std::endl;
                            ++different_fields;
                        }
                        ++f;
                    }
                    if(std::memcmp(reference_nchem.data(),React.Nchem.Getbuf(),reference_nchem.size())!=0){
                        std::cerr<<"Packed output mismatch: field=Nchem\n";
                        ++different_fields;
                    }
                    std::cout<<"Packed output validation: "<<(different_fields?"Failed":"Passed")
                        <<", cells="<<cell_count<<", different_fields="<<different_fields
                        <<", comparison=bitwise, halos=checked"<<std::endl;
                    if(different_fields) MPI_Abort(MPI_COMM_WORLD,1);
                }
            }

			std::cout << "Reaction unpack: " << (wall_seconds()-profile_unpack_start)*1000 << " ms\n";
			tr2 += wall_seconds() - reaction_start;
			
			const double profile_max_start = wall_seconds();
			React.PushNchemMax();
			std::cout << "Reaction history: " << (wall_seconds()-profile_max_start)*1000 << " ms\n";
			return;
		}
		/* Dynamic Load Balance */
		int nchemSumBalance = 0;
		/* Reassign work load */
		if (iteration % DLB_step == 0)
		{
			DLB.NchemSumBalance = 0;
			React.NchemSum = React.Nchem.SumNoBoundary(bc);
			MPI_Allgather(&React.NchemSum, 1, MPI_INT, &React.NchemTotal(0), 1, MPI_INT, MPI_COMM_WORLD);
			DLB.LoadDegree = double((React.NchemTotal.MaxValue() - React.NchemTotal.AveValue())) / React.NchemTotal.MaxValue();
			if (myid == 0)
				cout << "Load Degree: " << DLB.LoadDegree << endl;
			if (DLB.LoadDegree > DLB_tol)
			{
				// if (myid == 0)
				// {
				// cout << "Befor DLB: ";
				// React.NchemTotal.Print();
				// }
				DLB.transferNchem.clear();
				/* get the transfer records */
				DLB.DLBPriorityQueue(React.NchemTotal);
				DLB.transferMeshNum.resize(DLB.transferNchem.size());
				DLB.transferNmesh.resize(DLB.transferNchem.size());
				std::fill(DLB.transferMeshNum.begin(), DLB.transferMeshNum.end(), 0);
				std::fill(DLB.transferNmesh.begin(), DLB.transferNmesh.end(), 0);
				DLB.transferPrevDataLocal.clear();
				DLB.transferPrevData.clear();
				DLB.transferPrevNchemLocal.clear();
				DLB.transferPrevNchem.clear();
				DLB.transferIndex.clear();
				DLB.sentNmesh = 0;
				DLB.prevFlowDataLocal.clear();
				DLB.prevReactDataLocal.clear();
			}
		}
		MPI_Request sendRequests1[DLB.transferNchem.size()];
		MPI_Request sendRequests2[DLB.transferNchem.size()];
		MPI_Request sendRequests3[DLB.transferNchem.size()];
		MPI_Request recvRequests1[DLB.transferNchem.size()];
		MPI_Request recvRequests2[DLB.transferNchem.size()];
		MPI_Request recvRequests3[DLB.transferNchem.size()];
		MPI_Status statuses[DLB.transferNchem.size()];
		for (int i = 0; i < DLB.transferNchem.size(); i++)
		{
			sendRequests1[i] = MPI_REQUEST_NULL;
			sendRequests2[i] = MPI_REQUEST_NULL;
			sendRequests3[i] = MPI_REQUEST_NULL;
			recvRequests1[i] = MPI_REQUEST_NULL;
			recvRequests2[i] = MPI_REQUEST_NULL;
			recvRequests3[i] = MPI_REQUEST_NULL;
		}
		if (DLB.LoadDegree > DLB_tol)
		{
			double tt1 = wall_seconds();
			/* send myid judge transferIndex, package data, transfer mesh number */
			int sentNchem = DLB.GetSentNchem(myid);
			// cout << "\tmyid " << myid << " Nchem: " << sentNchem << endl;
			if (sentNchem != 0)
			{
				DLB.prevFlowDataLocal.clear();
				DLB.prevReactDataLocal.clear();

				std::vector<int> sendToTemp = DLB.sendTo;
				int nchem_temp = 0, id = 0, start = 0, n = 0;

				/* Get the transfer mesh number & transferIndex & package data */
				if (iteration % DLB_step == 0)
				{
					DLB.GetTransferMesh(React.Nchem, bc);
					DLB.sentNmesh = std::accumulate(DLB.transferMeshNum.begin(), DLB.transferMeshNum.end(), 0);
				}
				int x = 0, y = 0, z = 0;
				for (int i = 0; i < DLB.transferIndex.size() / 3; i++)
				{
					x = DLB.transferIndex[3 * i];
					y = DLB.transferIndex[3 * i + 1];
					z = DLB.transferIndex[3 * i + 2];
					TwoDim.PackagePrev(DLB.prevFlowDataLocal, x, y, z);
					React.PackagePrev(DLB.prevReactDataLocal, x, y, z);
				}
			}
			double tt2 = wall_seconds();
			dt1 += tt2 - tt1;
			if (iteration % DLB_step == 0)
			{ /* reduce transfer mesh number to 0 process */
				MPI_Reduce(DLB.transferMeshNum.data(), DLB.transferNmesh.data(), DLB.transferMeshNum.size(), MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Bcast(DLB.transferNmesh.data(), DLB.transferNmesh.size(), MPI_INT, 0, MPI_COMM_WORLD);
				if (myid == 0)
				{
					cout << "TransferNmesh: " << std::accumulate(DLB.transferNmesh.begin(), DLB.transferNmesh.end(), 0) << endl;
				}
			}
			double ttt1 = wall_seconds();
			if (sentNchem != 0)
				for (int i = 0; i < DLB.sendTo.size(); i++)
				{
					int start1 = std::accumulate(DLB.transferNmesh.begin() + DLB.sendTo[0], DLB.transferNmesh.begin() + DLB.sendTo[i], 0) * (5 + 3 * TwoDim.NS);
					int start2 = std::accumulate(DLB.transferNmesh.begin() + DLB.sendTo[0], DLB.transferNmesh.begin() + DLB.sendTo[i], 0);
					int size1 = DLB.transferNmesh[DLB.sendTo[i]] * (5 + 3 * TwoDim.NS);
					int size2 = DLB.transferNmesh[DLB.sendTo[i]];
					if (size2 != 0)
					{
						MPI_Isend(&DLB.prevFlowDataLocal[start1], size1, MPI_DOUBLE, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
								  DLB.sendTo[i], MPI_COMM_WORLD, &sendRequests1[DLB.sendTo[i]]);
						MPI_Isend(&DLB.prevReactDataLocal[start2], size2, MPI_INT, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
								  DLB.sendTo[i] + 10000, MPI_COMM_WORLD, &sendRequests2[DLB.sendTo[i]]);
					}
				}
			double ttt2 = wall_seconds();
			int recvNmesh = DLB.GetRecvNmesh(myid);
			/* Unpackage update data */
			if (recvNmesh != 0)
			{
				DLB.recvPrevFlowData.clear();
				DLB.recvPrevReactData.clear();
				DLB.recvPrevFlowData.resize(recvNmesh * (5 + 3 * TwoDim.NS));
				DLB.recvPrevReactData.resize(recvNmesh);
				for (int i = 0; i < DLB.recvFrom.size(); i++)
				{
					int start1 = std::accumulate(DLB.transferNmesh.begin() + DLB.recvFrom[0], DLB.transferNmesh.begin() + DLB.recvFrom[i], 0) * (5 + 3 * TwoDim.NS);
					int start2 = std::accumulate(DLB.transferNmesh.begin() + DLB.recvFrom[0], DLB.transferNmesh.begin() + DLB.recvFrom[i], 0);
					int size1 = DLB.transferNmesh[DLB.recvFrom[i]] * (5 + 3 * TwoDim.NS);
					int size2 = DLB.transferNmesh[DLB.recvFrom[i]];
					if (size2 != 0)
					{
						// MPI_Recv(&DLB.recvPrevFlowData[start1], size1, MPI_DOUBLE, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
						// 		 DLB.recvFrom[i], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						// MPI_Recv(&DLB.recvPrevReactData[start2], size2, MPI_INT, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
						// 		 DLB.recvFrom[i] + 10000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						MPI_Irecv(&DLB.recvPrevFlowData[start1], size1, MPI_DOUBLE, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
								  DLB.recvFrom[i], MPI_COMM_WORLD, &recvRequests1[DLB.recvFrom[i]]);
						MPI_Irecv(&DLB.recvPrevReactData[start2], size2, MPI_INT, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
								  DLB.recvFrom[i] + 10000, MPI_COMM_WORLD, &recvRequests2[DLB.recvFrom[i]]);
					}
				}
			}
		}
		int tag = 0;
		nchemSumBalance = 0;
		// #pragma omp parallel for num_threads(num_thread) collapse(3) schedule(dynamic, 100) reduction(+ : nchemSumBalance)
		for (int i = bc; i < (ni - 1) / 2 + bc; i++)
			for (int j = bc; j < nj + bc; j++)
				for (int k = bc; k < nk + bc; k++)
				{
					int n = 0;
					bool flag = false;
					while (n < DLB.sentNmesh && tag < DLB.sentNmesh)
					{
						if (i == DLB.transferIndex[3 * n] &&
							j == DLB.transferIndex[3 * n + 1] &&
							k == DLB.transferIndex[3 * n + 2])
						{
							flag = true;
							tag++;
							break;
						}
						n++;
					}
					if (!flag)
					{
						nchemSumBalance += React.Nchem(i, j, k);
						for (int step = 0; step < React.Nchem(i, j, k); step++)
						{
							/* Solve the reaction source term */
							React.Trapezoid(TwoDim.Mc, TwoDim.Di, TwoDim.Yi, TwoDim.T, TwoDim.dt, step, i, j, k);

							/* Update the flow field */
							TwoDim.Explicit(step, i, j, k);
						}
					}
				}
		DLB.NchemSumBalance += nchemSumBalance;

		tw1 = wall_seconds();
		MPI_Waitall(DLB.transferNchem.size(), sendRequests1, statuses);
		MPI_Waitall(DLB.transferNchem.size(), sendRequests2, statuses);
		MPI_Waitall(DLB.transferNchem.size(), recvRequests1, statuses);
		MPI_Waitall(DLB.transferNchem.size(), recvRequests2, statuses);
		tw2 = wall_seconds();

		if (DLB.LoadDegree > DLB_tol)
		{
			int recvNmesh = DLB.GetRecvNmesh(myid);
			/* Unpack transfer data */
			if (recvNmesh != 0)
			{
				nchemSumBalance = 0;
				ExtraFlow.ReConstruction(recvNmesh);
				ExtraFlow.UnpackagePrev(DLB.recvPrevFlowData, DLB.recvFrom, DLB.transferNmesh);
				ExtraReact.ReConstruction(recvNmesh);
				ExtraReact.UnpackagePrev(DLB.recvPrevReactData, DLB.recvFrom, DLB.transferNmesh);
				// cout << myid << " Start DLB\n";

				// #pragma omp parallel for num_threads(num_thread) collapse(1) schedule(dynamic) reduction(+ : nchemSumBalance)
				for (int i = 0; i < recvNmesh; i++)
				{
					// DLB.NchemSumBalance += ExtraReact.Nchem(i, 0, 0);
					nchemSumBalance += ExtraReact.Nchem(i, 0, 0);
					for (int step = 0; step < ExtraReact.Nchem(i, 0, 0); step++)
					{
						/* Solve the reaction source term */
						ExtraReact.Trapezoid(ExtraFlow.Mc, ExtraFlow.Di, ExtraFlow.Yi, ExtraFlow.T, TwoDim.dt, step, i, 0, 0);
						/* Update the flow field */
						ExtraFlow.Explicit(step, i, 0, 0);
					}
				}
				DLB.NchemSumBalance += nchemSumBalance;
				double tt4 = wall_seconds();
				// dt3 += tt4 - tt3;
				DLB.updateFlowDataLocal.clear();
				DLB.updateFlowDataLocal.resize(recvNmesh * (18 + 4 * TwoDim.NS));
				ExtraFlow.PackageUpdate(DLB.updateFlowDataLocal, DLB.recvFrom, DLB.transferNmesh);
				bool has_nan = std::any_of(DLB.updateFlowDataLocal.begin(), DLB.updateFlowDataLocal.end(), [](double x)
										   { return std::isnan(x); });
				if (has_nan)
				{
					std::cout << "DLB contains NaN." << std::endl;
					abort();
				}
				for (int i = 0; i < DLB.recvFrom.size(); i++)
				{
					int start = std::accumulate(DLB.transferNmesh.begin() + DLB.recvFrom[0], DLB.transferNmesh.begin() + DLB.recvFrom[i], 0) * (18 + 4 * TwoDim.NS);
					int size = DLB.transferNmesh[DLB.recvFrom[i]] * (18 + 4 * TwoDim.NS);
					if (size != 0)
						MPI_Isend(&DLB.updateFlowDataLocal[start], size, MPI_DOUBLE, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
								  DLB.recvFrom[i] + 20000, MPI_COMM_WORLD, &sendRequests3[DLB.recvFrom[i]]);
				}
			}
		}
		t3 = wall_seconds();

		t4 = wall_seconds();
		tr2 += t4 - t3; /* trapezoid time */

		if (DLB.LoadDegree > DLB_tol)
		{
			int sentNchem = DLB.GetSentNchem(myid);
			DLB.recvUpdateFlowData.clear();
			DLB.recvUpdateFlowData.resize(DLB.sentNmesh * (18 + 4 * TwoDim.NS));
			/* Unpackage update data */
			if (sentNchem != 0)
			{
				double tt3 = wall_seconds();
				for (int i = 0; i < DLB.sendTo.size(); i++)
				{
					int start = std::accumulate(DLB.transferNmesh.begin() + DLB.sendTo[0], DLB.transferNmesh.begin() + DLB.sendTo[i], 0) * (18 + 4 * TwoDim.NS);
					int size = DLB.transferNmesh[DLB.sendTo[i]] * (18 + 4 * TwoDim.NS);
					if (size != 0)
					{
						// MPI_Recv(&DLB.recvUpdateFlowData[start], size, MPI_DOUBLE, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
						// 		 DLB.sendTo[i] + 20000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						MPI_Irecv(&DLB.recvUpdateFlowData[start], size, MPI_DOUBLE, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
								  DLB.sendTo[i] + 20000, MPI_COMM_WORLD, &recvRequests3[DLB.sendTo[i]]);
					}
				}
				double tt4 = wall_seconds();
				dt6 += tt4 - tt3;
			}

			// dt5 += tt2 - tt1;
			// dt6 += tt3 - tt2;
		}

		nchemSumBalance = 0;
		// #pragma omp parallel for num_threads(num_thread) collapse(3) schedule(dynamic, 100) reduction(+ : nchemSumBalance)
		for (int i = (ni - 1) / 2 + bc; i < ni + bc; i++)
			for (int j = bc; j < nj + bc; j++)
				for (int k = bc; k < nk + bc; k++)
				{
					int n = 0;
					bool flag = false;
					while (n < DLB.sentNmesh && tag < DLB.sentNmesh)
					{
						if (i == DLB.transferIndex[3 * n] &&
							j == DLB.transferIndex[3 * n + 1] &&
							k == DLB.transferIndex[3 * n + 2])
						{
							flag = true;
							tag++;
							break;
						}
						n++;
					}
					if (!flag)
					{
						nchemSumBalance += React.Nchem(i, j, k);
						for (int step = 0; step < React.Nchem(i, j, k); step++)
						{
							/* Solve the reaction source term */
							React.Trapezoid(TwoDim.Mc, TwoDim.Di, TwoDim.Yi, TwoDim.T, TwoDim.dt, step, i, j, k);

							/* Update the flow field */
							TwoDim.Explicit(step, i, j, k);
						}
					}
				}
		DLB.NchemSumBalance += nchemSumBalance;

		tw3 = wall_seconds();
		MPI_Waitall(DLB.transferNchem.size(), sendRequests3, statuses);
		MPI_Waitall(DLB.transferNchem.size(), recvRequests3, statuses);
		tw4 = wall_seconds();

		if (DLB.LoadDegree > DLB_tol)
		{
			int sentNchem = DLB.GetSentNchem(myid);
			if (sentNchem != 0)
				TwoDim.UnpackageUpdate(DLB.recvUpdateFlowData, DLB.transferIndex, DLB.sendTo, DLB.transferNmesh);
		}
		t5 = wall_seconds();

		// dt3 += t5 - t4;
		React.PushNchemMax();
		t6 = wall_seconds();
		// MPI_Barrier(MPI_COMM_WORLD);
		t7 = wall_seconds();
		tr3 += t3 - ttp2 + t5 - t4;	  /* DLB time */
		tr4 += tw4 - tw3 + tw2 - tw1; /* wait time */

		/* Monitor the reaction iteration step */
		if (iteration % DLB_step == 0)
		{
			MPI_Allgather(&DLB.NchemSumBalance, 1, MPI_INT, &React.NchemTotalBalance(0), 1, MPI_INT, MPI_COMM_WORLD);
			DLB.LoadDegreeBalance = double((React.NchemTotalBalance.MaxValue() - React.NchemTotalBalance.AveValue())) / React.NchemTotalBalance.MaxValue();
			if (myid == 0)
			{
				// cout << "After DLB: ";
				// React.NchemTotal.Print();
				cout << "Balance Load Degree: " << DLB.LoadDegreeBalance << endl;
				// cout << "Balance Load: ";
				// React.NchemTotalBalance.Print();
				// cout << "Diff: \n";
				// for (int i = 0; i < React.NchemTotal.GetSize(); i++)
				// 	cout << React.NchemTotal(i) - React.NchemTotalBalance(i) << " ";
				// cout << endl;
				// std::cout << "\tTransfer records:" << std::endl;
				// for (int i = 0; i < DLB.transferNchem.size(); i++)
				// {
				// 	int sender = std::get<0>(DLB.transferNchem[i]);
				// 	int receiver = std::get<1>(DLB.transferNchem[i]);
				// 	int amount = std::get<2>(DLB.transferNchem[i]);
				// 	std::cout << "\tSender: " << sender << " Receiver: " << receiver << " Transfer Nchem: " << amount << " Transfer Nmesh: " << DLB.transferNmesh[i] << std::endl;
				// }
			}
		}
		int max_it = 0;
		MPI_Reduce(&React.NchemNow, &max_it, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		if (myid == 0)
			cout << "MAX Reaction iteration step:  " << max_it << endl;
		// cout << "Process " << myid << ":  Reaction iteration step:  " << React.NchemNow << endl;
	}
}
/*---------------------------------------------------------------------------------
	Name:		Euler::IMEX

	Input(0):	None

	Function:	Solve the reaction term by IMEX method

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::IMEX()
{
	double t1 = 0, t2 = 0, t3 = 0;
	/* Solve the reaction source term */
	t1 = wall_seconds();
	React.Diagonalized(TwoDim.Mc, TwoDim.Di, TwoDim.T, TwoDim.Partial_T);
	t2 = wall_seconds();
	tr1 += t2 - t1;
	/* Update the flow field */
	TwoDim.Update_IMEX(React.CMS, React.MD);
	t3 = wall_seconds();
	tr2 += t3 - t2;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::DNN

	Input(1):	The type of the reaction term solver

	Function:	Solve the reaction term by Cantera or DNN method
				(DNN (1) or Cantera (0))

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::DNN(int type)
{
	TwoDim.Update_after_Adv();

	// TwoDim.Di = React.ReactionS(type, TwoDim.Di, TwoDim.T, TwoDim.P, TwoDim.dt);

	TwoDim.Explicit();
}

/*---------------------------------------------------------------------------------
	Name:		NS::Output

	Input(1):	the number of the output file

	Function:	Output the result in a file

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::Output(int num)
{
    // EU3D_DUMP_STATE: compare final interior nodes across MPI decompositions.
    const char* dump_requested = std::getenv("EU3D_DUMP_STATE");
    if (dump_requested && std::strcmp(dump_requested, "1") == 0 && num == count)
    {
        const std::uint16_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1 || sizeof(double) != 8) {
            std::cerr << "State dump requires little-endian 8-byte doubles\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
        const std::string dump_name = "./output_" + std::to_string(numprocs)
            + "/results/state_check_" + std::to_string(num)
            + "_rank" + std::to_string(myid) + ".bin";
        std::ofstream dump(dump_name, std::ios::binary | std::ios::trunc);
        if (!dump) {
            std::cerr << "Cannot open state dump: " << dump_name << '\n';
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
        dump << "EU3D_STATE_V1_LE\n" << std::setprecision(17)
             << React.NS << ' ' << Mymesh.total_ni << ' ' << Mymesh.total_nj << ' ' << Mymesh.total_nk << ' '
             << Mymesh.ni << ' ' << Mymesh.nj << ' ' << Mymesh.nk << ' '
             << myid_x << ' ' << myid_y << ' ' << myid_z << ' '
             << m_block_x << ' ' << m_block_y << ' ' << m_block_z << ' '
             << myid << ' ' << numprocs << ' ' << iteration << ' ' << num << ' '
             << Final_Time << ' ' << dt << '\n';
        dump << "X Y Z D T U V W E Wav Rgas Cp P H Gamma C Ma";
        for (const char* field : {"Di", "Yi", "Mc"})
            for (int n=0;n<React.NS;++n) dump << ' ' << field << n;
        for (int n=0;n<React.NS+4;++n) dump << " CS" << n;
        for (int n=0;n<React.NS;++n) dump << " CMS" << n;
        dump << " Nchem\n";
        std::vector<double> row;
        row.reserve(22 + 5*React.NS);
        const int b = Mymesh.bc;
        for (int i=b;i<b+Mymesh.ni;++i)
            for (int j=b;j<b+Mymesh.nj;++j)
                for (int k=b;k<b+Mymesh.nk;++k) {
                    row = {Mymesh.xnode(i), Mymesh.ynode(j), Mymesh.znode(k),
                        TwoDim.D(i,j,k), TwoDim.T(i,j,k),
                        TwoDim.U(i,j,k), TwoDim.V(i,j,k), TwoDim.W(i,j,k),
                        TwoDim.E(i,j,k), TwoDim.Wav(i,j,k), TwoDim.Rgas(i,j,k),
                        TwoDim.Cp(i,j,k), TwoDim.P(i,j,k), TwoDim.H(i,j,k),
                        TwoDim.Gamma(i,j,k), TwoDim.C(i,j,k), TwoDim.Ma(i,j,k)};
                    for(int n=0;n<React.NS;++n) row.push_back(TwoDim.Di(i,j,k,n));
                    for(int n=0;n<React.NS;++n) row.push_back(TwoDim.Yi(i,j,k,n));
                    for(int n=0;n<React.NS;++n) row.push_back(TwoDim.Mc(i,j,k,n));
                    for(int n=0;n<React.NS+4;++n) row.push_back(TwoDim.CS(i,j,k,n));
                    for(int n=0;n<React.NS;++n) row.push_back(React.CMS(i,j,k,n));
                    row.push_back(React.Nchem(i,j,k));
                    dump.write(reinterpret_cast<const char*>(row.data()), row.size()*sizeof(double));
                }
        dump.close();
        if (!dump) {
            std::cerr << "State dump write failed: " << dump_name << '\n';
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
        std::cout << "State dump: rank=" << myid << ", file=" << dump_name << std::endl;
    }

	int ni = Mymesh.ni;
	int nj = Mymesh.nj;
	int nk = Mymesh.nk;
	int bc = Mymesh.bc;

	/* Set the filename */
	string rea = react;
	string filename;
	if (numprocs == 1)
		filename = "./output_" + to_string(numprocs) + "/results/Series" + rea + to_string(num) + +"_" + to_string(myid) + ".dat";
	else
		filename = "./output_" + to_string(numprocs) + "/results/" + rea + to_string(num) + +"_" + to_string(myid) + ".dat";

	/* Output the result */
	ofstream outfile;
	outfile.open(filename, ios::out);

	// /* The result without boundary */
	// outfile << "Variables = X,Y,Z,T,D,P,U,V,Ma,Y1,Y2,Y3,Y4,Y5,Y6,Y7,L,WaitTime\n";
	// outfile << "ZONE I=" << ni << '\t' << "J=" << nj << '\t' << "K=" << nk << '\n';
	// outfile << "datapacking=block\n";
	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << Mymesh.xnode(i) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << Mymesh.ynode(j) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << Mymesh.znode(k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.T(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.D(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.P(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.U(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.V(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Ma(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 0) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 1) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 2) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 3) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 4) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 5) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 6) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << React.Nchem(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << tr4 << '\n';

	/* The result with boundary */
	outfile << "Variables = X,Y,Z,T,D,P,U,V,Ma,Y1,Y2,Y3,Y4,Y5,Y6,Y7\n";
	outfile << "ZONE I=" << ni + 2 * bc << '\t' << "J=" << nj + 2 * bc << '\t' << "K=" << nk + 2 * bc << '\n';
	outfile << "datapacking=block\n";
	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << Mymesh.xnode(i) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << Mymesh.ynode(j) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << Mymesh.znode(k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.T(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.D(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.P(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.U(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.V(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Ma(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 0) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 1) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 2) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 3) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 4) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 5) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 6) << '\n';

	outfile.close();

	return;
}

/*---------------------------------------------------------------------------------
	Name:		NS::Output

	Input(1):	the number of the output file

	Function:	Output the result in a file

	Return:		None
 *---------------------------------------------------------------------------------*/
void Euler::Output_Total_Time(double t1, double t2, double t3, double t4, double t5, double t6, double t7, double t8, double t9, double t10)
{
	auto t12 = (t2 - t1);
	auto t23 = (t3 - t2);
	auto t34 = (t4 - t3);
	auto t45 = (t5 - t4);
	auto t56 = (t6 - t5);
	auto t67 = (t7 - t6);
	auto t78 = (t8 - t7);
	auto t89 = (t9 - t8);
	auto t90 = (t10 - t9);

	string filename = "./output_" + to_string(numprocs) + "/time_record/total/Total_Time_" + to_string(myid) + ".txt";
	ofstream outfile(filename);
	outfile << "myid:" << '\t' << myid << '\n';
	outfile << "FileRead:" << '\t' << t12 << '\n';
	outfile << "Mpi_Initial:" << '\t' << t23 << '\n';
	outfile << "MeshProcess:" << '\t' << t34 << '\n';
	outfile << "InputRead:" << '\t' << t45 << '\n';
	outfile << "ReactionRead:" << '\t' << t56 << '\n';
	outfile << "FieldInitial:" << '\t' << t67 << '\n';
	outfile << "MPI_Barrier:" << '\t' << t78 << '\n';
	outfile << "Computing:" << '\t' << t89 << '\n';
	outfile << "Output:" << '\t' << t90 << '\n';
	outfile.close();

	double maxt = 0;
	MPI_Reduce(&t89, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (myid == 0)
	{
		string rea = react;
		filename = "./output_" + to_string(numprocs) + "/time_record/compare/" + rea + "_mpi_" + to_string(m_block_x * m_block_y) + "_omp_" + to_string(num_thread) + ".txt";
		ofstream outfile1(filename);
		outfile1 << maxt << '\t' << trackt;
		outfile1.close();
		std::cout << "Loop time: " << maxt << std::endl;
	}
}
void Euler::Output_Compute_Time(double t12, double t23, double t34, double t45, double t67)
{
	string file_compute = "./output_" + to_string(numprocs) + "/time_record/compute/Compute_Time_" + to_string(myid) + ".txt";
	ofstream outfile(file_compute);
	outfile << "myid:" << '\t' << myid << '\n';
	outfile << "FieldBoundary:" << '\t' << t12 << '\n';
	outfile << "Mpi_Boundary:" << '\t' << t23 << '\n';
	outfile << "Advection:" << '\t' << t34 << '\n';
	outfile << "Reaction:" << '\t' << t45 << '\n';
	outfile << "Output:" << '\t' << t67 << '\n';
	switch (Reaction_Sch)
	{
	case 0:
		outfile << "Update_after_Advection:" << '\t' << tr1 << '\n';
		outfile << "Trapezoid_Prediction:" << '\t' << dt4 << '\n';
		outfile << "Trapezoid:" << '\t' << tr2 + dt3 << '\n';
		outfile << "DLB:" << '\t' << tr3 - dt3 << '\n';
		outfile << "Wait:" << '\t' << tr4 << '\n';
		break;
	case 1:
		outfile << "Diagnalized:" << '\t' << tr1 << '\n';
		outfile << "Update_IMEX:" << '\t' << tr2 << '\n';
		outfile << "FieldBoundary:" << TwoDim.ft1 << " " << TwoDim.ft2 << " " << TwoDim.ft3 << " " << TwoDim.ft4 << " " << TwoDim.ft5 << " " << TwoDim.ft6 << '\n';
		break;
	}

	outfile.close();
	double recv_wait_time[m_block_x * m_block_y];
	double recv_react_time[m_block_x * m_block_y];
	double recv_advect_time[m_block_x * m_block_y];
	double recv_DLB_time[m_block_x * m_block_y];
	double recv_DLB_time1[m_block_x * m_block_y];
	double recv_DLB_time2[m_block_x * m_block_y];
	double recv_DLB_time3[m_block_x * m_block_y];
	double recv_DLB_time4[m_block_x * m_block_y];
	double recv_DLB_time5[m_block_x * m_block_y];
	double recv_DLB_time6[m_block_x * m_block_y];
	double react_time = tr1 + tr2 + dt3 + dt4;
	double dlb_time = tr3 - dt3;
	MPI_Gather(&tr4, 1, MPI_DOUBLE, &recv_wait_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&react_time, 1, MPI_DOUBLE, &recv_react_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&t34, 1, MPI_DOUBLE, &recv_advect_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dlb_time, 1, MPI_DOUBLE, &recv_DLB_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt1, 1, MPI_DOUBLE, &recv_DLB_time1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt2, 1, MPI_DOUBLE, &recv_DLB_time2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt3, 1, MPI_DOUBLE, &recv_DLB_time3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt5, 1, MPI_DOUBLE, &recv_DLB_time4, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt6, 1, MPI_DOUBLE, &recv_DLB_time5, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt7, 1, MPI_DOUBLE, &recv_DLB_time6, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (myid == 0)
	{
		string file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_wait.txt";
		ofstream outfile1(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile1 << recv_wait_time[i] << endl;
		outfile1.close();

		file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_react.txt";
		ofstream outfile2(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile2 << recv_react_time[i] << endl;
		outfile2.close();

		file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_advect.txt";
		ofstream outfile3(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile3 << recv_advect_time[i] << endl;
		outfile3.close();

		file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_DLB.txt";
		ofstream outfile4(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile4 << recv_DLB_time[i] << '\t' << recv_DLB_time1[i] << '\t' << recv_DLB_time2[i] << '\t'
					 << recv_DLB_time3[i] << '\t' << recv_DLB_time4[i] << '\t' << recv_DLB_time5[i] << '\t' << recv_DLB_time6[i] << endl;
		outfile4.close();
	}
}
