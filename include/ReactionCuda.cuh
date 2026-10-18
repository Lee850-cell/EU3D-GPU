#pragma once
#include <cuda_runtime.h>

constexpr int REACTION_MAX_SPECIES = 16;
constexpr int REACTION_MAX_REACTIONS = 24;

struct ReactionFlowState
{
    double species_density[REACTION_MAX_SPECIES];
    double mass_fractions[REACTION_MAX_SPECIES];
    double concentration[REACTION_MAX_SPECIES];

    // 各组分密度 + 三个动量 + 总能量密度
    double conserved[REACTION_MAX_SPECIES + 4];

    double density;
    double temperature;

    double u;
    double v;
    double w;
    double internal_energy;

    double mean_molecular_weight;
    double gas_constant;
    double specific_heat_cp;

    double pressure;
    double enthalpy;
    double gamma;
    double sound_speed;
    double mach;
};

cudaError_t launch_thermo_test_cuda(
    const double* temperatures,
    int nt,
    double R,
    int ns,
    const double* coeff0,
    const double* coeff1,
    double* hi,
    double* si,
    double* cp);

void thermo_test_host(
    const double* h_temperatures,
    int nt,
    double R,
    int ns,
    const double* h_coeff0,
    const double* h_coeff1,
    double* h_hi,
    double* h_si,
    double* h_cp);

cudaError_t launch_reaction_rates_test_cuda(
    double T, double R, double Ru, double P0,
    const double* concentration,
    const double* gibbs,
    const double* af,
    const double* bf,
    const double* eaf,
    const double* stoi_f,
    const double* stoi_b,
    const double* efficiencies,
    int ns,
    int nr,
    double* results);

void reaction_rates_test_host(
    double T, double R, double Ru, double P0,
    const double* h_concentration,
    const double* h_gibbs,
    const double* h_af,
    const double* h_bf,
    const double* h_eaf,
    const double* h_stoi_f,
    const double* h_stoi_b,
    const double* h_efficiencies,
    int ns,
    int nr,
    double* h_results);

cudaError_t launch_species_mass_rates_test_cuda(
    int ns,
    int nr,
    const double* molecular_weights,
    const double* stoi_f,
    const double* stoi_b,
    const double* third_body,
    const double* net_rate,
    double* mass_rates);

void species_mass_rates_test_host(
    int ns,
    int nr,
    const double* h_molecular_weights,
    const double* h_stoi_f,
    const double* h_stoi_b,
    const double* h_third_body,
    const double* h_net_rate,
    double* h_mass_rates);

cudaError_t launch_reaction_substep_test_cuda(
    double T, double substep_dt,
    double R, double Ru, double P0,
    const double* original_concentration,
    const double* original_density,
    const double* molecular_weights,
    const double* coeff0,
    const double* coeff1,
    const double* af,
    const double* bf,
    const double* eaf,
    const double* stoi_f,
    const double* stoi_b,
    const double* efficiencies,
    int ns, int nr,
    double* updated_density,
    double* mass_sources);

void reaction_substep_test_host(
    double T, double substep_dt,
    double R, double Ru, double P0,
    const double* h_concentration,
    const double* h_density,
    const double* h_molecular_weights,
    const double* h_coeff0,
    const double* h_coeff1,
    const double* h_af,
    const double* h_bf,
    const double* h_eaf,
    const double* h_stoi_f,
    const double* h_stoi_b,
    const double* h_efficiencies,
    int ns, int nr,
    double* h_updated_density,
    double* h_mass_sources);

cudaError_t launch_composition_temperature_test_cuda(
    const double* species_density,
    const double* molecular_weights,
    const double* species_gas_constants,
    const double* coeff0,
    const double* coeff1,
    int ns,
    double initial_temperature,
    double internal_energy,
    double* mass_fractions,
    double* original_concentration,
    double* state);

void composition_temperature_test_host(
    const double* h_density,
    const double* h_mw,
    const double* h_ri,
    const double* h_coeff0,
    const double* h_coeff1,
    int ns,
    double initial_temperature,
    double internal_energy,
    double* h_yi,
    double* h_mc,
    double* h_state);

cudaError_t launch_full_reaction_substep_test_cuda(
    double T,
    double u, double v, double w,
    double internal_energy,
    double substep_dt,
    double R, double Ru, double P0,
    const double* concentration,
    const double* species_density,
    const double* molecular_weights,
    const double* species_gas_constants,
    const double* coeff0,
    const double* coeff1,
    const double* af,
    const double* bf,
    const double* eaf,
    const double* stoi_f,
    const double* stoi_b,
    const double* efficiencies,
    int ns, int nr,
    ReactionFlowState* output_state,
    double* mass_sources);

void full_reaction_substep_test_host(
    double T,
    double u, double v, double w,
    double internal_energy,
    double substep_dt,
    double R, double Ru, double P0,
    const double* h_concentration,
    const double* h_density,
    const double* h_mw,
    const double* h_ri,
    const double* h_coeff0,
    const double* h_coeff1,
    const double* h_af,
    const double* h_bf,
    const double* h_eaf,
    const double* h_stoi_f,
    const double* h_stoi_b,
    const double* h_efficiencies,
    int ns, int nr,
    ReactionFlowState& output_state,
    double* h_mass_sources);

cudaError_t launch_reaction_cells_cuda(
    ReactionFlowState* states,
    const int* nchem,
    int cell_count,
    double dt,
    double R, double Ru, double P0,
    const double* molecular_weights,
    const double* species_gas_constants,
    const double* coeff0,
    const double* coeff1,
    const double* af,
    const double* bf,
    const double* eaf,
    const double* stoi_f,
    const double* stoi_b,
    const double* efficiencies,
    int ns, int nr,
    double* mass_sources);

void reaction_cells_host(
    ReactionFlowState* h_states,
    int* h_nchem,
    int cell_count,
    double dt,
    double R, double Ru, double P0,
    const double* h_mw,
    const double* h_ri,
    const double* h_coeff0,
    const double* h_coeff1,
    const double* h_af,
    const double* h_bf,
    const double* h_eaf,
    const double* h_stoi_f,
    const double* h_stoi_b,
    const double* h_efficiencies,
    int ns, int nr,
    double* h_mass_sources,
    bool validate_nchem = true);

cudaError_t launch_predict_reaction_cells_cuda(
    const ReactionFlowState* states,
    int* nchem,
    int cell_count,
    double dt,
    double R, double Ru, double P0,
    const double* molecular_weights,
    const double* coeff0,
    const double* coeff1,
    const double* af,
    const double* bf,
    const double* eaf,
    const double* stoi_f,
    const double* stoi_b,
    const double* efficiencies,
    int ns, int nr);
void post_advection_state_host(ReactionFlowState* states, int count, int ns, double R, const double* mw, const double* ri, const double* c0, const double* c1);

// Enable only immediately before the single-rank adaptive reaction path.
void configure_post_advection_chain(bool supported);
bool post_advection_chain_pending();

// Returns the packed host output of the last reaction call, or nullptr.
// Pointer is valid until the next reaction call. validate requests exact scatter checking.
const double* reaction_packed_output(int count, int ns, bool& validate);

void configure_ee_state_chain(bool supported);
// Start a new solve; reuse internal-cell state across steps only in supported modes.
void configure_resident_ee_state(bool supported);
bool ee_state_chain_enabled();
bool ee_state_chain_pending();
void publish_ee_state_rhs(const double* rhs,int nx,int ny,int nz,int nvar,int bc);
void post_advection_from_ee_host(const double* cs,const double* temperature,const double* energy,
    int ni,int nj,int nk,int bc,int ns,double R,
    const double* mw,const double* ri,const double* coeff0,const double* coeff1);

// Explicit host synchronization for output in full-resident mode.
const double* download_resident_output(int count, int ns, int* nchem);
