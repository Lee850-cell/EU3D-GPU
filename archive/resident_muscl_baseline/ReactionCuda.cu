#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include "ReactionCuda.cuh"
#include "WallClock.hpp"
#include <cstdio>
#include <cstdlib>
#include "ThermoConstants.hpp"
#include <vector>


namespace {
struct StateChainContext {
    bool enabled = false;
    bool validate = false;
    ReactionFlowState* pending = nullptr; // Borrowed; owned by state-update storage.
    int count = 0;
    int ns = 0;
};
StateChainContext state_chain;
struct ResidentEeContext {
    bool enabled = false;
    bool ready = false;
    ReactionFlowState* states = nullptr; // Borrowed from EeStateStorage.
    int ni = 0, nj = 0, nk = 0, bc = 0, ns = 0;
};
ResidentEeContext resident_ee;
bool chain_env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}
void chain_fail(const char* message) {
    std::fprintf(stderr, "State chain error: %s\n", message);
    std::exit(EXIT_FAILURE);
}
}
void configure_resident_ee_state(bool supported) {
    resident_ee = ResidentEeContext{};
    const bool requested = chain_env_enabled("EU3D_RESIDENT_EE");
    if (requested && (!supported || !chain_env_enabled("EU3D_CHAIN_STATE")
                     || !chain_env_enabled("EU3D_EE_STATE")
                     || chain_env_enabled("EU3D_VALIDATE_STATE")))
        chain_fail("resident EE requires single-rank fixed-dt GPU EE/MUSCL_2 adaptive state chain");
    resident_ee.enabled = requested;
}
void configure_post_advection_chain(bool supported) {
    if (state_chain.pending) chain_fail("previous device state was not consumed");
    const bool requested = chain_env_enabled("EU3D_CHAIN_STATE");
    const bool validate = chain_env_enabled("EU3D_VALIDATE_CHAIN");
    if (validate && !requested) chain_fail("EU3D_VALIDATE_CHAIN requires EU3D_CHAIN_STATE=1");
    if (requested && (!supported || chain_env_enabled("EU3D_VALIDATE_STATE")))
        chain_fail("requires local adaptive GPU mode (DLB disabled for multiple ranks) with CPU/state validation disabled");
    state_chain.enabled = requested;
    state_chain.validate = validate;
}
bool post_advection_chain_pending() {
    return state_chain.pending != nullptr;
}


namespace {
struct EeStateContext {
    bool enabled=false, validate=false;
    const double* rhs=nullptr; // Borrowed from TimeAdv, consumed before the next EE call.
    int nx=0,ny=0,nz=0,nvar=0,bc=0;
};
EeStateContext ee_state;
}
void configure_ee_state_chain(bool supported){
    if(ee_state.rhs) chain_fail("previous EE RHS was not consumed");
    const bool requested=chain_env_enabled("EU3D_EE_STATE");
    const bool validate=chain_env_enabled("EU3D_VALIDATE_EE_STATE");
    if(validate && !requested) chain_fail("EE validation requires EU3D_EE_STATE=1");
    if(requested && (!supported || !chain_env_enabled("EU3D_CHAIN_STATE")
                    || chain_env_enabled("EU3D_VALIDATE_STATE")))
        chain_fail("EE bridge requires local GPU EE/MUSCL_2/adaptive chain mode with CPU/state validation disabled");
    ee_state.enabled=requested;ee_state.validate=validate;
}
bool ee_state_chain_enabled(){return ee_state.enabled;}
bool ee_state_chain_pending(){return ee_state.rhs!=nullptr;}
void publish_ee_state_rhs(const double* rhs,int nx,int ny,int nz,int nvar,int bc){
    if(!ee_state.enabled || ee_state.rhs || !rhs) chain_fail("invalid EE RHS publication");
    ee_state.rhs=rhs;ee_state.nx=nx;ee_state.ny=ny;ee_state.nz=nz;
    ee_state.nvar=nvar;ee_state.bc=bc;
}

struct ReactionBuffers
{
    double* parameters[10] = {};

    ReactionFlowState* states = nullptr;
    int* nchem = nullptr;
    double* mass_sources = nullptr;

    // 当前缓冲区对应的尺寸
    int cell_count = 0;
    int ns = 0;
    int nr = 0;

    ReactionBuffers() = default;

    // 禁止复制，避免重复释放同一块显存
    ReactionBuffers(const ReactionBuffers&) = delete;
    ReactionBuffers& operator=(const ReactionBuffers&) = delete;

    void release()
    {
        for (int n = 0; n < 10; ++n)
        {
            if (parameters[n])
                cudaFree(parameters[n]);

            parameters[n] = nullptr;
        }

        if (states) cudaFree(states);
        if (nchem) cudaFree(nchem);
        if (mass_sources) cudaFree(mass_sources);

        states = nullptr;
        nchem = nullptr;
        mass_sources = nullptr;

        cell_count = 0;
        ns = 0;
        nr = 0;
    }

    ~ReactionBuffers()
    {
        release();
    }
};

static ReactionBuffers& get_reaction_buffers()
{
    static ReactionBuffers buffers;
    return buffers;
}

struct ReactionCellData
{
    double concentration[REACTION_MAX_SPECIES];
    double gibbs[REACTION_MAX_SPECIES];

    double k_forward[REACTION_MAX_REACTIONS];
    double k_reverse[REACTION_MAX_REACTIONS];

    double product_forward[REACTION_MAX_REACTIONS];
    double product_reverse[REACTION_MAX_REACTIONS];

    double third_body[REACTION_MAX_REACTIONS];
    double net_rate[REACTION_MAX_REACTIONS];
};

__device__ __forceinline__
double get_hi_cuda(
    double T,
    double R,
    int species,
    int ns,
    const double* coeff0,
    const double* coeff1)
{
    const double* coeff =
        T < THERMO_SWITCH_T
        ? coeff0
        : coeff1;

    double a0 = coeff[0 * ns + species];
    double a1 = coeff[1 * ns + species];
    double a2 = coeff[2 * ns + species];
    double a3 = coeff[3 * ns + species];
    double a4 = coeff[4 * ns + species];
    double a5 = coeff[5 * ns + species];
    double a6 = coeff[6 * ns + species];
    double a7 = coeff[7 * ns + species];

    const double invT = 1.0 / T;
    const double T2 = T * T;
    const double T3 = T2 * T;
    const double T4 = T2 * T2;
    const double T5 = T4 * T;

    return R * (
        -a0 * invT
        + a1 * log(T)
        + a2 * T
        + a3 * T2 / 2.0
        + a4 * T3 / 3.0
        + a5 * T4 / 4.0
        + a6 * T5 / 5.0
        + a7);
}

__device__ __forceinline__
double get_si_cuda(
    double T,
    double R,
    int species,
    int ns,
    const double* coeff0,
    const double* coeff1)
{
    const double* coeff =
        T < THERMO_SWITCH_T
        ? coeff0
        : coeff1;


    double a0 = coeff[0 * ns + species];
    double a1 = coeff[1 * ns + species];
    double a2 = coeff[2 * ns + species];
    double a3 = coeff[3 * ns + species];
    double a4 = coeff[4 * ns + species];
    double a5 = coeff[5 * ns + species];
    double a6 = coeff[6 * ns + species];
    double a8 = coeff[8 * ns + species];

    return R * (
        -a0 * pow(T, -2.0) / 2.0
        - a1 * pow(T, -1.0)
        + a2 * log(T)
        + a3 * T
        + a4 * pow(T, 2.0) / 2.0
        + a5 * pow(T, 3.0) / 3.0
        + a6 * pow(T, 4.0) / 4.0
        + a8);
}

__device__ __forceinline__
double get_cp_cuda(
    double T,
    double R,
    int species,
    int ns,
    const double* coeff0,
    const double* coeff1)
{
    const double* coeff =
        T < THERMO_SWITCH_T
        ? coeff0
        : coeff1;


    double a0 = coeff[0 * ns + species];
    double a1 = coeff[1 * ns + species];
    double a2 = coeff[2 * ns + species];
    double a3 = coeff[3 * ns + species];
    double a4 = coeff[4 * ns + species];
    double a5 = coeff[5 * ns + species];
    double a6 = coeff[6 * ns + species];

    return R * (
        a0 * pow(T, -2.0)
        + a1 * pow(T, -1.0)
        + a2
        + a3 * T
        + a4 * pow(T, 2.0)
        + a5 * pow(T, 3.0)
        + a6 * pow(T, 4.0));
}

__device__ __forceinline__
double forward_rate_cuda(
    double T,
    double A,
    double b,
    double Ea,
    double Ru)
{
    return A * pow(T, b) * exp(-Ea / (Ru * T));
}

__device__ __forceinline__
double concentration_product_cuda(
    const double* concentration,
    const double* stoich,
    int ns,
    int nr,
    int reaction)
{
    double product = 1.0;

    for (int s = 0; s < ns; ++s)
    {
        double exponent = stoich[s * nr + reaction];
        product *= pow(concentration[s], exponent);
    }

    return product;
}

__device__ __forceinline__
double reverse_rate_cuda(
    double T,
    double R,
    double P0,
    double forward_rate,
    const double* gibbs,
    const double* stoi_f,
    const double* stoi_b,
    int ns,
    int nr,
    int reaction)
{
    double delta_n = 0.0;
    double delta_g = 0.0;

    for (int s = 0; s < ns; ++s)
    {
        int idx = s * nr + reaction;

        double difference = stoi_b[idx] - stoi_f[idx];

        delta_n += difference;
        delta_g += difference * gibbs[s];
    }

    double kp =
        exp(-delta_g / (R * T))
        * pow(P0 * 1e-6, delta_n);

    double kc = kp * pow(R * T, -delta_n);

    return forward_rate / kc;
}   

__device__ __forceinline__
double third_body_cuda(
    const double* concentration,
    const double* efficiencies,
    int ns,
    int nr,
    int reaction)
{
    double factor = 0.0;

    for (int s = 0; s < ns; ++s)
    {
        int idx = s * nr + reaction;

        factor += concentration[s] * efficiencies[idx];
    }

    // 沿用原 CPU 程序的处理方式
    if (factor == 0.0)
        factor = 1.0;

    return factor;
}

__device__ __forceinline__
void evaluate_reaction_cuda(
    double T,
    double R,
    double Ru,
    double P0,
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
    int reaction,
    double& k_forward,
    double& k_reverse,
    double& product_forward,
    double& product_reverse,
    double& third_body,
    double& net_rate)
{
    k_forward = forward_rate_cuda(
        T, af[reaction], bf[reaction], eaf[reaction], Ru);

    k_reverse = reverse_rate_cuda(
        T, R, P0, k_forward,
        gibbs, stoi_f, stoi_b,
        ns, nr, reaction);

    product_forward = concentration_product_cuda(
        concentration, stoi_f, ns, nr, reaction);

    product_reverse = concentration_product_cuda(
        concentration, stoi_b, ns, nr, reaction);

    third_body = third_body_cuda(
        concentration, efficiencies, ns, nr, reaction);

    net_rate = k_forward * product_forward
             - k_reverse * product_reverse;
}

__device__ __forceinline__
double species_mass_rate_cuda(
    int species,
    int nr,
    double molecular_weight,
    const double* stoi_f,
    const double* stoi_b,
    const double* third_body,
    const double* net_rate)
{
    double sum = 0.0;

    for (int r = 0; r < nr; ++r)
    {
        int idx = species * nr + r;

        sum += (stoi_b[idx] - stoi_f[idx])
             * third_body[r]
             * net_rate[r];
    }

    return molecular_weight * sum;
}

__device__ __forceinline__
int predict_reaction_substeps_cuda(
    double dt,
    const ReactionFlowState& state,
    const ReactionCellData& cell,
    const double* molecular_weights,
    const double* stoi_f,
    const double* stoi_b,
    int ns,
    int nr)
{
    double allowed_dt = dt;

    for (int s = 0; s < ns; ++s)
    {
        // 沿用 CPU：忽略质量分数很小的组分
        if (state.mass_fractions[s] >= 1e-6)
        {
            double wi = species_mass_rate_cuda(
                s, nr,
                molecular_weights[s],
                stoi_f, stoi_b,
                cell.third_body, cell.net_rate);

            if (wi != 0.0)
            {
                double density =
                    state.species_density[s] * 1e-3;

                double candidate = fabs(-density / wi);

                if (candidate < allowed_dt)
                    allowed_dt = candidate;
            }
        }
    }

    // 非法结果返回 0，后续由调用方检查并报错
    if (!(allowed_dt > 0.0))
        return 0;

    double count = ceil(dt / allowed_dt);

    if (!isfinite(count) || count < 1.0 ||
        count > 2147483647.0)
        return 0;

    return static_cast<int>(count);
}

__device__ __forceinline__
void species_pq_cuda(
    int species,
    int nr,
    double concentration,
    double molecular_weight,
    const double* stoi_f,
    const double* stoi_b,
    const double* k_forward,
    const double* k_reverse,
    const double* product_forward,
    const double* product_reverse,
    const double* third_body,
    double& loss_coefficient,
    double& production)
{
    double p = 0.0;
    double q = 0.0;

    for (int r = 0; r < nr; ++r)
    {
        int idx = species * nr + r;

        double forward = k_forward[r] * product_forward[r];
        double reverse = k_reverse[r] * product_reverse[r];

        q += (stoi_b[idx] * forward
            + stoi_f[idx] * reverse) * third_body[r];

        p += (stoi_b[idx] * reverse
            + stoi_f[idx] * forward) * third_body[r];
    }

    production = molecular_weight * q;

    loss_coefficient =
        concentration == 0.0 ? 0.0 : p / concentration;
}

__device__ __forceinline__
double trapezoid_species_update_cuda(
    double density,
    double dt,
    double loss_coefficient,
    double production)
{
    double half_dt_p = 0.5 * dt * loss_coefficient;

    return (
        (1.0 - half_dt_p) * density
        + dt * production
    ) / (1.0 + half_dt_p);
}

__device__ __forceinline__
void prepare_reaction_cell_cuda(
    double T,
    double R,
    double Ru,
    double P0,
    const double* original_concentration,
    const double* coeff0,
    const double* coeff1,
    const double* af,
    const double* bf,
    const double* eaf,
    const double* stoi_f,
    const double* stoi_b,
    const double* efficiencies,
    int ns,
    int nr,
    ReactionCellData& cell)
{
    for (int s = 0; s < ns; ++s)
    {
        // 与 CPU 的 Mc_temp → Mc 换算一致
        cell.concentration[s] =
            original_concentration[s] * 1e-6;

        double hi = get_hi_cuda(
            T, R, s, ns, coeff0, coeff1);

        double si = get_si_cuda(
            T, R, s, ns, coeff0, coeff1);

        cell.gibbs[s] = hi - T * si;
    }

    for (int r = 0; r < nr; ++r)
    {
        evaluate_reaction_cuda(
            T, R, Ru, P0,
            cell.concentration, cell.gibbs,
            af, bf, eaf,
            stoi_f, stoi_b, efficiencies,
            ns, nr, r,
            cell.k_forward[r],
            cell.k_reverse[r],
            cell.product_forward[r],
            cell.product_reverse[r],
            cell.third_body[r],
            cell.net_rate[r]);
    }
}

__device__ __forceinline__
void update_reaction_cell_cuda(
    double substep_dt,
    const double* original_density,
    const double* molecular_weights,
    const double* stoi_f,
    const double* stoi_b,
    int ns,
    int nr,
    const ReactionCellData& cell,
    double* updated_density,
    double* mass_sources)
{
    for (int s = 0; s < ns; ++s)
    {
        double p;
        double q;

        species_pq_cuda(
            s, nr,
            cell.concentration[s],
            molecular_weights[s],
            stoi_f, stoi_b,
            cell.k_forward, cell.k_reverse,
            cell.product_forward, cell.product_reverse,
            cell.third_body,
            p, q);

        double density = original_density[s] * 1e-3;

        updated_density[s] =
            trapezoid_species_update_cuda(
                density, substep_dt, p, q) * 1e3;

        double wi = species_mass_rate_cuda(
            s, nr,
            molecular_weights[s],
            stoi_f, stoi_b,
            cell.third_body, cell.net_rate);

        mass_sources[s] = wi * 1e3;
    }
}


__device__ __forceinline__
void rebuild_composition_cuda(
    const double* species_density,
    const double* molecular_weights,
    int ns,
    double& total_density,
    double* mass_fractions,
    double* original_concentration)
{
    total_density = 0.0;

    for (int s = 0; s < ns; ++s)
        total_density += species_density[s];

    for (int s = 0; s < ns; ++s)
    {
        mass_fractions[s] =
            species_density[s] / total_density;

        original_concentration[s] =
            species_density[s] / molecular_weights[s] * 1000.0;
    }
}

__device__ __forceinline__
double recover_temperature_cuda(
    double initial_temperature,
    double internal_energy,
    const double* mass_fractions,
    const double* species_gas_constants,
    const double* coeff0,
    const double* coeff1,
    int ns, bool trace=false)
{
    double temperature = initial_temperature;
    if(trace){
        printf("TEMPTRACE GPU input T=%.17g E=%.17g\n",initial_temperature,internal_energy);
        for(int n=0;n<ns;++n) printf("TEMPTRACE GPU Yi[%d]=%.17g\n",n,mass_fractions[n]);
    }

    for (int iteration = 0; iteration < 10; ++iteration)
    {
        double old_temperature = temperature;

        double mixture_h = 0.0;
        double mixture_cp = 0.0;
        double mixture_r = 0.0;

        for (int s = 0; s < ns; ++s)
        {
            double yi = mass_fractions[s];
            double ri = species_gas_constants[s];

            mixture_h += yi * get_hi_cuda(
                old_temperature, ri, s, ns, coeff0, coeff1);

            mixture_cp += yi * get_cp_cuda(
                old_temperature, ri, s, ns, coeff0, coeff1);

            mixture_r += yi * ri;
            if(trace) printf("TEMPPART GPU iter=%d species=%d h=%.17g cp=%.17g ri=%.17g hsum=%.17g cpsum=%.17g rsum=%.17g\n",iteration,s,get_hi_cuda(old_temperature,ri,s,ns,coeff0,coeff1),get_cp_cuda(old_temperature,ri,s,ns,coeff0,coeff1),ri,mixture_h,mixture_cp,mixture_r);
        }

        double residual =
            (mixture_h - internal_energy * 1e-3)
            - mixture_r * old_temperature;

        if(trace && ns>8){
            double x=old_temperature; double r=species_gas_constants[8]; double a[8]; const double* cf=x<THERMO_SWITCH_T ? coeff0 : coeff1; for(int n=0;n<8;++n) a[n]=cf[n*ns+8];
            double v[7]={pow(x,-1.0),log(x),x,pow(x,2.0),pow(x,3.0),pow(x,4.0),pow(x,5.0)};
            double q[8]={-a[0]*v[0],a[1]*v[1],a[2]*v[2],a[3]*v[3]/2.0,a[4]*v[4]/3.0,a[5]*v[5]/4.0,a[6]*v[6]/5.0,a[7]};
            printf("TEMPHI GPU iter=%d T=%.17g R=%.17g\n",iteration,x,r);
            for(int n=0;n<7;++n) printf("TEMPHI GPU iter=%d primitive=%d value=%.17g\n",iteration,n,v[n]);
            double running=q[0];
            for(int n=0;n<8;++n){
                if(n) running+=q[n];
                printf("TEMPHI GPU iter=%d term=%d coeff=%.17g value=%.17g prefix=%.17g\n",iteration,n,a[n],q[n],running);
            }
            printf("TEMPHI GPU iter=%d explicit_h=%.17g\n",iteration,r*running);
        }
        if(trace) printf("TEMPPART GPU iter=%d h=%.17g cp=%.17g r=%.17g e_scaled=%.17g h_minus_e=%.17g r_times_T=%.17g residual=%.17g\n",iteration,mixture_h,mixture_cp,mixture_r,internal_energy*1e-3,mixture_h-internal_energy*1e-3,mixture_r*old_temperature,residual);
        double derivative = mixture_cp - mixture_r;

        temperature =
            old_temperature - residual / derivative;

        if(trace) printf("TEMPTRACE GPU iter=%d T=%.17g low=%d residual=%.17g derivative=%.17g next=%.17g stop=%d\n",iteration,old_temperature,old_temperature<THERMO_SWITCH_T,residual,derivative,temperature,fabs(temperature-old_temperature)<1e-6);
        if (fabs(temperature - old_temperature) < 1e-6)
            break;
    }

    return temperature;
}

__device__ __forceinline__
void mixture_properties_cuda(
    double T,
    double R,
    const double* mass_fractions,
    const double* molecular_weights,
    const double* species_gas_constants,
    const double* coeff0,
    const double* coeff1,
    int ns,
    double& mean_molecular_weight,
    double& gas_constant,
    double& specific_heat_cp)
{
    double inverse_weight = 0.0;
    double mixture_cp = 0.0;

    for (int s = 0; s < ns; ++s)
    {
        double yi = mass_fractions[s];

        inverse_weight += yi / molecular_weights[s];

        mixture_cp += yi * get_cp_cuda(
            T, species_gas_constants[s],
            s, ns, coeff0, coeff1);
    }

    mean_molecular_weight = 1.0 / inverse_weight;
    gas_constant = R * 1000.0 / mean_molecular_weight;
    specific_heat_cp = mixture_cp * 1000.0;
}

__device__ __forceinline__
void derived_flow_properties_cuda(
    double temperature,
    double density,
    double internal_energy,
    double u,
    double v,
    double w,
    double gas_constant,
    double specific_heat_cp,
    double& pressure,
    double& enthalpy,
    double& gamma,
    double& sound_speed,
    double& mach)
{
    pressure = temperature * density * gas_constant;

    enthalpy = pressure / density + internal_energy;

    gamma = specific_heat_cp
          / (specific_heat_cp - gas_constant);

    sound_speed = sqrt(gamma * pressure / density);

    double speed = sqrt(u * u + v * v + w * w);

    mach = speed / sound_speed;
}

__device__ __forceinline__
void rebuild_reaction_state_cuda(
    const double* updated_density,
    double old_temperature,
    double old_u,
    double old_v,
    double old_w,
    double old_internal_energy,
    double R,
    const double* molecular_weights,
    const double* species_gas_constants,
    const double* coeff0,
    const double* coeff1,
    int ns,
    ReactionFlowState& state)
{
    // 1. 保存反应后的组分密度
    for (int s = 0; s < ns; ++s)
    {
        state.species_density[s] = updated_density[s];
        state.conserved[s] = updated_density[s];
    }

    // 2. 重建总密度、质量分数和浓度
    rebuild_composition_cuda(
        state.species_density,
        molecular_weights,
        ns,
        state.density,
        state.mass_fractions,
        state.concentration);

    // 3. 对应 CPU Explicit() 中的守恒量重建
    state.conserved[ns]     = old_u * state.density;
    state.conserved[ns + 1] = old_v * state.density;
    state.conserved[ns + 2] = old_w * state.density;

    state.conserved[ns + 3] =
        state.density * old_internal_energy
        + 0.5 * (old_u * old_u
               + old_v * old_v
               + old_w * old_w) * state.density;

    // 4. CPU 先用旧的比内能恢复温度
    state.temperature = recover_temperature_cuda(
        old_temperature,
        old_internal_energy,
        state.mass_fractions,
        species_gas_constants,
        coeff0, coeff1,
        ns);

    // 5. 再从守恒量恢复速度、比内能
    state.u = state.conserved[ns] / state.density;
    state.v = state.conserved[ns + 1] / state.density;
    state.w = state.conserved[ns + 2] / state.density;

    state.internal_energy =
        (state.conserved[ns + 3]
         - 0.5 * state.density
         * (state.u * state.u
          + state.v * state.v
          + state.w * state.w)) / state.density;

    // 6. 更新混合物性质
    mixture_properties_cuda(
        state.temperature, R,
        state.mass_fractions,
        molecular_weights,
        species_gas_constants,
        coeff0, coeff1,
        ns,
        state.mean_molecular_weight,
        state.gas_constant,
        state.specific_heat_cp);

    // 7. 更新压力、焓、声速等
    derived_flow_properties_cuda(
        state.temperature,
        state.density,
        state.internal_energy,
        state.u, state.v, state.w,
        state.gas_constant,
        state.specific_heat_cp,
        state.pressure,
        state.enthalpy,
        state.gamma,
        state.sound_speed,
        state.mach);
}

__device__ __forceinline__
void advance_reaction_substep_cuda(
    double substep_dt,
    double R,
    double Ru,
    double P0,
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
    int ns,
    int nr,
    ReactionFlowState& state,
    double* mass_sources)
{
    ReactionCellData cell;
    double updated_density[REACTION_MAX_SPECIES];

    // 当前状态 → 反应速率
    prepare_reaction_cell_cuda(
        state.temperature, R, Ru, P0,
        state.concentration,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr, cell);

    // 反应速率 → 新组分密度
    update_reaction_cell_cuda(
        substep_dt,
        state.species_density,
        molecular_weights,
        stoi_f, stoi_b,
        ns, nr, cell,
        updated_density,
        mass_sources);

    // 新组分密度 → 完整的新状态
    rebuild_reaction_state_cuda(
        updated_density,
        state.temperature,
        state.u, state.v, state.w,
        state.internal_energy,
        R,
        molecular_weights,
        species_gas_constants,
        coeff0, coeff1,
        ns,
        state);
}

__global__ void reaction_cells_kernel(
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
    int ns,
    int nr,
    double* mass_sources)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell_idx >= cell_count)
        return;

    ReactionFlowState state = states[cell_idx];

    int substeps = nchem[cell_idx];
    double substep_dt = dt / substeps;

    double* cell_sources =
        mass_sources + static_cast<size_t>(cell_idx) * ns;

    for (int step = 0; step < substeps; ++step)
    {
        advance_reaction_substep_cuda(
            substep_dt,
            R, Ru, P0,
            molecular_weights,
            species_gas_constants,
            coeff0, coeff1,
            af, bf, eaf,
            stoi_f, stoi_b, efficiencies,
            ns, nr,
            state,
            cell_sources);
    }

    states[cell_idx] = state;
}

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
    double* mass_sources)
{
    if (cell_count <= 0 ||
        ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS)
    {
        return cudaErrorInvalidValue;
    }

    int threads = 128;
    int blocks = (cell_count - 1) / threads + 1;

    reaction_cells_kernel<<<blocks, threads>>>(
        states, nchem, cell_count,
        dt, R, Ru, P0,
        molecular_weights,
        species_gas_constants,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr,
        mass_sources);

    return cudaGetLastError();
}

__global__ void reaction_cells_fused_kernel(
    ReactionFlowState* states,
    int* nchem,
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
    int ns,
    int nr,
    double* mass_sources)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= cell_count)
        return;

    ReactionFlowState state = states[cell_idx];
    ReactionCellData cell;
    double updated_density[REACTION_MAX_SPECIES];

    // 预测与首个子步共享同一初始状态，只计算一次反应速率。
    prepare_reaction_cell_cuda(
        state.temperature, R, Ru, P0,
        state.concentration, coeff0, coeff1,
        af, bf, eaf, stoi_f, stoi_b, efficiencies,
        ns, nr, cell);

    int substeps = predict_reaction_substeps_cuda(
        dt, state, cell, molecular_weights,
        stoi_f, stoi_b, ns, nr);
    nchem[cell_idx] = substeps;

    // 主机在使用输出前检查 nchem；非法子步数不参与积分。
    if (substeps < 1)
        return;

    double substep_dt = dt / substeps;
    double* cell_sources =
        mass_sources + static_cast<size_t>(cell_idx) * ns;

    for (int step = 0; step < substeps; ++step)
    {
        if (step > 0)
        {
            // 后续子步的温度和浓度已改变，必须重新计算速率。
            prepare_reaction_cell_cuda(
                state.temperature, R, Ru, P0,
                state.concentration, coeff0, coeff1,
                af, bf, eaf, stoi_f, stoi_b, efficiencies,
                ns, nr, cell);
        }

        update_reaction_cell_cuda(
            substep_dt, state.species_density,
            molecular_weights, stoi_f, stoi_b,
            ns, nr, cell, updated_density, cell_sources);

        rebuild_reaction_state_cuda(
            updated_density, state.temperature,
            state.u, state.v, state.w, state.internal_energy,
            R, molecular_weights, species_gas_constants,
            coeff0, coeff1, ns, state);
    }

    states[cell_idx] = state;
}

static cudaError_t launch_reaction_cells_fused_cuda(
    ReactionFlowState* states,
    int* nchem,
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
    double* mass_sources)
{
    if (cell_count <= 0 ||
        ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS ||
        !std::isfinite(dt) || dt <= 0.0)
    {
        return cudaErrorInvalidValue;
    }

    int threads = 128;
    int blocks = (cell_count - 1) / threads + 1;

    reaction_cells_fused_kernel<<<blocks, threads>>>(
        states, nchem, cell_count,
        dt, R, Ru, P0,
        molecular_weights,
        species_gas_constants,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr,
        mass_sources);

    return cudaGetLastError();
}

__global__ void full_reaction_substep_test_kernel(
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
    double* mass_sources)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    ReactionFlowState state{};

    // 初始化这个网格点在反应前的状态
    state.temperature = T;
    state.u = u;
    state.v = v;
    state.w = w;
    state.internal_energy = internal_energy;

    for (int s = 0; s < ns; ++s)
    {
        state.concentration[s] = concentration[s];
        state.species_density[s] = species_density[s];
    }

    // 完成一次反应计算及状态更新
    advance_reaction_substep_cuda(
        substep_dt,
        R, Ru, P0,
        molecular_weights,
        species_gas_constants,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr,
        state,
        mass_sources);

    *output_state = state;
}

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
    double* mass_sources)
{
    if (ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS)
    {
        return cudaErrorInvalidValue;
    }

    full_reaction_substep_test_kernel<<<1, 1>>>(
        T, u, v, w,
        internal_energy, substep_dt,
        R, Ru, P0,
        concentration, species_density,
        molecular_weights, species_gas_constants,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr,
        output_state, mass_sources);

    return cudaGetLastError();
}

__global__ void reaction_substep_test_kernel(
    double T,
    double substep_dt,
    double R,
    double Ru,
    double P0,
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
    int ns,
    int nr,
    double* updated_density,
    double* mass_sources)
{
    // 这个测试只用一个线程处理一个网格点
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    ReactionCellData cell;

    prepare_reaction_cell_cuda(
        T, R, Ru, P0,
        original_concentration,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr, cell);

    update_reaction_cell_cuda(
        substep_dt,
        original_density,
        molecular_weights,
        stoi_f, stoi_b,
        ns, nr, cell,
        updated_density,
        mass_sources);
}

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
    double* mass_sources)
{
    if (ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS)
    {
        return cudaErrorInvalidValue;
    }

    reaction_substep_test_kernel<<<1, 1>>>(
        T, substep_dt, R, Ru, P0,
        original_concentration,
        original_density,
        molecular_weights,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr,
        updated_density, mass_sources);

    return cudaGetLastError();
}

__global__ void species_mass_rates_test_kernel(
    int ns,
    int nr,
    const double* molecular_weights,
    const double* stoi_f,
    const double* stoi_b,
    const double* third_body,
    const double* net_rate,
    double* mass_rates)
{
    int species = blockIdx.x * blockDim.x + threadIdx.x;

    if (species >= ns)
        return;

    mass_rates[species] = species_mass_rate_cuda(
        species,
        nr,
        molecular_weights[species],
        stoi_f,
        stoi_b,
        third_body,
        net_rate);
}

cudaError_t launch_species_mass_rates_test_cuda(
    int ns,
    int nr,
    const double* molecular_weights,
    const double* stoi_f,
    const double* stoi_b,
    const double* third_body,
    const double* net_rate,
    double* mass_rates)
{
    int threads = 128;
    int blocks = (ns + threads - 1) / threads;

    species_mass_rates_test_kernel<<<blocks, threads>>>(
        ns, nr,
        molecular_weights,
        stoi_f, stoi_b,
        third_body, net_rate,
        mass_rates);

    return cudaGetLastError();
}

__global__ void reaction_rates_test_kernel(
    double T,
    double R,
    double Ru,
    double P0,
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
    double* results)
{
    int reaction = blockIdx.x * blockDim.x + threadIdx.x;

    if (reaction >= nr)
        return;

    double k_forward;
    double k_reverse;
    double product_forward;
    double product_reverse;
    double third_body;
    double net_rate;

    evaluate_reaction_cuda(
        T, R, Ru, P0,
        concentration, gibbs,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr, reaction,
        k_forward, k_reverse,
        product_forward, product_reverse,
        third_body, net_rate);

    int base = reaction * 6;

    results[base + 0] = k_forward;
    results[base + 1] = k_reverse;
    results[base + 2] = product_forward;
    results[base + 3] = product_reverse;
    results[base + 4] = third_body;
    results[base + 5] = net_rate;
}

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
    double* results)
{
    int threads = 128;
    int blocks = (nr + threads - 1) / threads;

    reaction_rates_test_kernel<<<blocks, threads>>>(
        T, R, Ru, P0,
        concentration, gibbs,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr, results);

    return cudaGetLastError();
}


__global__ void thermo_test_kernel(
    const double* temperatures,
    int nt,
    double R,
    int ns,
    const double* coeff0,
    const double* coeff1,
    double* hi,
    double* si,
    double* cp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nt * ns)
        return;

    int t = idx / ns;
    int species = idx % ns;

    double T = temperatures[t];

    hi[idx] = get_hi_cuda(T, R, species, ns, coeff0, coeff1);
    si[idx] = get_si_cuda(T, R, species, ns, coeff0, coeff1);
    cp[idx] = get_cp_cuda(T, R, species, ns, coeff0, coeff1);
}

cudaError_t launch_thermo_test_cuda(
    const double* temperatures,
    int nt,
    double R,
    int ns,
    const double* coeff0,
    const double* coeff1,
    double* hi,
    double* si,
    double* cp)
{
    int threads = 128;
    int blocks = (nt * ns + threads - 1) / threads;

    thermo_test_kernel<<<blocks, threads>>>(
        temperatures, nt, R, ns,
        coeff0, coeff1, hi, si, cp);

    return cudaGetLastError();
}

void thermo_test_host(
    const double* h_temperatures,
    int nt,
    double R,
    int ns,
    const double* h_coeff0,
    const double* h_coeff1,
    double* h_hi,
    double* h_si,
    double* h_cp)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "Thermo CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    size_t temperature_bytes =
        static_cast<size_t>(nt) * sizeof(double);

    size_t coeff_bytes =
        static_cast<size_t>(9) * ns * sizeof(double);

    size_t result_bytes =
        static_cast<size_t>(nt) * ns * sizeof(double);

    double* d_temperatures = nullptr;
    double* d_coeff0 = nullptr;
    double* d_coeff1 = nullptr;
    double* d_hi = nullptr;
    double* d_si = nullptr;
    double* d_cp = nullptr;

    check(cudaMalloc(&d_temperatures, temperature_bytes));
    check(cudaMalloc(&d_coeff0, coeff_bytes));
    check(cudaMalloc(&d_coeff1, coeff_bytes));
    check(cudaMalloc(&d_hi, result_bytes));
    check(cudaMalloc(&d_si, result_bytes));
    check(cudaMalloc(&d_cp, result_bytes));

    check(cudaMemcpy(
        d_temperatures, h_temperatures, temperature_bytes,
        cudaMemcpyHostToDevice));

    check(cudaMemcpy(
        d_coeff0, h_coeff0, coeff_bytes,
        cudaMemcpyHostToDevice));

    check(cudaMemcpy(
        d_coeff1, h_coeff1, coeff_bytes,
        cudaMemcpyHostToDevice));

    check(launch_thermo_test_cuda(
        d_temperatures, nt, R, ns,
        d_coeff0, d_coeff1,
        d_hi, d_si, d_cp));

    check(cudaDeviceSynchronize());

    check(cudaMemcpy(
        h_hi, d_hi, result_bytes,
        cudaMemcpyDeviceToHost));

    check(cudaMemcpy(
        h_si, d_si, result_bytes,
        cudaMemcpyDeviceToHost));

    check(cudaMemcpy(
        h_cp, d_cp, result_bytes,
        cudaMemcpyDeviceToHost));

    check(cudaFree(d_temperatures));
    check(cudaFree(d_coeff0));
    check(cudaFree(d_coeff1));
    check(cudaFree(d_hi));
    check(cudaFree(d_si));
    check(cudaFree(d_cp));
}

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
    double* h_results)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "Reaction rates CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    size_t species_bytes =
        static_cast<size_t>(ns) * sizeof(double);

    size_t reaction_bytes =
        static_cast<size_t>(nr) * sizeof(double);

    size_t table_bytes =
        static_cast<size_t>(ns) * nr * sizeof(double);

    size_t result_bytes =
        static_cast<size_t>(nr) * 6 * sizeof(double);

    const double* host_inputs[8] = {
        h_concentration,
        h_gibbs,
        h_af,
        h_bf,
        h_eaf,
        h_stoi_f,
        h_stoi_b,
        h_efficiencies
    };

    size_t input_bytes[8] = {
        species_bytes,
        species_bytes,
        reaction_bytes,
        reaction_bytes,
        reaction_bytes,
        table_bytes,
        table_bytes,
        table_bytes
    };

    double* device_inputs[8] = {};

    for (int n = 0; n < 8; ++n)
    {
        check(cudaMalloc(&device_inputs[n], input_bytes[n]));

        check(cudaMemcpy(
            device_inputs[n], host_inputs[n], input_bytes[n],
            cudaMemcpyHostToDevice));
    }

    double* d_results = nullptr;
    check(cudaMalloc(&d_results, result_bytes));

    check(launch_reaction_rates_test_cuda(
        T, R, Ru, P0,
        device_inputs[0], device_inputs[1],
        device_inputs[2], device_inputs[3], device_inputs[4],
        device_inputs[5], device_inputs[6], device_inputs[7],
        ns, nr, d_results));

    check(cudaDeviceSynchronize());

    check(cudaMemcpy(
        h_results, d_results, result_bytes,
        cudaMemcpyDeviceToHost));

    for (int n = 0; n < 8; ++n)
        check(cudaFree(device_inputs[n]));

    check(cudaFree(d_results));
}

void species_mass_rates_test_host(
    int ns,
    int nr,
    const double* h_molecular_weights,
    const double* h_stoi_f,
    const double* h_stoi_b,
    const double* h_third_body,
    const double* h_net_rate,
    double* h_mass_rates)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "Mass rates CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    size_t species_bytes =
        static_cast<size_t>(ns) * sizeof(double);

    size_t reaction_bytes =
        static_cast<size_t>(nr) * sizeof(double);

    size_t table_bytes =
        static_cast<size_t>(ns) * nr * sizeof(double);

    const double* host_inputs[5] = {
        h_molecular_weights,
        h_stoi_f,
        h_stoi_b,
        h_third_body,
        h_net_rate
    };

    size_t input_bytes[5] = {
        species_bytes,
        table_bytes,
        table_bytes,
        reaction_bytes,
        reaction_bytes
    };

    double* device_inputs[5] = {};

    for (int n = 0; n < 5; ++n)
    {
        check(cudaMalloc(&device_inputs[n], input_bytes[n]));

        check(cudaMemcpy(
            device_inputs[n], host_inputs[n], input_bytes[n],
            cudaMemcpyHostToDevice));
    }

    double* d_mass_rates = nullptr;
    check(cudaMalloc(&d_mass_rates, species_bytes));

    check(launch_species_mass_rates_test_cuda(
        ns, nr,
        device_inputs[0],
        device_inputs[1], device_inputs[2],
        device_inputs[3], device_inputs[4],
        d_mass_rates));

    check(cudaDeviceSynchronize());

    check(cudaMemcpy(
        h_mass_rates, d_mass_rates, species_bytes,
        cudaMemcpyDeviceToHost));

    for (int n = 0; n < 5; ++n)
        check(cudaFree(device_inputs[n]));

    check(cudaFree(d_mass_rates));
}

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
    double* h_mass_sources)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "Reaction substep CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    // 在计算大小和分配显存之前检查容量
    if (ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS)
    {
        check(cudaErrorInvalidValue);
    }

    size_t species_bytes =
        static_cast<size_t>(ns) * sizeof(double);

    size_t coeff_bytes = 9 * species_bytes;

    size_t reaction_bytes =
        static_cast<size_t>(nr) * sizeof(double);

    size_t table_bytes =
        static_cast<size_t>(ns) * nr * sizeof(double);

    const double* host_inputs[11] = {
        h_concentration,
        h_density,
        h_molecular_weights,
        h_coeff0,
        h_coeff1,
        h_af,
        h_bf,
        h_eaf,
        h_stoi_f,
        h_stoi_b,
        h_efficiencies
    };

    size_t input_bytes[11] = {
        species_bytes,
        species_bytes,
        species_bytes,
        coeff_bytes,
        coeff_bytes,
        reaction_bytes,
        reaction_bytes,
        reaction_bytes,
        table_bytes,
        table_bytes,
        table_bytes
    };

    double* device_inputs[11] = {};

    for (int n = 0; n < 11; ++n)
    {
        check(cudaMalloc(&device_inputs[n], input_bytes[n]));

        check(cudaMemcpy(
            device_inputs[n], host_inputs[n], input_bytes[n],
            cudaMemcpyHostToDevice));
    }

    double* d_updated_density = nullptr;
    double* d_mass_sources = nullptr;

    check(cudaMalloc(&d_updated_density, species_bytes));
    check(cudaMalloc(&d_mass_sources, species_bytes));

    check(launch_reaction_substep_test_cuda(
        T, substep_dt, R, Ru, P0,
        device_inputs[0],
        device_inputs[1],
        device_inputs[2],
        device_inputs[3], device_inputs[4],
        device_inputs[5], device_inputs[6], device_inputs[7],
        device_inputs[8], device_inputs[9], device_inputs[10],
        ns, nr,
        d_updated_density, d_mass_sources));

    check(cudaDeviceSynchronize());

    check(cudaMemcpy(
        h_updated_density, d_updated_density, species_bytes,
        cudaMemcpyDeviceToHost));

    check(cudaMemcpy(
        h_mass_sources, d_mass_sources, species_bytes,
        cudaMemcpyDeviceToHost));

    for (int n = 0; n < 11; ++n)
        check(cudaFree(device_inputs[n]));

    check(cudaFree(d_updated_density));
    check(cudaFree(d_mass_sources));
}

__global__ void composition_temperature_test_kernel(
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
    double* state)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    double total_density;

    rebuild_composition_cuda(
        species_density,
        molecular_weights,
        ns,
        total_density,
        mass_fractions,
        original_concentration);

    double temperature = recover_temperature_cuda(
        initial_temperature,
        internal_energy,
        mass_fractions,
        species_gas_constants,
        coeff0, coeff1,
        ns);

    state[0] = total_density;
    state[1] = temperature;
}

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
    double* state)
{
    if (ns <= 0)
        return cudaErrorInvalidValue;

    composition_temperature_test_kernel<<<1, 1>>>(
        species_density,
        molecular_weights,
        species_gas_constants,
        coeff0, coeff1,
        ns,
        initial_temperature,
        internal_energy,
        mass_fractions,
        original_concentration,
        state);

    return cudaGetLastError();
}

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
    double* h_state)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "State update CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    if (ns <= 0)
        check(cudaErrorInvalidValue);

    size_t species_bytes =
        static_cast<size_t>(ns) * sizeof(double);

    size_t coeff_bytes = 9 * species_bytes;

    const double* host_inputs[5] = {
        h_density, h_mw, h_ri, h_coeff0, h_coeff1
    };

    size_t input_bytes[5] = {
        species_bytes, species_bytes, species_bytes,
        coeff_bytes, coeff_bytes
    };

    double* device_inputs[5] = {};

    for (int n = 0; n < 5; ++n)
    {
        check(cudaMalloc(&device_inputs[n], input_bytes[n]));

        check(cudaMemcpy(
            device_inputs[n], host_inputs[n], input_bytes[n],
            cudaMemcpyHostToDevice));
    }

    double* d_yi = nullptr;
    double* d_mc = nullptr;
    double* d_state = nullptr;

    check(cudaMalloc(&d_yi, species_bytes));
    check(cudaMalloc(&d_mc, species_bytes));
    check(cudaMalloc(&d_state, 2 * sizeof(double)));

    check(launch_composition_temperature_test_cuda(
        device_inputs[0],
        device_inputs[1],
        device_inputs[2],
        device_inputs[3], device_inputs[4],
        ns, initial_temperature, internal_energy,
        d_yi, d_mc, d_state));

    check(cudaDeviceSynchronize());

    check(cudaMemcpy(
        h_yi, d_yi, species_bytes, cudaMemcpyDeviceToHost));

    check(cudaMemcpy(
        h_mc, d_mc, species_bytes, cudaMemcpyDeviceToHost));

    check(cudaMemcpy(
        h_state, d_state, 2 * sizeof(double),
        cudaMemcpyDeviceToHost));

    for (int n = 0; n < 5; ++n)
        check(cudaFree(device_inputs[n]));

    check(cudaFree(d_yi));
    check(cudaFree(d_mc));
    check(cudaFree(d_state));
}

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
    double* h_mass_sources)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "Full reaction substep CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    if (ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS)
    {
        check(cudaErrorInvalidValue);
    }

    size_t species_bytes =
        static_cast<size_t>(ns) * sizeof(double);

    size_t coeff_bytes = 9 * species_bytes;

    size_t reaction_bytes =
        static_cast<size_t>(nr) * sizeof(double);

    size_t table_bytes =
        static_cast<size_t>(ns) * nr * sizeof(double);

    const double* host_inputs[12] = {
        h_concentration,
        h_density,
        h_mw,
        h_ri,
        h_coeff0,
        h_coeff1,
        h_af,
        h_bf,
        h_eaf,
        h_stoi_f,
        h_stoi_b,
        h_efficiencies
    };

    size_t input_bytes[12] = {
        species_bytes,
        species_bytes,
        species_bytes,
        species_bytes,
        coeff_bytes,
        coeff_bytes,
        reaction_bytes,
        reaction_bytes,
        reaction_bytes,
        table_bytes,
        table_bytes,
        table_bytes
    };

    double* device_inputs[12] = {};

    for (int n = 0; n < 12; ++n)
    {
        check(cudaMalloc(&device_inputs[n], input_bytes[n]));

        check(cudaMemcpy(
            device_inputs[n], host_inputs[n], input_bytes[n],
            cudaMemcpyHostToDevice));
    }

    ReactionFlowState* d_state = nullptr;
    double* d_mass_sources = nullptr;

    check(cudaMalloc(&d_state, sizeof(ReactionFlowState)));
    check(cudaMalloc(&d_mass_sources, species_bytes));

    check(launch_full_reaction_substep_test_cuda(
        T, u, v, w,
        internal_energy, substep_dt,
        R, Ru, P0,
        device_inputs[0], device_inputs[1],
        device_inputs[2], device_inputs[3],
        device_inputs[4], device_inputs[5],
        device_inputs[6], device_inputs[7], device_inputs[8],
        device_inputs[9], device_inputs[10], device_inputs[11],
        ns, nr,
        d_state, d_mass_sources));

    check(cudaDeviceSynchronize());

    check(cudaMemcpy(
        &output_state, d_state, sizeof(ReactionFlowState),
        cudaMemcpyDeviceToHost));

    check(cudaMemcpy(
        h_mass_sources, d_mass_sources, species_bytes,
        cudaMemcpyDeviceToHost));

    for (int n = 0; n < 12; ++n)
        check(cudaFree(device_inputs[n]));

    check(cudaFree(d_state));
    check(cudaFree(d_mass_sources));
}


namespace {
struct PackedReactionStorage {
    double* device=nullptr;
    double* host=nullptr; // Page-locked host memory, reused across iterations.
    size_t capacity=0;
    bool ready=false, validate=false;
    int count=0, ns=0;
    PackedReactionStorage()=default;
    PackedReactionStorage(const PackedReactionStorage&)=delete;
    PackedReactionStorage& operator=(const PackedReactionStorage&)=delete;
    ~PackedReactionStorage(){cudaFree(device);cudaFreeHost(host);}
};
PackedReactionStorage& packed_reaction_storage(){static PackedReactionStorage s;return s;}
void packed_cuda_check(cudaError_t error){
    if(error!=cudaSuccess){
        std::fprintf(stderr,"Packed output CUDA error: %s\n",cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}
}
__global__ void pack_reaction_output_kernel(const ReactionFlowState* states,
                                           const double* sources, double* out,
                                           int count, int ns)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=count) return;
    const auto& state=states[cell];
    size_t base=0;
    for(int s=0;s<ns;++s) out[base+static_cast<size_t>(cell)*ns+s]=state.species_density[s];
    base+=static_cast<size_t>(count)*ns;
    for(int s=0;s<ns;++s) out[base+static_cast<size_t>(cell)*ns+s]=state.mass_fractions[s];
    base+=static_cast<size_t>(count)*ns;
    for(int s=0;s<ns;++s) out[base+static_cast<size_t>(cell)*ns+s]=state.concentration[s];
    base+=static_cast<size_t>(count)*ns;
    for(int s=0;s<ns;++s) out[base+static_cast<size_t>(cell)*ns+s]=sources[static_cast<size_t>(cell)*ns+s];
    base+=static_cast<size_t>(count)*ns;
    for(int s=0;s<ns+4;++s) out[base+static_cast<size_t>(cell)*(ns+4)+s]=state.conserved[s];
    base+=static_cast<size_t>(count)*(ns+4);
    out[base+cell]=state.density; base+=count;
    out[base+cell]=state.temperature; base+=count;
    out[base+cell]=state.u; base+=count;
    out[base+cell]=state.v; base+=count;
    out[base+cell]=state.w; base+=count;
    out[base+cell]=state.internal_energy; base+=count;
    out[base+cell]=state.mean_molecular_weight; base+=count;
    out[base+cell]=state.gas_constant; base+=count;
    out[base+cell]=state.specific_heat_cp; base+=count;
    out[base+cell]=state.pressure; base+=count;
    out[base+cell]=state.enthalpy; base+=count;
    out[base+cell]=state.gamma; base+=count;
    out[base+cell]=state.sound_speed; base+=count;
    out[base+cell]=state.mach; base+=count;

}
static void download_packed_reaction(const ReactionFlowState* states,const double* sources,
                                     int count,int ns,bool validate)
{
    auto& output=packed_reaction_storage();
    const size_t elements=static_cast<size_t>(count)*(5*ns+18);
    if(elements>output.capacity){
        packed_cuda_check(cudaFree(output.device)); output.device=nullptr;
        packed_cuda_check(cudaFreeHost(output.host)); output.host=nullptr;
        output.capacity=0;
        packed_cuda_check(cudaMalloc(&output.device,elements*sizeof(double)));
        packed_cuda_check(cudaMallocHost(&output.host,elements*sizeof(double)));
        output.capacity=elements;
    }
    pack_reaction_output_kernel<<<(count+127)/128,128>>>(states,sources,output.device,count,ns);
    packed_cuda_check(cudaGetLastError());
    packed_cuda_check(cudaMemcpy(output.host,output.device,elements*sizeof(double),cudaMemcpyDeviceToHost));
    output.count=count;output.ns=ns;output.validate=validate;output.ready=true;
    std::printf("Packed reaction output: cells=%d, bytes=%zu, validation=%d\n",
                count,elements*sizeof(double),int(validate));
}
const double* reaction_packed_output(int count,int ns,bool& validate)
{
    const auto& output=packed_reaction_storage();
    validate=false;
    if(!output.ready) return nullptr;
    if(output.count!=count || output.ns!=ns) chain_fail("packed output dimensions do not match");
    validate=output.validate;
    return output.host;
}

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
    bool validate_nchem)
{
    auto check = [](cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "Reaction cells CUDA error: %s\n",
                         cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
    };

    if (cell_count <= 0 ||
        ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS)
    {
        check(cudaErrorInvalidValue);
    }

    const bool chained = post_advection_chain_pending();
    if (state_chain.enabled && !chained) chain_fail("missing device state");
    if (chained && (state_chain.count != cell_count || state_chain.ns != ns || validate_nchem))
        chain_fail("device state dimensions or validation mode do not match");
    const bool validate_chain = chained && state_chain.validate;
    const bool packed = chain_env_enabled("EU3D_PACKED_OUTPUT");
    const bool validate_packed = chain_env_enabled("EU3D_VALIDATE_PACKED");
    packed_reaction_storage().ready=false;
    if(packed && !chained) chain_fail("EU3D_PACKED_OUTPUT requires the enabled state chain");
    if(validate_packed && !packed) chain_fail("EU3D_VALIDATE_PACKED requires EU3D_PACKED_OUTPUT=1");

    size_t state_bytes =
        static_cast<size_t>(cell_count)
        * sizeof(ReactionFlowState);

    size_t nchem_bytes =
        static_cast<size_t>(cell_count) * sizeof(int);

    size_t source_bytes =
        static_cast<size_t>(cell_count) * ns * sizeof(double);

    size_t species_bytes =
        static_cast<size_t>(ns) * sizeof(double);

    size_t coeff_bytes = 9 * species_bytes;

    size_t reaction_bytes =
        static_cast<size_t>(nr) * sizeof(double);

    size_t table_bytes =
        static_cast<size_t>(ns) * nr * sizeof(double);

    const double* host_parameters[10] = {
        h_mw,
        h_ri,
        h_coeff0,
        h_coeff1,
        h_af,
        h_bf,
        h_eaf,
        h_stoi_f,
        h_stoi_b,
        h_efficiencies
    };

    size_t parameter_bytes[10] = {
        species_bytes,
        species_bytes,
        coeff_bytes,
        coeff_bytes,
        reaction_bytes,
        reaction_bytes,
        reaction_bytes,
        table_bytes,
        table_bytes,
        table_bytes
    };

    ReactionBuffers& buffers = get_reaction_buffers();

    // 第一次调用，或尺寸变化时，才重新申请显存
    bool size_changed =
        buffers.cell_count != cell_count ||
        buffers.ns != ns ||
        buffers.nr != nr;

    if (size_changed)
    {
        buffers.release();

        for (int n = 0; n < 10; ++n)
        {
            check(cudaMalloc(
                &buffers.parameters[n],
                parameter_bytes[n]));
        }

        check(cudaMalloc(&buffers.states, state_bytes));
        check(cudaMalloc(&buffers.nchem, nchem_bytes));
        check(cudaMalloc(&buffers.mass_sources, source_bytes));

        buffers.cell_count = cell_count;
        buffers.ns = ns;
        buffers.nr = nr;
    }

    // 使用缓冲区保存的地址，后续代码沿用原变量名
    double** device_parameters = buffers.parameters;
    ReactionFlowState* d_states = chained ? state_chain.pending : buffers.states;
    int* d_nchem = buffers.nchem;
    double* d_mass_sources = buffers.mass_sources;

    const double upload_start = wall_seconds();

    // 参数仍每次上传，先只优化显存申请与释放
    for (int n = 0; n < 10; ++n)
    {
        check(cudaMemcpy(
            device_parameters[n],
            host_parameters[n],
            parameter_bytes[n],
            cudaMemcpyHostToDevice));
    }

    if (!chained) {
        check(cudaMemcpy(d_states, h_states, state_bytes, cudaMemcpyHostToDevice));
    }
    
    // 单独计量上传，避免将尚未完成的传输记入预测阶段。
    check(cudaDeviceSynchronize());
    const double upload_end = wall_seconds();

    std::vector<ReactionFlowState> reference_states;
    if (validate_chain) {
        reference_states.resize(cell_count);
        check(cudaMemcpy(reference_states.data(), d_states, state_bytes, cudaMemcpyDeviceToHost));
        for (auto& full : reference_states) {
            ReactionFlowState packed{};
            packed.temperature = full.temperature;
            packed.u = full.u; packed.v = full.v; packed.w = full.w;
            packed.internal_energy = full.internal_energy;
            for (int s = 0; s < ns; ++s) {
                packed.species_density[s] = full.species_density[s];
                packed.concentration[s] = full.concentration[s];
                packed.mass_fractions[s] = full.mass_fractions[s];
            }
            full = packed;
        }
        check(cudaMemcpy(buffers.states, reference_states.data(), state_bytes, cudaMemcpyHostToDevice));
    }
    const double fused_start = wall_seconds();
    check(launch_reaction_cells_fused_cuda(
        d_states, d_nchem, cell_count,
        dt, R, Ru, P0,
        device_parameters[0], device_parameters[1],
        device_parameters[2], device_parameters[3],
        device_parameters[4], device_parameters[5],
        device_parameters[6],
        device_parameters[7], device_parameters[8],
        device_parameters[9], ns, nr, d_mass_sources));
    check(cudaDeviceSynchronize());
    const double fused_end = wall_seconds();

    // 积分已在设备上完成；先检查预测结果，再将状态交给调用方。
    std::vector<int> predicted_nchem(cell_count);

    check(cudaMemcpy(
        predicted_nchem.data(), d_nchem, nchem_bytes,
        cudaMemcpyDeviceToHost));

    int mismatches = 0;
    int max_substeps = 0;

    for (int c = 0; c < cell_count; ++c)
    {
        int predicted = predicted_nchem[c];

        if (predicted > max_substeps)
            max_substeps = predicted;

        if (predicted < 1 ||
            (validate_nchem && predicted != h_nchem[c]))
        {
            if (mismatches < 10)
            {
                std::fprintf(stderr,
                    "Nchem error: cell=%d, input=%d, GPU=%d\n",
                    c, h_nchem[c], predicted);
            }

            ++mismatches;
        }
    }

    if (validate_nchem)
    {
        std::printf(
            "Nchem validation: %s, mismatches=%d, max substeps=%d\n",
            mismatches == 0 ? "Passed" : "Failed",
            mismatches, max_substeps);
    }
    else
    {
        std::printf("GPU Nchem: invalid=%d, max substeps=%d\n",
                    mismatches, max_substeps);
    }

    if (mismatches != 0)
        std::exit(EXIT_FAILURE);

    for (int c = 0; c < cell_count; ++c)
        h_nchem[c] = predicted_nchem[c];

    const double check_end = wall_seconds();

    if (packed)
        download_packed_reaction(d_states,d_mass_sources,cell_count,ns,validate_packed);
    // Retain the original transfer as an independent reference in validation modes.
    if (!packed || validate_packed || validate_chain || validate_nchem) {
    // 取回更新后的全部网格状态
    check(cudaMemcpy(
        h_states, d_states, state_bytes,
        cudaMemcpyDeviceToHost));

    // 取回各网格最后一个反应子步的质量源项
    check(cudaMemcpy(
        h_mass_sources, d_mass_sources, source_bytes,
        cudaMemcpyDeviceToHost));
    }

    const double download_end = wall_seconds();
    if (resident_ee.enabled) {
        if (!chained || d_states != resident_ee.states
            || cell_count != resident_ee.ni*resident_ee.nj*resident_ee.nk
            || ns != resident_ee.ns)
            chain_fail("resident EE reaction state ownership mismatch");
        // Reaction completed in the same allocation used by the EE bridge.
        // Host scatter below remains unchanged; only the next upload is skipped.
        resident_ee.ready = true;
    }
    if (chained) {
        state_chain.pending = nullptr;
        state_chain.count = state_chain.ns = 0;
    }
    if (validate_chain) {
        const double reference_start = wall_seconds();
        check(launch_reaction_cells_fused_cuda(
            buffers.states, d_nchem, cell_count, dt, R, Ru, P0,
            device_parameters[0], device_parameters[1], device_parameters[2],
            device_parameters[3], device_parameters[4], device_parameters[5],
            device_parameters[6], device_parameters[7], device_parameters[8],
            device_parameters[9], ns, nr, d_mass_sources));
        check(cudaMemcpy(reference_states.data(), buffers.states, state_bytes, cudaMemcpyDeviceToHost));
        std::vector<int> reference_nchem(cell_count);
        std::vector<double> reference_sources(static_cast<size_t>(cell_count) * ns);
        check(cudaMemcpy(reference_nchem.data(), d_nchem, nchem_bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(reference_sources.data(), d_mass_sources, source_bytes, cudaMemcpyDeviceToHost));
        size_t differences = 0;
        auto compare = [&](const char* name, int cell, int component, double expected, double actual) {
            // Identical GPU arithmetic: require exact finite values, no tolerance exception.
            if (!std::isfinite(expected) || !std::isfinite(actual) || expected != actual) {
                if (differences < 8)
                    std::fprintf(stderr, "Chain mismatch: %s cell=%d component=%d reference=%.17g chained=%.17g\n",
                                 name, cell, component, expected, actual);
                ++differences;
            }
        };
        for (int c = 0; c < cell_count; ++c) {
            const auto& expected = reference_states[c];
            const auto& actual = h_states[c];
            compare("Nchem", c, -1, reference_nchem[c], h_nchem[c]);
            for (int s = 0; s < ns; ++s) {
                compare("Di", c, s, expected.species_density[s], actual.species_density[s]);
                compare("Yi", c, s, expected.mass_fractions[s], actual.mass_fractions[s]);
                compare("Mc", c, s, expected.concentration[s], actual.concentration[s]);
                const size_t index = static_cast<size_t>(c) * ns + s;
                compare("CMS", c, s, reference_sources[index], h_mass_sources[index]);
            }
            for (int s = 0; s < ns + 4; ++s)
                compare("CS", c, s, expected.conserved[s], actual.conserved[s]);
#define COMPARE_STATE_MEMBER(name) compare(#name, c, -1, expected.name, actual.name)
            COMPARE_STATE_MEMBER(density); COMPARE_STATE_MEMBER(temperature);
            COMPARE_STATE_MEMBER(u); COMPARE_STATE_MEMBER(v); COMPARE_STATE_MEMBER(w);
            COMPARE_STATE_MEMBER(internal_energy); COMPARE_STATE_MEMBER(mean_molecular_weight);
            COMPARE_STATE_MEMBER(gas_constant); COMPARE_STATE_MEMBER(specific_heat_cp);
            COMPARE_STATE_MEMBER(pressure); COMPARE_STATE_MEMBER(enthalpy);
            COMPARE_STATE_MEMBER(gamma); COMPARE_STATE_MEMBER(sound_speed); COMPARE_STATE_MEMBER(mach);
#undef COMPARE_STATE_MEMBER
        }
        std::printf("State chain validation: %s, cells=%d, mismatches=%zu, comparison=exact, reference_ms=%.3f\n",
                    differences ? "Failed" : "Passed", cell_count, differences,
                    (wall_seconds() - reference_start) * 1000.0);
        if (differences) chain_fail("reference round-trip comparison failed");
    }
    if (chained)
        std::printf("State chain: reused device state for %d cells; intermediate download/pack/upload skipped\n", cell_count);
    // fused_compute 含预测和全部反应子步的启动、执行及同步。
    // nchem_check 含子步数回传与 CPU 检查；不再单独启动预测核函数。
    std::printf(
        "Reaction breakdown: upload=%.3f ms, "
        "fused_compute=%.3f ms, nchem_check=%.3f ms, "
        "download=%.3f ms\n",
        (upload_end - upload_start) * 1000.0,
        (fused_end - fused_start) * 1000.0,
        (check_end - fused_end) * 1000.0,
        (download_end - check_end) * 1000.0);


}

__global__ void predict_reaction_cells_kernel(
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
    int ns,
    int nr)
{
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell_idx >= cell_count)
        return;

    const ReactionFlowState& state = states[cell_idx];
    ReactionCellData cell;

    // 使用当前温度和浓度计算反应速率
    prepare_reaction_cell_cuda(
        state.temperature,
        R, Ru, P0,
        state.concentration,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr,
        cell);

    // 根据各组分的变化速率决定子步数
    nchem[cell_idx] = predict_reaction_substeps_cuda(
        dt,
        state,
        cell,
        molecular_weights,
        stoi_f, stoi_b,
        ns, nr);
}

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
    int ns, int nr)
{
    if (cell_count <= 0 ||
        ns <= 0 || ns > REACTION_MAX_SPECIES ||
        nr <= 0 || nr > REACTION_MAX_REACTIONS ||
        !std::isfinite(dt) || dt <= 0.0)
    {
        return cudaErrorInvalidValue;
    }

    int threads = 128;
    int blocks = (cell_count - 1) / threads + 1;

    predict_reaction_cells_kernel<<<blocks, threads>>>(
        states, nchem, cell_count,
        dt, R, Ru, P0,
        molecular_weights,
        coeff0, coeff1,
        af, bf, eaf,
        stoi_f, stoi_b, efficiencies,
        ns, nr);

    return cudaGetLastError();
}

__global__ void post_advection_state_kernel(ReactionFlowState* states, int count, int ns, double R, const double* mw, const double* ri, const double* c0, const double* c1, int trace_id=-1)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    ReactionFlowState& s = states[id];
    for (int n=0;n<ns;++n) s.species_density[n]=s.conserved[n];
    rebuild_composition_cuda(s.species_density,mw,ns,s.density,s.mass_fractions,s.concentration);
    s.temperature=recover_temperature_cuda(s.temperature,s.internal_energy,s.mass_fractions,ri,c0,c1,ns,id==trace_id);
    s.u=s.conserved[ns]/s.density;
    s.v=s.conserved[ns+1]/s.density;
    s.w=s.conserved[ns+2]/s.density;
    s.internal_energy=(s.conserved[ns+3]-0.5*s.density*(s.u*s.u+s.v*s.v+s.w*s.w))/s.density;
    mixture_properties_cuda(s.temperature,R,s.mass_fractions,mw,ri,c0,c1,ns,s.mean_molecular_weight,s.gas_constant,s.specific_heat_cp);
    s.pressure=s.temperature*s.density*s.gas_constant;
    s.enthalpy=s.pressure/s.density+s.internal_energy;
    double inverse_weight=0;
    for(int n=0;n<ns;++n) inverse_weight+=s.mass_fractions[n]/mw[n];
    s.gamma=s.specific_heat_cp/(s.specific_heat_cp-R*inverse_weight*1000);
    s.sound_speed=sqrt(s.gamma*s.pressure/s.density);
    s.mach=sqrt(s.u*s.u+s.v*s.v+s.w*s.w)/s.sound_speed;
}
void post_advection_state_host(ReactionFlowState* states, int count, int ns, double R, const double* mw, const double* ri, const double* c0, const double* c1)
{
    auto check=[](cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"Post-advection CUDA error: %s\n",cudaGetErrorString(e));std::abort();}};
    if(ns<1 || ns>REACTION_MAX_SPECIES || count<1) std::abort();
    struct Storage {
        ReactionFlowState* d=nullptr; double* p=nullptr; int capacity=0;
        ~Storage(){cudaFree(d);cudaFree(p);}
    };
    static Storage b;
    if(b.capacity<count){check(cudaFree(b.d));b.d=nullptr;check(cudaMalloc(&b.d,size_t(count)*sizeof(*states)));b.capacity=count;}
    if(!b.p) check(cudaMalloc(&b.p,20*REACTION_MAX_SPECIES*sizeof(double)));
    double* dm=b.p; double* dr=dm+ns; double* d0=dr+ns; double* d1=d0+9*ns;
    check(cudaMemcpy(dm,mw,ns*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dr,ri,ns*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(d0,c0,9*ns*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(d1,c1,9*ns*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.d,states,size_t(count)*sizeof(*states),cudaMemcpyHostToDevice));
    post_advection_state_kernel<<<(count+127)/128,128>>>(b.d,count,ns,R,dm,dr,d0,d1,std::getenv("EU3D_TRACE_TEMP") ? 142912 : -1);
    check(cudaGetLastError());
    if (state_chain.enabled) {
        if (state_chain.pending) chain_fail("attempted to overwrite pending device state");
        check(cudaDeviceSynchronize());
        state_chain.pending = b.d;
        state_chain.count = count;
        state_chain.ns = ns;
        return;
    }
    check(cudaMemcpy(states,b.d,size_t(count)*sizeof(*states),cudaMemcpyDeviceToHost));
}


namespace {
struct EeStateStorage {
    double* host_input=nullptr;
    double* device_input=nullptr;
    double* parameters=nullptr;
    ReactionFlowState* states=nullptr;
    size_t input_capacity=0, state_capacity=0;
    EeStateStorage()=default;
    EeStateStorage(const EeStateStorage&)=delete;
    EeStateStorage& operator=(const EeStateStorage&)=delete;
    ~EeStateStorage(){cudaFreeHost(host_input);cudaFree(device_input);cudaFree(parameters);cudaFree(states);}
};
}
__global__ void prepare_ee_state_kernel(const double* cs,const double* temperature,const double* energy,
    const double* rhs,ReactionFlowState* states,int ni,int nj,int nk,int bc,int ns)
{
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    const size_t count=static_cast<size_t>(ni)*nj*nk;
    if(static_cast<size_t>(cell)>=count) return;
    const int i=cell/(nj*nk)+bc;
    const int j=(cell/nk)%nj+bc;
    const int k=cell%nk+bc;
    const size_t grid=(static_cast<size_t>(i)*(nj+2*bc)+j)*(nk+2*bc)+k;
    ReactionFlowState& state=states[cell];
    state.temperature=temperature[grid];
    state.internal_energy=energy[grid];
    for(int s=0;s<ns+4;++s){
        const size_t index=grid*(ns+4)+s;
        state.conserved[s]=cs[index]+rhs[index];
    }
}
__global__ void advance_resident_ee_kernel(const double* rhs, ReactionFlowState* states,
    int ni, int nj, int nk, int bc, int ns)
{
    const size_t cell = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (cell >= static_cast<size_t>(ni)*nj*nk) return;
    const size_t i = cell/(nj*nk)+bc;
    const size_t j = (cell/nk)%nj+bc;
    const size_t k = cell%nk+bc;
    const size_t grid = (i*(nj+2*bc)+j)*(nk+2*bc)+k;
    for (int s=0; s<ns+4; ++s)
        states[cell].conserved[s] = states[cell].conserved[s] + rhs[grid*(ns+4)+s];
    // Temperature and internal energy are already the previous reaction result.
}
void post_advection_from_ee_host(const double* cs,const double* temperature,const double* energy,
    int ni,int nj,int nk,int bc,int ns,double R,
    const double* mw,const double* ri,const double* coeff0,const double* coeff1)
{
    if(!state_chain.enabled || state_chain.pending || !ee_state.enabled || !ee_state.rhs)
        chain_fail("EE/state bridge called outside its valid lifetime");
    if(ni<=0 || nj<=0 || nk<=0 || bc<1 || ns<1 || ns>REACTION_MAX_SPECIES
       || ee_state.nx!=ni+2*bc || ee_state.ny!=nj+2*bc || ee_state.nz!=nk+2*bc
       || ee_state.nvar!=ns+4 || ee_state.bc!=bc)
        chain_fail("EE/state bridge dimensions do not match");
    auto check=[](cudaError_t e){packed_cuda_check(e);};
    const size_t count=static_cast<size_t>(ni)*nj*nk;
    const size_t full=static_cast<size_t>(ni+2*bc)*(nj+2*bc)*(nk+2*bc);
    const size_t cs_elements=full*(ns+4);
    const size_t input_elements=full*(ns+6);
    static EeStateStorage b;
    const bool reuse = resident_ee.enabled && resident_ee.ready
        && resident_ee.states == b.states && resident_ee.ni == ni
        && resident_ee.nj == nj && resident_ee.nk == nk
        && resident_ee.bc == bc && resident_ee.ns == ns;
    // Any new solve or changed shape starts by seeding from host state.
    resident_ee.ready = false;
    if(!reuse && b.input_capacity<input_elements){
        check(cudaFreeHost(b.host_input));b.host_input=nullptr;
        check(cudaFree(b.device_input));b.device_input=nullptr;b.input_capacity=0;
        check(cudaMallocHost(&b.host_input,input_elements*sizeof(double)));
        check(cudaMalloc(&b.device_input,input_elements*sizeof(double)));
        b.input_capacity=input_elements;
    }
    if(b.state_capacity<count){
        check(cudaFree(b.states));b.states=nullptr;b.state_capacity=0;
        check(cudaMalloc(&b.states,count*sizeof(ReactionFlowState)));b.state_capacity=count;
    }
    if(!b.parameters) check(cudaMalloc(&b.parameters,20*REACTION_MAX_SPECIES*sizeof(double)));
    double* dm=b.parameters;double* dr=dm+ns;double* d0=dr+ns;double* d1=d0+9*ns;
    check(cudaMemcpy(dm,mw,ns*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dr,ri,ns*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(d0,coeff0,9*ns*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(d1,coeff1,9*ns*sizeof(double),cudaMemcpyHostToDevice));
    // Bulk copies preserve the original grid layout, including unused halo slots.
    const double input_start=wall_seconds();
    if (!reuse) {
    std::memcpy(b.host_input,cs,cs_elements*sizeof(double));
    std::memcpy(b.host_input+cs_elements,temperature,full*sizeof(double));
    std::memcpy(b.host_input+cs_elements+full,energy,full*sizeof(double));
    check(cudaMemcpy(b.device_input,b.host_input,input_elements*sizeof(double),cudaMemcpyHostToDevice));
    }
    const double input_end=wall_seconds();
    if (reuse) {
        advance_resident_ee_kernel<<<(count+127)/128,128>>>(ee_state.rhs,b.states,ni,nj,nk,bc,ns);
    } else {
    prepare_ee_state_kernel<<<(count+127)/128,128>>>(b.device_input,b.device_input+cs_elements,
        b.device_input+cs_elements+full,ee_state.rhs,b.states,ni,nj,nk,bc,ns);
    }
    check(cudaGetLastError());
    post_advection_state_kernel<<<(count+127)/128,128>>>(b.states,int(count),ns,R,dm,dr,d0,d1);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    const double compute_end=wall_seconds();
    if(ee_state.validate){
        // Independent reference: the original CPU CS += RHS and per-cell pack,
        // followed by the already-validated GPU state-recovery kernel.
        std::vector<double> host_rhs(cs_elements);
        check(cudaMemcpy(host_rhs.data(),ee_state.rhs,cs_elements*sizeof(double),cudaMemcpyDeviceToHost));
        std::vector<ReactionFlowState> reference(count),actual(count);
        size_t cell=0;
        for(int i=bc;i<ni+bc;++i) for(int j=bc;j<nj+bc;++j) for(int k=bc;k<nk+bc;++k,++cell){
            const size_t grid=(static_cast<size_t>(i)*(nj+2*bc)+j)*(nk+2*bc)+k;
            reference[cell].temperature=temperature[grid];
            reference[cell].internal_energy=energy[grid];
            for(int s=0;s<ns+4;++s){
                const size_t index=grid*(ns+4)+s;
                reference[cell].conserved[s]=cs[index]+host_rhs[index];
            }
        }
        ReactionFlowState* device_reference=nullptr;
        check(cudaMalloc(&device_reference,count*sizeof(ReactionFlowState)));
        check(cudaMemcpy(device_reference,reference.data(),count*sizeof(ReactionFlowState),cudaMemcpyHostToDevice));
        post_advection_state_kernel<<<(count+127)/128,128>>>(device_reference,int(count),ns,R,dm,dr,d0,d1);
        check(cudaGetLastError());
        check(cudaMemcpy(reference.data(),device_reference,count*sizeof(ReactionFlowState),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(),b.states,count*sizeof(ReactionFlowState),cudaMemcpyDeviceToHost));
        check(cudaFree(device_reference));
        size_t differences=0;
        auto compare=[&](const char* field,size_t c,int component,double expected,double result){
            if(!std::isfinite(expected)||!std::isfinite(result)||expected!=result){
                if(differences<8) std::fprintf(stderr,"EE state mismatch: field=%s cell=%zu component=%d reference=%.17g actual=%.17g\n",
                                               field,c,component,expected,result);
                ++differences;
            }
        };
        for(size_t c=0;c<count;++c){
            const auto& expected=reference[c];const auto& result=actual[c];
            for(int s=0;s<ns;++s){
                compare("Di",c,s,expected.species_density[s],result.species_density[s]);
                compare("Yi",c,s,expected.mass_fractions[s],result.mass_fractions[s]);
                compare("Mc",c,s,expected.concentration[s],result.concentration[s]);
            }
            for(int s=0;s<ns+4;++s) compare("CS",c,s,expected.conserved[s],result.conserved[s]);
#define CHECK_EE_FIELD(field) compare(#field,c,-1,expected.field,result.field)
            CHECK_EE_FIELD(density);CHECK_EE_FIELD(temperature);
            CHECK_EE_FIELD(u);CHECK_EE_FIELD(v);CHECK_EE_FIELD(w);CHECK_EE_FIELD(internal_energy);
            CHECK_EE_FIELD(mean_molecular_weight);CHECK_EE_FIELD(gas_constant);CHECK_EE_FIELD(specific_heat_cp);
            CHECK_EE_FIELD(pressure);CHECK_EE_FIELD(enthalpy);CHECK_EE_FIELD(gamma);
            CHECK_EE_FIELD(sound_speed);CHECK_EE_FIELD(mach);
#undef CHECK_EE_FIELD
        }
        std::printf("EE state validation: %s, cells=%zu, mismatches=%zu, comparison=exact\n",
                    differences?"Failed":"Passed",count,differences);
        if(differences) chain_fail("EE/state bridge reference comparison failed");
    }
    // The state allocation is owned by this storage and consumed by reaction_cells_host.
    state_chain.pending=b.states;state_chain.count=int(count);state_chain.ns=ns;
    if (resident_ee.enabled) {
        resident_ee.states=b.states;
        resident_ee.ni=ni; resident_ee.nj=nj; resident_ee.nk=nk;
        resident_ee.bc=bc; resident_ee.ns=ns;
        std::printf("Resident EE: %s, cells=%zu, skipped_input_bytes=%zu\n",
                    reuse?"reused":"seeded",count,reuse?input_elements*sizeof(double):0);
    }
    ee_state.rhs=nullptr;
    std::printf("EE state bridge: input_bytes=%zu, pack_upload=%.3f ms, state_compute=%.3f ms\n",
                reuse?0:input_elements*sizeof(double),(input_end-input_start)*1000,(compute_end-input_end)*1000);
}
