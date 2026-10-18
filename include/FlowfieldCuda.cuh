#pragma once

void launch_muscl2_cuda(
    const double* fi,
    double* flr,
    int nx,
    int ny,
    int nz,
    int bc,
    int direaction);

void muscl2_host(
    const double* h_fi,
    double* h_flr,
    int nx,
    int ny,
    int nz,
    int bc,
    int direction);

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
    int direction);

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
    int direction);

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
    bool copy_flux_to_host = true);

const double* get_ausm_device_flux(int direction);
// Scope is one uninterrupted x/y/z sweep over unchanged host fields.
void begin_muscl_upload_scope(bool enabled);
void end_muscl_upload_scope();
