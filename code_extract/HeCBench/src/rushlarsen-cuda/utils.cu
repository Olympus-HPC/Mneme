#include "utils.h"

// Init state values
void init_state_values(double* states, int n)
{
  for (int i = 0; i < n; i++) {
    states[n * STATE_Xr1 + i] = 0.0165;
    states[n * STATE_Xr2 + i] = 0.473;
    states[n * STATE_Xs + i] = 0.0174;
    states[n * STATE_m + i] = 0.00165;
    states[n * STATE_h + i] = 0.749;
    states[n * STATE_j + i] = 0.6788;
    states[n * STATE_d + i] = 3.288e-05;
    states[n * STATE_f + i] = 0.7026;
    states[n * STATE_f2 + i] = 0.9526;
    states[n * STATE_fCass + i] = 0.9942;
    states[n * STATE_s + i] = 0.999998;
    states[n * STATE_r + i] = 2.347e-08;
    states[n * STATE_Ca_i + i] = 0.000153;
    states[n * STATE_R_prime + i] = 0.8978;
    states[n * STATE_Ca_SR + i] = 4.272;
    states[n * STATE_Ca_ss + i] = 0.00042;
    states[n * STATE_Na_i + i] = 10.132;
    states[n * STATE_V + i] = -85.423;
    states[n * STATE_K_i + i] = 138.52;
  }
}

// Default parameter values
void init_parameters_values(double* parameters, int n)
{
  for (int i = 0; i < n; i++) {
    parameters[n * PARAM_P_kna + i] = 0.03;
    parameters[n * PARAM_g_K1 + i] = 5.405;
    parameters[n * PARAM_g_Kr + i] = 0.153;
    parameters[n * PARAM_g_Ks + i] = 0.098;
    parameters[n * PARAM_g_Na + i] = 14.838;
    parameters[n * PARAM_g_bna + i] = 0.00029;
    parameters[n * PARAM_g_CaL + i] = 3.98e-05;
    parameters[n * PARAM_g_bca + i] = 0.000592;
    parameters[n * PARAM_g_to + i] = 0.294;
    parameters[n * PARAM_K_mNa + i] = 40;
    parameters[n * PARAM_K_mk + i] = 1;
    parameters[n * PARAM_P_NaK + i] = 2.724;
    parameters[n * PARAM_K_NaCa + i] = 1000;
    parameters[n * PARAM_K_sat + i] = 0.1;
    parameters[n * PARAM_Km_Ca + i] = 1.38;
    parameters[n * PARAM_Km_Nai + i] = 87.5;
    parameters[n * PARAM_alpha + i] = 2.5;
    parameters[n * PARAM_gamma + i] = 0.35;
    parameters[n * PARAM_K_pCa + i] = 0.0005;
    parameters[n * PARAM_g_pCa + i] = 0.1238;
    parameters[n * PARAM_g_pK + i] = 0.0146;
    parameters[n * PARAM_Buf_c + i] = 0.2;
    parameters[n * PARAM_Buf_sr + i] = 10;
    parameters[n * PARAM_Buf_ss + i] = 0.4;
    parameters[n * PARAM_Ca_o + i] = 2;
    parameters[n * PARAM_EC + i] = 1.5;
    parameters[n * PARAM_K_buf_c + i] = 0.001;
    parameters[n * PARAM_K_buf_sr + i] = 0.3;
    parameters[n * PARAM_K_buf_ss + i] = 0.00025;
    parameters[n * PARAM_K_up + i] = 0.00025;
    parameters[n * PARAM_V_leak + i] = 0.00036;
    parameters[n * PARAM_V_rel + i] = 0.102;
    parameters[n * PARAM_V_sr + i] = 0.001094;
    parameters[n * PARAM_V_ss + i] = 5.468e-05;
    parameters[n * PARAM_V_xfer + i] = 0.0038;
    parameters[n * PARAM_Vmax_up + i] = 0.006375;
    parameters[n * PARAM_k1_prime + i] = 0.15;
    parameters[n * PARAM_k2_prime + i] = 0.045;
    parameters[n * PARAM_k3 + i] = 0.06;
    parameters[n * PARAM_k4 + i] = 0.005;
    parameters[n * PARAM_max_sr + i] = 2.5;
    parameters[n * PARAM_min_sr + i] = 1.0;
    parameters[n * PARAM_Na_o + i] = 140;
    parameters[n * PARAM_Cm + i] = 0.185;
    parameters[n * PARAM_F + i] = 96485.3415;
    parameters[n * PARAM_R + i] = 8314.472;
    parameters[n * PARAM_T + i] = 310;
    parameters[n * PARAM_V_c + i] = 0.016404;
    parameters[n * PARAM_stim_amplitude + i] = 52;
    parameters[n * PARAM_stim_duration + i] = 1;
    parameters[n * PARAM_stim_period + i] = 1000;
    parameters[n * PARAM_stim_start + i] = 10;
    parameters[n * PARAM_K_o + i] = 5.4;
  }
}


