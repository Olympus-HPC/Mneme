#include <math.h>
#include "utils.h"

void forward_rush_larsen(double* states, const double t, const double dt,
                         const double* parameters, const int n)
{
  for (int i = 0; i < n; i++) {
    // Assign states
    const double Xr1 = states[n * STATE_Xr1 + i];
    const double Xr2 = states[n * STATE_Xr2 + i];
    const double Xs = states[n * STATE_Xs + i];
    const double m = states[n * STATE_m + i];
    const double h = states[n * STATE_h + i];
    const double j = states[n * STATE_j + i];
    const double d = states[n * STATE_d + i];
    const double f = states[n * STATE_f + i];
    const double f2 = states[n * STATE_f2 + i];
    const double fCass = states[n * STATE_fCass + i];
    const double s = states[n * STATE_s + i];
    const double r = states[n * STATE_r + i];
    const double Ca_i = states[n * STATE_Ca_i + i];
    const double R_prime = states[n * STATE_R_prime + i];
    const double Ca_SR = states[n * STATE_Ca_SR + i];
    const double Ca_ss = states[n * STATE_Ca_ss + i];
    const double Na_i = states[n * STATE_Na_i + i];
    const double V = states[n * STATE_V + i];
    const double K_i = states[n * STATE_K_i + i];

    // Assign parameters
    const double P_kna = parameters[n * PARAM_P_kna + i];
    const double g_K1 = parameters[n * PARAM_g_K1 + i];
    const double g_Kr = parameters[n * PARAM_g_Kr + i];
    const double g_Ks = parameters[n * PARAM_g_Ks + i];
    const double g_Na = parameters[n * PARAM_g_Na + i];
    const double g_bna = parameters[n * PARAM_g_bna + i];
    const double g_CaL = parameters[n * PARAM_g_CaL + i];
    const double g_bca = parameters[n * PARAM_g_bca + i];
    const double g_to = parameters[n * PARAM_g_to + i];
    const double K_mNa = parameters[n * PARAM_K_mNa + i];
    const double K_mk = parameters[n * PARAM_K_mk + i];
    const double P_NaK = parameters[n * PARAM_P_NaK + i];
    const double K_NaCa = parameters[n * PARAM_K_NaCa + i];
    const double K_sat = parameters[n * PARAM_K_sat + i];
    const double Km_Ca = parameters[n * PARAM_Km_Ca + i];
    const double Km_Nai = parameters[n * PARAM_Km_Nai + i];
    const double alpha = parameters[n * PARAM_alpha + i];
    const double gamma = parameters[n * PARAM_gamma + i];
    const double K_pCa = parameters[n * PARAM_K_pCa + i];
    const double g_pCa = parameters[n * PARAM_g_pCa + i];
    const double g_pK = parameters[n * PARAM_g_pK + i];
    const double Buf_c = parameters[n * PARAM_Buf_c + i];
    const double Buf_sr = parameters[n * PARAM_Buf_sr + i];
    const double Buf_ss = parameters[n * PARAM_Buf_ss + i];
    const double Ca_o = parameters[n * PARAM_Ca_o + i];
    const double EC = parameters[n * PARAM_EC + i];
    const double K_buf_c = parameters[n * PARAM_K_buf_c + i];
    const double K_buf_sr = parameters[n * PARAM_K_buf_sr + i];
    const double K_buf_ss = parameters[n * PARAM_K_buf_ss + i];
    const double K_up = parameters[n * PARAM_K_up + i];
    const double V_leak = parameters[n * PARAM_V_leak + i];
    const double V_rel = parameters[n * PARAM_V_rel + i];
    const double V_sr = parameters[n * PARAM_V_sr + i];
    const double V_ss = parameters[n * PARAM_V_ss + i];
    const double V_xfer = parameters[n * PARAM_V_xfer + i];
    const double Vmax_up = parameters[n * PARAM_Vmax_up + i];
    const double k1_prime = parameters[n * PARAM_k1_prime + i];
    const double k2_prime = parameters[n * PARAM_k2_prime + i];
    const double k3 = parameters[n * PARAM_k3 + i];
    const double k4 = parameters[n * PARAM_k4 + i];
    const double max_sr = parameters[n * PARAM_max_sr + i];
    const double min_sr = parameters[n * PARAM_min_sr + i];
    const double Na_o = parameters[n * PARAM_Na_o + i];
    const double Cm = parameters[n * PARAM_Cm + i];
    const double F = parameters[n * PARAM_F + i];
    const double R = parameters[n * PARAM_R + i];
    const double T = parameters[n * PARAM_T + i];
    const double V_c = parameters[n * PARAM_V_c + i];
    const double stim_amplitude = parameters[n * PARAM_stim_amplitude + i];
    const double stim_duration = parameters[n * PARAM_stim_duration + i];
    const double stim_period = parameters[n * PARAM_stim_period + i];
    const double stim_start = parameters[n * PARAM_stim_start + i];
    const double K_o = parameters[n * PARAM_K_o + i];

    // Expressions for the Reversal potentials component
    const double E_Na = R*T*std::log(Na_o/Na_i)/F;
    const double E_K = R*T*std::log(K_o/K_i)/F;
    const double E_Ks = R*T*std::log((K_o + Na_o*P_kna)/(P_kna*Na_i + K_i))/F;
    const double E_Ca = 0.5*R*T*std::log(Ca_o/Ca_i)/F;

    // Expressions for the Inward rectifier potassium current component
    const double alpha_K1 = 0.1/(1. + 6.14421235332821e-6*std::exp(0.06*V -
          0.06*E_K));
    const double beta_K1 = (0.367879441171442*std::exp(0.1*V - 0.1*E_K) +
        3.06060402008027*std::exp(0.0002*V - 0.0002*E_K))/(1. + std::exp(0.5*E_K
          - 0.5*V));
    const double xK1_inf = alpha_K1/(alpha_K1 + beta_K1);
    const double i_K1 = 0.430331482911935*g_K1*std::sqrt(K_o)*(-E_K + V)*xK1_inf;

    // Expressions for the Rapid time dependent potassium current component
    const double i_Kr = 0.430331482911935*g_Kr*std::sqrt(K_o)*(-E_K + V)*Xr1*Xr2;

    // Expressions for the Xr1 gate component
    const double xr1_inf = 1.0/(1. + std::exp(-26./7. - V/7.));
    const double alpha_xr1 = 450./(1. + std::exp(-9./2. - V/10.));
    const double beta_xr1 = 6./(1. +
        13.5813245225782*std::exp(0.0869565217391304*V));
    const double tau_xr1 = alpha_xr1*beta_xr1;
    const double dXr1_dt = (-Xr1 + xr1_inf)/tau_xr1;
    const double dXr1_dt_linearized = -1./tau_xr1;
    states[n * STATE_Xr1 + i] = (std::fabs(dXr1_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dXr1_dt_linearized))*dXr1_dt/dXr1_dt_linearized : dt*dXr1_dt)
      + Xr1;

    // Expressions for the Xr2 gate component
    const double xr2_inf = 1.0/(1. + std::exp(11./3. + V/24.));
    const double alpha_xr2 = 3./(1. + std::exp(-3. - V/20.));
    const double beta_xr2 = 1.12/(1. + std::exp(-3. + V/20.));
    const double tau_xr2 = alpha_xr2*beta_xr2;
    const double dXr2_dt = (-Xr2 + xr2_inf)/tau_xr2;
    const double dXr2_dt_linearized = -1./tau_xr2;
    states[n * STATE_Xr2 + i] = (std::fabs(dXr2_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dXr2_dt_linearized))*dXr2_dt/dXr2_dt_linearized : dt*dXr2_dt)
      + Xr2;

    // Expressions for the Slow time dependent potassium current component
    const double i_Ks = g_Ks*(Xs*Xs)*(-E_Ks + V);

    // Expressions for the Xs gate component
    const double xs_inf = 1.0/(1. + std::exp(-5./14. - V/14.));
    const double alpha_xs = 1400./std::sqrt(1. + std::exp(5./6. - V/6.));
    const double beta_xs = 1.0/(1. + std::exp(-7./3. + V/15.));
    const double tau_xs = 80. + alpha_xs*beta_xs;
    const double dXs_dt = (-Xs + xs_inf)/tau_xs;
    const double dXs_dt_linearized = -1./tau_xs;
    states[n * STATE_Xs + i] = (std::fabs(dXs_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dXs_dt_linearized))*dXs_dt/dXs_dt_linearized : dt*dXs_dt) +
      Xs;

    // Expressions for the Fast sodium current component
    const double i_Na = g_Na*(m*m*m)*(-E_Na + V)*h*j;

    // Expressions for the m gate component
    const double m_inf = 1.0/((1. +
          0.00184221158116513*std::exp(-0.110741971207087*V))*(1. +
          0.00184221158116513*std::exp(-0.110741971207087*V)));
    const double alpha_m = 1.0/(1. + std::exp(-12. - V/5.));
    const double beta_m = 0.1/(1. + std::exp(7. + V/5.)) + 0.1/(1. +
        std::exp(-1./4. + V/200.));
    const double tau_m = alpha_m*beta_m;
    const double dm_dt = (-m + m_inf)/tau_m;
    const double dm_dt_linearized = -1./tau_m;
    states[n * STATE_m + i] = (std::fabs(dm_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dm_dt_linearized))*dm_dt/dm_dt_linearized : dt*dm_dt) + m;

    // Expressions for the h gate component
    const double h_inf = 1.0/((1. +
          15212.5932856544*std::exp(0.134589502018843*V))*(1. +
          15212.5932856544*std::exp(0.134589502018843*V)));
    const double alpha_h = (V < -40. ?
        4.43126792958051e-7*std::exp(-0.147058823529412*V) : 0.);
    const double beta_h = (V < -40. ? 310000.*std::exp(0.3485*V) +
        2.7*std::exp(0.079*V) : 0.77/(0.13 +
          0.0497581410839387*std::exp(-0.0900900900900901*V)));
    const double tau_h = 1.0/(alpha_h + beta_h);
    const double dh_dt = (-h + h_inf)/tau_h;
    const double dh_dt_linearized = -1./tau_h;
    states[n * STATE_h + i] = (std::fabs(dh_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dh_dt_linearized))*dh_dt/dh_dt_linearized : dt*dh_dt) + h;

    // Expressions for the j gate component
    const double j_inf = 1.0/((1. +
          15212.5932856544*std::exp(0.134589502018843*V))*(1. +
          15212.5932856544*std::exp(0.134589502018843*V)));
    const double alpha_j = (V < -40. ? (37.78 + V)*(-25428.*std::exp(0.2444*V)
          - 6.948e-6*std::exp(-0.04391*V))/(1. + 50262745825.954*std::exp(0.311*V))
        : 0.);
    const double beta_j = (V < -40. ? 0.02424*std::exp(-0.01052*V)/(1. +
          0.00396086833990426*std::exp(-0.1378*V)) : 0.6*std::exp(0.057*V)/(1. +
          0.0407622039783662*std::exp(-0.1*V)));
    const double tau_j = 1.0/(alpha_j + beta_j);
    const double dj_dt = (-j + j_inf)/tau_j;
    const double dj_dt_linearized = -1./tau_j;
    states[n * STATE_j + i] = (std::fabs(dj_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dj_dt_linearized))*dj_dt/dj_dt_linearized : dt*dj_dt) + j;

    // Expressions for the Sodium background current component
    const double i_b_Na = g_bna*(-E_Na + V);

    // Expressions for the L_type Ca current component
    const double V_eff = (std::fabs(-15. + V) < 0.01 ? 0.01 : -15. + V);
    const double i_CaL = 4.*g_CaL*(F*F)*(-Ca_o +
        0.25*Ca_ss*std::exp(2.*F*V_eff/(R*T)))*V_eff*d*f*f2*fCass/(R*T*(-1. +
          std::exp(2.*F*V_eff/(R*T))));

    // Expressions for the d gate component
    const double d_inf = 1.0/(1. +
        0.344153786865412*std::exp(-0.133333333333333*V));
    const double alpha_d = 0.25 + 1.4/(1. + std::exp(-35./13. - V/13.));
    const double beta_d = 1.4/(1. + std::exp(1. + V/5.));
    const double gamma_d = 1.0/(1. + std::exp(5./2. - V/20.));
    const double tau_d = alpha_d*beta_d + gamma_d;
    const double dd_dt = (-d + d_inf)/tau_d;
    const double dd_dt_linearized = -1./tau_d;
    states[n * STATE_d + i] = (std::fabs(dd_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dd_dt_linearized))*dd_dt/dd_dt_linearized : dt*dd_dt) + d;

    // Expressions for the f gate component
    const double f_inf = 1.0/(1. + std::exp(20./7. + V/7.));
    const double tau_f = 20. + 180./(1. + std::exp(3. + V/10.)) + 200./(1. +
        std::exp(13./10. - V/10.)) + 1102.5*std::exp(-((27. + V)*(27. + V))/225.);
    const double df_dt = (-f + f_inf)/tau_f;
    const double df_dt_linearized = -1./tau_f;
    states[n * STATE_f + i] = (std::fabs(df_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*df_dt_linearized))*df_dt/df_dt_linearized : dt*df_dt) + f;

    // Expressions for the F2 gate component
    const double f2_inf = 0.33 + 0.67/(1. + std::exp(5. + V/7.));
    const double tau_f2 = 31./(1. + std::exp(5./2. - V/10.)) + 80./(1. +
        std::exp(3. + V/10.)) + 562.*std::exp(-((27. + V)*(27. + V))/240.);
    const double df2_dt = (-f2 + f2_inf)/tau_f2;
    const double df2_dt_linearized = -1./tau_f2;
    states[n * STATE_f2 + i] = (std::fabs(df2_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*df2_dt_linearized))*df2_dt/df2_dt_linearized : dt*df2_dt) +
      f2;

    // Expressions for the FCass gate component
    const double fCass_inf = 0.4 + 0.6/(1. + 400.0*(Ca_ss*Ca_ss));
    const double tau_fCass = 2. + 80./(1. + 400.0*(Ca_ss*Ca_ss));
    const double dfCass_dt = (-fCass + fCass_inf)/tau_fCass;
    const double dfCass_dt_linearized = -1./tau_fCass;
    states[n * STATE_fCass + i] = (std::fabs(dfCass_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dfCass_dt_linearized))*dfCass_dt/dfCass_dt_linearized :
        dt*dfCass_dt) + fCass;

    // Expressions for the Calcium background current component
    const double i_b_Ca = g_bca*(-E_Ca + V);

    // Expressions for the Transient outward current component
    const double i_to = g_to*(-E_K + V)*r*s;

    // Expressions for the s gate component
    const double s_inf = 1.0/(1. + std::exp(4. + V/5.));
    const double tau_s = 3. + 5./(1. + std::exp(-4. + V/5.)) +
      85.*std::exp(-((45. + V)*(45. + V))/320.);
    const double ds_dt = (-s + s_inf)/tau_s;
    const double ds_dt_linearized = -1./tau_s;
    states[n * STATE_s + i] = (std::fabs(ds_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*ds_dt_linearized))*ds_dt/ds_dt_linearized : dt*ds_dt) + s;

    // Expressions for the r gate component
    const double r_inf = 1.0/(1. + std::exp(10./3. - V/6.));
    const double tau_r = 0.8 + 9.5*std::exp(-((40. + V)*(40. + V))/1800.);
    const double dr_dt = (-r + r_inf)/tau_r;
    const double dr_dt_linearized = -1./tau_r;
    states[n * STATE_r + i] = (std::fabs(dr_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dr_dt_linearized))*dr_dt/dr_dt_linearized : dt*dr_dt) + r;

    // Expressions for the Sodium potassium pump current component
    const double i_NaK = K_o*P_NaK*Na_i/((K_mNa + Na_i)*(K_mk + K_o)*(1. +
          0.0353*std::exp(-F*V/(R*T)) + 0.1245*std::exp(-0.1*F*V/(R*T))));

    // Expressions for the Sodium calcium exchanger current component
    const double i_NaCa =
      K_NaCa*(Ca_o*(Na_i*Na_i*Na_i)*std::exp(F*gamma*V/(R*T)) -
          alpha*(Na_o*Na_o*Na_o)*Ca_i*std::exp(F*(-1. + gamma)*V/(R*T)))/((1. +
            K_sat*std::exp(F*(-1. + gamma)*V/(R*T)))*(Ca_o +
            Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)));

    // Expressions for the Calcium pump current component
    const double i_p_Ca = g_pCa*Ca_i/(K_pCa + Ca_i);

    // Expressions for the Potassium pump current component
    const double i_p_K = g_pK*(-E_K + V)/(1. +
        65.4052157419383*std::exp(-0.167224080267559*V));

    // Expressions for the Calcium dynamics component
    const double i_up = Vmax_up/(1. + (K_up*K_up)/(Ca_i*Ca_i));
    const double i_leak = V_leak*(-Ca_i + Ca_SR);
    const double i_xfer = V_xfer*(-Ca_i + Ca_ss);
    const double kcasr = max_sr - (max_sr - min_sr)/(1. + (EC*EC)/(Ca_SR*Ca_SR));
    const double Ca_i_bufc = 1.0/(1. + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c
            + Ca_i)));
    const double Ca_sr_bufsr = 1.0/(1. + Buf_sr*K_buf_sr/((K_buf_sr +
            Ca_SR)*(K_buf_sr + Ca_SR)));
    const double Ca_ss_bufss = 1.0/(1. + Buf_ss*K_buf_ss/((K_buf_ss +
            Ca_ss)*(K_buf_ss + Ca_ss)));
    const double dCa_i_dt = (V_sr*(-i_up + i_leak)/V_c - Cm*(-2.*i_NaCa +
          i_b_Ca + i_p_Ca)/(2.*F*V_c) + i_xfer)*Ca_i_bufc;
    const double dCa_i_bufc_dCa_i = 2.*Buf_c*K_buf_c/(((1. +
            Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i)))*(1. +
            Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i))))*((K_buf_c +
            Ca_i)*(K_buf_c + Ca_i)*(K_buf_c + Ca_i)));
    const double di_NaCa_dCa_i = -K_NaCa*alpha*(Na_o*Na_o*Na_o)*std::exp(F*(-1.
          + gamma)*V/(R*T))/((1. + K_sat*std::exp(F*(-1. + gamma)*V/(R*T)))*(Ca_o +
            Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)));
    const double di_up_dCa_i = 2.*Vmax_up*(K_up*K_up)/(((1. +
            (K_up*K_up)/(Ca_i*Ca_i))*(1. +
            (K_up*K_up)/(Ca_i*Ca_i)))*(Ca_i*Ca_i*Ca_i));
    const double di_p_Ca_dCa_i = g_pCa/(K_pCa + Ca_i) - g_pCa*Ca_i/((K_pCa +
          Ca_i)*(K_pCa + Ca_i));
    const double dE_Ca_dCa_i = -0.5*R*T/(F*Ca_i);
    const double dCa_i_dt_linearized = (-V_xfer + V_sr*(-V_leak -
          di_up_dCa_i)/V_c - Cm*(-2.*di_NaCa_dCa_i - g_bca*dE_Ca_dCa_i +
            di_p_Ca_dCa_i)/(2.*F*V_c))*Ca_i_bufc + (V_sr*(-i_up + i_leak)/V_c -
            Cm*(-2.*i_NaCa + i_b_Ca + i_p_Ca)/(2.*F*V_c) + i_xfer)*dCa_i_bufc_dCa_i;
    states[n * STATE_Ca_i + i] = Ca_i + (std::fabs(dCa_i_dt_linearized) > 1.0e-8 ?
        (-1.0 + std::exp(dt*dCa_i_dt_linearized))*dCa_i_dt/dCa_i_dt_linearized :
        dt*dCa_i_dt);
    const double k1 = k1_prime/kcasr;
    const double k2 = k2_prime*kcasr;
    const double O = (Ca_ss*Ca_ss)*R_prime*k1/(k3 + (Ca_ss*Ca_ss)*k1);
    const double dR_prime_dt = k4*(1. - R_prime) - Ca_ss*R_prime*k2;
    const double dR_prime_dt_linearized = -k4 - Ca_ss*k2;
    states[n * STATE_R_prime + i] = (std::fabs(dR_prime_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dR_prime_dt_linearized))*dR_prime_dt/dR_prime_dt_linearized :
        dt*dR_prime_dt) + R_prime;
    const double i_rel = V_rel*(-Ca_ss + Ca_SR)*O;
    const double dCa_SR_dt = (-i_leak - i_rel + i_up)*Ca_sr_bufsr;
    const double dkcasr_dCa_SR = -2.*(EC*EC)*(max_sr - min_sr)/(((1. +
            (EC*EC)/(Ca_SR*Ca_SR))*(1. + (EC*EC)/(Ca_SR*Ca_SR)))*(Ca_SR*Ca_SR*Ca_SR));
    const double dCa_sr_bufsr_dCa_SR = 2.*Buf_sr*K_buf_sr/(((1. +
            Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR)))*(1. +
            Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR))))*((K_buf_sr +
            Ca_SR)*(K_buf_sr + Ca_SR)*(K_buf_sr + Ca_SR)));
    const double di_rel_dO = V_rel*(-Ca_ss + Ca_SR);
    const double dk1_dkcasr = -k1_prime/(kcasr*kcasr);
    const double dO_dk1 = (Ca_ss*Ca_ss)*R_prime/(k3 + (Ca_ss*Ca_ss)*k1) -
      std::pow(Ca_ss, 4.)*R_prime*k1/((k3 + (Ca_ss*Ca_ss)*k1)*(k3 +
            (Ca_ss*Ca_ss)*k1));
    const double di_rel_dCa_SR = V_rel*O + V_rel*(-Ca_ss +
        Ca_SR)*dO_dk1*dk1_dkcasr*dkcasr_dCa_SR;
    const double dCa_SR_dt_linearized = (-V_leak - di_rel_dCa_SR -
        dO_dk1*di_rel_dO*dk1_dkcasr*dkcasr_dCa_SR)*Ca_sr_bufsr + (-i_leak - i_rel
          + i_up)*dCa_sr_bufsr_dCa_SR;
    states[n * STATE_Ca_SR + i] = Ca_SR + (std::fabs(dCa_SR_dt_linearized) > 1.0e-8 ?
        (-1.0 + std::exp(dt*dCa_SR_dt_linearized))*dCa_SR_dt/dCa_SR_dt_linearized
        : dt*dCa_SR_dt);
    const double dCa_ss_dt = (V_sr*i_rel/V_ss - V_c*i_xfer/V_ss -
        Cm*i_CaL/(2.*F*V_ss))*Ca_ss_bufss;
    const double dO_dCa_ss = -2.*(Ca_ss*Ca_ss*Ca_ss)*(k1*k1)*R_prime/((k3 +
          (Ca_ss*Ca_ss)*k1)*(k3 + (Ca_ss*Ca_ss)*k1)) + 2.*Ca_ss*R_prime*k1/(k3 +
          (Ca_ss*Ca_ss)*k1);
    const double di_rel_dCa_ss = -V_rel*O + V_rel*(-Ca_ss + Ca_SR)*dO_dCa_ss;
    const double dCa_ss_bufss_dCa_ss = 2.*Buf_ss*K_buf_ss/(((1. +
            Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss)))*(1. +
            Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss))))*((K_buf_ss +
            Ca_ss)*(K_buf_ss + Ca_ss)*(K_buf_ss + Ca_ss)));
    const double di_CaL_dCa_ss =
      1.0*g_CaL*(F*F)*V_eff*d*std::exp(2.*F*V_eff/(R*T))*f*f2*fCass/(R*T*(-1. +
            std::exp(2.*F*V_eff/(R*T))));
    const double dCa_ss_dt_linearized = (V_sr*(dO_dCa_ss*di_rel_dO +
          di_rel_dCa_ss)/V_ss - V_c*V_xfer/V_ss -
        Cm*di_CaL_dCa_ss/(2.*F*V_ss))*Ca_ss_bufss + (V_sr*i_rel/V_ss -
        V_c*i_xfer/V_ss - Cm*i_CaL/(2.*F*V_ss))*dCa_ss_bufss_dCa_ss;
    states[n * STATE_Ca_ss + i] = Ca_ss + (std::fabs(dCa_ss_dt_linearized) > 1.0e-8 ?
        (-1.0 + std::exp(dt*dCa_ss_dt_linearized))*dCa_ss_dt/dCa_ss_dt_linearized
        : dt*dCa_ss_dt);

    // Expressions for the Sodium dynamics component
    const double dNa_i_dt = Cm*(-i_Na - i_b_Na - 3.*i_NaCa - 3.*i_NaK)/(F*V_c);
    const double dE_Na_dNa_i = -R*T/(F*Na_i);
    const double di_NaCa_dNa_i =
      3.*Ca_o*K_NaCa*(Na_i*Na_i)*std::exp(F*gamma*V/(R*T))/((1. +
            K_sat*std::exp(F*(-1. + gamma)*V/(R*T)))*(Ca_o +
            Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)));
    const double di_Na_dE_Na = -g_Na*(m*m*m)*h*j;
    const double di_NaK_dNa_i = K_o*P_NaK/((K_mNa + Na_i)*(K_mk + K_o)*(1. +
          0.0353*std::exp(-F*V/(R*T)) + 0.1245*std::exp(-0.1*F*V/(R*T)))) -
      K_o*P_NaK*Na_i/(((K_mNa + Na_i)*(K_mNa + Na_i))*(K_mk + K_o)*(1. +
            0.0353*std::exp(-F*V/(R*T)) + 0.1245*std::exp(-0.1*F*V/(R*T))));
    const double dNa_i_dt_linearized = Cm*(-3.*di_NaCa_dNa_i - 3.*di_NaK_dNa_i
        + g_bna*dE_Na_dNa_i - dE_Na_dNa_i*di_Na_dE_Na)/(F*V_c);
    states[n * STATE_Na_i + i] = Na_i + (std::fabs(dNa_i_dt_linearized) > 1.0e-8 ?
        (-1.0 + std::exp(dt*dNa_i_dt_linearized))*dNa_i_dt/dNa_i_dt_linearized :
        dt*dNa_i_dt);

    // Expressions for the Membrane component
    const double i_Stim = (t - stim_period*std::floor(t/stim_period) <=
        stim_duration + stim_start && t - stim_period*std::floor(t/stim_period)
        >= stim_start ? -stim_amplitude : 0.);
    const double dV_dt = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK -
      i_Stim - i_b_Ca - i_b_Na - i_p_Ca - i_p_K - i_to;
    const double dalpha_K1_dV = -3.68652741199693e-8*std::exp(0.06*V -
        0.06*E_K)/((1. + 6.14421235332821e-6*std::exp(0.06*V - 0.06*E_K))*(1. +
            6.14421235332821e-6*std::exp(0.06*V - 0.06*E_K)));
    const double di_CaL_dV_eff = 4.*g_CaL*(F*F)*(-Ca_o +
        0.25*Ca_ss*std::exp(2.*F*V_eff/(R*T)))*d*f*f2*fCass/(R*T*(-1. +
          std::exp(2.*F*V_eff/(R*T)))) - 8.*g_CaL*(F*F*F)*(-Ca_o +
        0.25*Ca_ss*std::exp(2.*F*V_eff/(R*T)))*V_eff*d*std::exp(2.*F*V_eff/(R*T))*f*f2*fCass/((R*R)*(T*T)*((-1.
            + std::exp(2.*F*V_eff/(R*T)))*(-1. + std::exp(2.*F*V_eff/(R*T))))) +
        2.0*g_CaL*(F*F*F)*Ca_ss*V_eff*d*std::exp(2.*F*V_eff/(R*T))*f*f2*fCass/((R*R)*(T*T)*(-1.
              + std::exp(2.*F*V_eff/(R*T))));
    const double di_Ks_dV = g_Ks*(Xs*Xs);
    const double di_p_K_dV = g_pK/(1. +
        65.4052157419383*std::exp(-0.167224080267559*V)) +
      10.9373270471469*g_pK*(-E_K + V)*std::exp(-0.167224080267559*V)/((1. +
            65.4052157419383*std::exp(-0.167224080267559*V))*(1. +
            65.4052157419383*std::exp(-0.167224080267559*V)));
    const double di_to_dV = g_to*r*s;
    const double dxK1_inf_dbeta_K1 = -alpha_K1/((alpha_K1 + beta_K1)*(alpha_K1 +
          beta_K1));
    const double dxK1_inf_dalpha_K1 = 1.0/(alpha_K1 + beta_K1) -
      alpha_K1/((alpha_K1 + beta_K1)*(alpha_K1 + beta_K1));
    const double dbeta_K1_dV = (0.000612120804016053*std::exp(0.0002*V -
          0.0002*E_K) + 0.0367879441171442*std::exp(0.1*V - 0.1*E_K))/(1. +
          std::exp(0.5*E_K - 0.5*V)) + 0.5*(0.367879441171442*std::exp(0.1*V -
            0.1*E_K) + 3.06060402008027*std::exp(0.0002*V -
              0.0002*E_K))*std::exp(0.5*E_K - 0.5*V)/((1. + std::exp(0.5*E_K -
                  0.5*V))*(1. + std::exp(0.5*E_K - 0.5*V)));
    const double di_K1_dV = 0.430331482911935*g_K1*std::sqrt(K_o)*xK1_inf +
      0.430331482911935*g_K1*std::sqrt(K_o)*(-E_K +
          V)*(dalpha_K1_dV*dxK1_inf_dalpha_K1 + dbeta_K1_dV*dxK1_inf_dbeta_K1);
    const double dV_eff_dV = (std::fabs(-15. + V) < 0.01 ? 0. : 1.);
    const double di_Na_dV = g_Na*(m*m*m)*h*j;
    const double di_Kr_dV = 0.430331482911935*g_Kr*std::sqrt(K_o)*Xr1*Xr2;
    const double di_NaK_dV = K_o*P_NaK*(0.0353*F*std::exp(-F*V/(R*T))/(R*T) +
        0.01245*F*std::exp(-0.1*F*V/(R*T))/(R*T))*Na_i/((K_mNa + Na_i)*(K_mk +
          K_o)*((1. + 0.0353*std::exp(-F*V/(R*T)) +
              0.1245*std::exp(-0.1*F*V/(R*T)))*(1. + 0.0353*std::exp(-F*V/(R*T)) +
              0.1245*std::exp(-0.1*F*V/(R*T)))));
    const double di_K1_dxK1_inf = 0.430331482911935*g_K1*std::sqrt(K_o)*(-E_K +
        V);
    const double di_NaCa_dV =
      K_NaCa*(Ca_o*F*gamma*(Na_i*Na_i*Na_i)*std::exp(F*gamma*V/(R*T))/(R*T) -
          F*alpha*(Na_o*Na_o*Na_o)*(-1. + gamma)*Ca_i*std::exp(F*(-1. +
              gamma)*V/(R*T))/(R*T))/((1. + K_sat*std::exp(F*(-1. +
                  gamma)*V/(R*T)))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) +
                (Na_o*Na_o*Na_o))) - F*K_NaCa*K_sat*(-1. +
                gamma)*(Ca_o*(Na_i*Na_i*Na_i)*std::exp(F*gamma*V/(R*T)) -
                  alpha*(Na_o*Na_o*Na_o)*Ca_i*std::exp(F*(-1. +
                      gamma)*V/(R*T)))*std::exp(F*(-1. + gamma)*V/(R*T))/(R*T*((1. +
                        K_sat*std::exp(F*(-1. + gamma)*V/(R*T)))*(1. + K_sat*std::exp(F*(-1. +
                            gamma)*V/(R*T))))*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) +
                        (Na_o*Na_o*Na_o)));
    const double dV_dt_linearized = -g_bca - g_bna - di_K1_dV - di_Kr_dV -
      di_Ks_dV - di_NaCa_dV - di_NaK_dV - di_Na_dV - di_p_K_dV - di_to_dV -
      (dalpha_K1_dV*dxK1_inf_dalpha_K1 +
       dbeta_K1_dV*dxK1_inf_dbeta_K1)*di_K1_dxK1_inf - dV_eff_dV*di_CaL_dV_eff;
    states[n * STATE_V + i] = (std::fabs(dV_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dV_dt_linearized))*dV_dt/dV_dt_linearized : dt*dV_dt) + V;

    // Expressions for the Potassium dynamics component
    const double dK_i_dt = Cm*(-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to +
        2.*i_NaK)/(F*V_c);
    const double dE_Ks_dK_i = -R*T/(F*(P_kna*Na_i + K_i));
    const double dbeta_K1_dE_K = (-0.000612120804016053*std::exp(0.0002*V -
          0.0002*E_K) - 0.0367879441171442*std::exp(0.1*V - 0.1*E_K))/(1. +
          std::exp(0.5*E_K - 0.5*V)) - 0.5*(0.367879441171442*std::exp(0.1*V -
            0.1*E_K) + 3.06060402008027*std::exp(0.0002*V -
              0.0002*E_K))*std::exp(0.5*E_K - 0.5*V)/((1. + std::exp(0.5*E_K -
                  0.5*V))*(1. + std::exp(0.5*E_K - 0.5*V)));
    const double di_Kr_dE_K = -0.430331482911935*g_Kr*std::sqrt(K_o)*Xr1*Xr2;
    const double dE_K_dK_i = -R*T/(F*K_i);
    const double di_Ks_dE_Ks = -g_Ks*(Xs*Xs);
    const double di_to_dE_K = -g_to*r*s;
    const double dalpha_K1_dE_K = 3.68652741199693e-8*std::exp(0.06*V -
        0.06*E_K)/((1. + 6.14421235332821e-6*std::exp(0.06*V - 0.06*E_K))*(1. +
            6.14421235332821e-6*std::exp(0.06*V - 0.06*E_K)));
    const double di_K1_dE_K = -0.430331482911935*g_K1*std::sqrt(K_o)*xK1_inf +
      0.430331482911935*g_K1*std::sqrt(K_o)*(-E_K +
          V)*(dalpha_K1_dE_K*dxK1_inf_dalpha_K1 + dbeta_K1_dE_K*dxK1_inf_dbeta_K1);
    const double di_p_K_dE_K = -g_pK/(1. +
        65.4052157419383*std::exp(-0.167224080267559*V));
    const double dK_i_dt_linearized =
      Cm*(-(dE_K_dK_i*dalpha_K1_dE_K*dxK1_inf_dalpha_K1 +
            dE_K_dK_i*dbeta_K1_dE_K*dxK1_inf_dbeta_K1)*di_K1_dxK1_inf -
          dE_K_dK_i*di_K1_dE_K - dE_K_dK_i*di_Kr_dE_K - dE_K_dK_i*di_p_K_dE_K -
          dE_K_dK_i*di_to_dE_K - dE_Ks_dK_i*di_Ks_dE_Ks)/(F*V_c);
    states[n * STATE_K_i + i] = K_i + (std::fabs(dK_i_dt_linearized) > 1.0e-8 ? (-1.0 +
          std::exp(dt*dK_i_dt_linearized))*dK_i_dt/dK_i_dt_linearized : dt*dK_i_dt);
  }
}

