inline int reference_timestep(float t, float dt)
{
  return (int)((t + 1e-3f*dt)/dt); 
}

void reference_cobahh (
    float* d_h, float* d_m, float* d_n, float* d_ge,
    float* d_v, float* d_gi, const float* d_lastspike, 
    char* d_not_refractory, const int _N ,
    const float dt,
    const float t,
    const int    _lio_1,
    const float  _lio_2,
    const float  _lio_3,
    const float  _lio_4,
    const float  _lio_5,
    const float  _lio_6,
    const float  _lio_7,
    const float  _lio_8,
    const float  _lio_9,
    const float _lio_10,
    const float _lio_11,
    const float _lio_12,
    const float _lio_13,
    const float _lio_14,
    const float _lio_15,
    const float _lio_16,
    const float _lio_17,
    const float _lio_18,
    const float _lio_19,
    const float _lio_20,
    const float _lio_21,
    const float _lio_22,
    const float _lio_23,
    const float _lio_24,
    const float _lio_25,
    const float _lio_26,
    const float _lio_27,
    const float _lio_28,
    const float _lio_29,
    const float _lio_30,
    const float _lio_31,
    const float _lio_32,
    const float _lio_33)
{
  for (int _idx = 0; _idx < _N; _idx++) {
    float h = d_h[_idx];
    float m = d_m[_idx];
    float n = d_n[_idx];
    float ge = d_ge[_idx];
    float v = d_v[_idx];
    const float lastspike = d_lastspike[_idx];
    float gi = d_gi[_idx];
    char not_refractory;
    not_refractory = reference_timestep(t - lastspike, dt) >= _lio_1;
    const float _BA_h = (_lio_2 * expf(_lio_3 * v))/(((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * v)))) - (_lio_2 * expf(_lio_3 * v)));
    const float _h = (- _BA_h) + ((_BA_h + h) * expf(dt * (((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * v)))) - (_lio_2 * expf(_lio_3 * v)))));
    const float _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))))/(((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))));
    const float _m = (- _BA_m) + ((_BA_m + m) * expf(dt * (((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))))));
    const float _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))))/(((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_21 * expf(_lio_22 * v))));
    const float _n = (- _BA_n) + ((_BA_n + n) * expf(dt * (((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_21 * expf(_lio_22 * v))))));
    const float _ge = _lio_23 * ge;
    const float _BA_v = (_lio_24 + ((((_lio_25 * (n*n*n*n)) + (_lio_26 * (h * (m*m*m)))) + (_lio_27 * ge)) + (_lio_28 * gi)))/((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)));
    const float _v = (- _BA_v) + ((_BA_v + v) * expf(dt * ((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)))));
    const float _gi = _lio_33 * gi;

    d_h[_idx] = _h;
    d_m[_idx] = _m;
    d_n[_idx] = _n;
    d_ge[_idx] = _ge;
    d_v[_idx] = _v;
    d_gi[_idx] = _gi;
    d_not_refractory[_idx] = not_refractory;
  }
}

void neurongroup_stateupdater_host (
    float* h_ge,
    float* h_gi,
    float* h_h,
    float* h_m,
    float* h_n,
    float* h_v,
    float* h_lastspike,
    float* h_dt,
    float* h_t,
     char* h_not_refractory,
    const int _N,
    const int iteration ) 
{
  const float dt = h_dt[0];
  const float t = h_t[0];
  const int    _lio_1 = reference_timestep(0.003, dt);
  const float  _lio_2 = 9.939082f; //1.0f*(0.3291372 * exp(1.0f*(0.055555556 * (-0.063f))/0.001f))/0.001f;
  const float  _lio_3 = -55.555556f; //1.0f*(-0.055555556)/0.001f;
  const float  _lio_4 = 0.00001f; //2980.958 * (0.001f * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float  _lio_5 = -200.0f; //1.0f*(-0.2)/0.001f;
  const float  _lio_6 = -0.02016f; //0.32 * (-0.063f);
  const float  _lio_7 = -0.000001f; //- 0.001f * 0.001f;
  const float  _lio_8 = 0; //25.79034 * ((0.001f * 0.001f) * exp(1.0f*(0.25 * (-0.063f))/0.001f));
  const float  _lio_9 = -250.0f; //1.0f*(-0.25)/0.001f;
  const float _lio_10 = 0.00416f; //4.16 * 0.001f;
  const float _lio_11 = 0.02016f; //(-0.32) * (-0.063f);
  const float _lio_12 = -0.01764f; //0.28 * (-0.063f);
  const float _lio_13 = -0.000001f; //(-1.0) * (0.001f * 0.001f);
  const float _lio_14 = 0.000099f; //0.00033546262 * ((0.001f * 0.001f) * exp(1.0f*((-0.2) * (-0.063f))/0.001f));
  const float _lio_15 = 200.0f; //1.0f*0.2/0.001f;
  const float _lio_16 = 0.0112f; //11.2 * 0.001f;
  const float _lio_17 = -0.002016f; //0.032 * (-0.063f);
  const float _lio_18 = 0; //20.085537 * ((0.001f * 0.001f) * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float _lio_19 = 0.00048f; //0.48 * 0.001f;
  const float _lio_20 = 0.002016f; //(-0.032) * (-0.063f);
  const float _lio_21 = 132.901474f; //1.0f*(0.6420127 * exp(1.0f*(0.025 * (-0.063f))/0.001f))/0.001f;
  const float _lio_22 = -25.0f ;//1.0f*(-0.025)/0.001f;
  const float _lio_23 = expf(-2000.0f*dt);
  const float _lio_24 = -3.0f; //1.0f*((-0.06f) * 1e-08f)/2e-10f;
  const float _lio_25 = -2700.0f; //1.0f*((-0.09f) * 6e-06f)/2e-10f;
  const float _lio_26 = 5000.0f; //1.0f*(0.05f * 2e-05f)/2e-10f;
  const float _lio_27 = 0; // 1.0f*0.0f/2e-10f;
  const float _lio_28 = -400000000.0f; //1.0f*(-0.08f)/2e-10f;
  const float _lio_29 = -50.0f; //0.0 - (1.0f*1e-08f/2e-10f);
  const float _lio_30 = -30000.0f; //1.0f*(- 6e-06f)/2e-10f;
  const float _lio_31 = 100000.0f; //1.0f*2e-05f/2e-10f;
  const float _lio_32 = 5000000000.0f; //1.0f*1.0/2e-10f;
  const float _lio_33 = expf(-100.0f*dt);

  for (int n = 0; n < iteration; n++) {
    reference_cobahh(h_h, h_m, h_n, h_ge, h_v, h_gi, h_lastspike, h_not_refractory, _N, 
        dt,
        t,
        _lio_1,
        _lio_2,
        _lio_3,
        _lio_4,
        _lio_5,
        _lio_6,
        _lio_7,
        _lio_8,
        _lio_9,
        _lio_10,
        _lio_11,
        _lio_12,
        _lio_13,
        _lio_14,
        _lio_15,
        _lio_16,
        _lio_17,
        _lio_18,
        _lio_19,
        _lio_20,
        _lio_21,
        _lio_22,
        _lio_23,
        _lio_24,
        _lio_25,
        _lio_26,
        _lio_27,
        _lio_28,
        _lio_29,
        _lio_30,
        _lio_31,
        _lio_32,
        _lio_33);
  }
}
