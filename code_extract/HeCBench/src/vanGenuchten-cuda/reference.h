/**
 * @brief       Genuchten conversion of soil moisture - pressure
 *
 * @param      Ksat   Saturated hydraulic conductivity in soil [L/T]
 * @param      psi    Pressure head [L]
 * @param      C      Specific soil moisture capacity [1/L]
 * @param      theta  Soil moisture [-]
 * @param      K      Hydraulic conductivity in soil at moisture theta [L/T]
 * @param[in]  size   Size of the domain
 */
const double alpha   = 0.02;
const double theta_S = 0.45;
const double theta_R = 0.1;
const double n       = 1.8;

void reference (
  const double * Ksat,
  const double * psi,
        double * C,
        double * theta,
        double * K,
  const int size)
{
  double Se, _theta, _psi, lambda, m;

  for (int i = 0; i < size; i++)
  {
    lambda = n - 1.0;
    m = lambda/n;

    // Compute the volumetric moisture content [eqn 21]
    _psi = psi[i] * 100;
    if ( _psi < 0 )
      _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha*(-_psi)),n), m) + theta_R;
    else
      _theta = theta_S;

    theta[i] = _theta;

    // Compute the effective saturation [eqn 2]
    Se = (_theta - theta_R)/(theta_S - theta_R);

    /* . . .Compute the hydraulic conductivity [eqn 8] . . .*/
    K[i] = Ksat[i] * sqrt(Se) * (1.0 - pow( 1.0-pow(Se,1.0/m), m) ) * (1.0 - pow( 1.0-pow( Se, 1.0/m), m ));

    // Compute the specific moisture storage derivative of eqn (21).
    // So we have to calculate C = d(theta)/dh. Then the unit is converted into [1/m].
    if (_psi < 0)
      C[i] = 100 * alpha * n * (1.0/n-1.0)*pow(alpha*abs(_psi), n-1)
        * (theta_R-theta_S) * pow(pow(alpha*abs(_psi), n)+1, 1.0/n-2.0);
    else
      C[i] = 0.0;
  }
}

