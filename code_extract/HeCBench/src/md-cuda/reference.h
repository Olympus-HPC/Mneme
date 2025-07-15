// ****************************************************************************
// Function: checkResults
//
// Purpose:
//   Check device results against cpu results -- this is the CPU equivalent of
//
// Arguments:
//      d_force:   forces calculated on the device
//      position:  positions of atoms
//      neighList: atom neighbor list
//      nAtom:     number of atoms
// Returns:  true if results match, false otherwise
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ****************************************************************************
template <class T, class forceVecType, class posVecType>
bool checkResults(forceVecType* d_force, posVecType *position,
    int *neighList, int nAtom)
{
  T max_error = 0;
  for (int i = 0; i < nAtom; i++)
  {
    posVecType ipos = position[i];
    forceVecType f = zero;
    int j = 0;
    while (j < maxNeighbors)
    {
      int jidx = neighList[j*nAtom + i];

      // Calculate distance
      posVecType jpos = position[jidx];
      // Calculate distance
      T delx = ipos.x - jpos.x;
      T dely = ipos.y - jpos.y;
      T delz = ipos.z - jpos.z;
      T r2inv = delx*delx + dely*dely + delz*delz;

      // If distance is less than cutoff, calculate force
      if (r2inv > 0 && r2inv < cutsq) {

        r2inv = (T)1.0/r2inv;
        T r6inv = r2inv * r2inv * r2inv;
        T force = r2inv*r6inv*(lj1*r6inv - lj2);

        f.x += delx * force;
        f.y += dely * force;
        f.z += delz * force;
      }
      j++;
    }
    // Check the maximum error when the floating-pont results don't exactly match
    assert(std::isnan(d_force[i].x) == false);
    assert(std::isnan(d_force[i].y) == false);
    assert(std::isnan(d_force[i].z) == false);
    T fxe = std::fabs(f.x - d_force[i].x);
    T fye = std::fabs(f.y - d_force[i].y);
    T fze = std::fabs(f.z - d_force[i].z);
    if (fxe > max_error) max_error = fxe;
    if (fye > max_error) max_error = fye;
    if (fze > max_error) max_error = fze;
  }
  std::cout << "Max error between host and device: " << max_error <<"\n";
  return true;
}
