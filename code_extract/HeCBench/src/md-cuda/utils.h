template <class T, class posVecType>
inline T distance(const posVecType* position, const int i, const int j);

template <class T>
inline void insertInOrder(std::list<T>& currDist, std::list<int>& currList,
    const int j, const T distIJ, const int maxNeighbors);

template <class T, class posVecType>
inline int buildNeighborList(const int nAtom, const posVecType* position,
    int* neighborList);

template <class T>
inline int populateNeighborList(std::list<T>& currDist,
    std::list<int>& currList, const int j, const int nAtom,
    int* neighborList);

// ********************************************************
// Function: distance
//
// Purpose:
//   Calculates distance squared between two atoms
//
// Arguments:
//   position: atom position information
//   i, j: indexes of the two atoms
//
// Returns:  the computed distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T, class posVecType>
inline T distance(const posVecType* position, const int i, const int j)
{
  posVecType ipos = position[i];
  posVecType jpos = position[j];
  T delx = ipos.x - jpos.x;
  T dely = ipos.y - jpos.y;
  T delz = ipos.z - jpos.z;
  T r2inv = delx * delx + dely * dely + delz * delz;
  return r2inv;
}

// ********************************************************
// Function: insertInOrder
//
// Purpose:
//   Adds atom j to current neighbor list and distance list
//   if it's distance is low enough.
//
// Arguments:
//   currDist: distance between current atom and each of its neighbors in the
//             current list, sorted in ascending order
//   currList: neighbor list for current atom, sorted by distance in asc. order
//   j:        atom to insert into neighbor list
//   distIJ:   distance between current atom and atom J
//   maxNeighbors: max length of neighbor list
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T>
inline void insertInOrder(
  std::list<T>& currDist,
  std::list<int>& currList,
  const int j, 
  const T distIJ,
  const int maxNeighbors)
{

  typename std::list<T>::iterator   it;
  typename std::list<int>::iterator it2;

  it2 = currList.begin();

  T currMax = currDist.back();

  if (distIJ > currMax) return;

  for (it=currDist.begin(); it!=currDist.end(); it++)
  {
    if (distIJ < (*it))
    {
      // Insert into appropriate place in list
      currDist.insert(it,distIJ);
      currList.insert(it2, j);

      // Trim end of list
      currList.resize(maxNeighbors);
      currDist.resize(maxNeighbors);
      return;
    }
    it2++;
  }
}
// ********************************************************
// Function: buildNeighborList
//
// Purpose:
//   Builds the neighbor list structure for all atoms for GPU coalesced reads
//   and counts the number of pairs within the cutoff distance, so
//   the benchmark gets an accurate FLOPS count
//
// Arguments:
//   nAtom:    total number of atoms
//   position: pointer to the atom's position information
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T, class posVecType>
inline int buildNeighborList(const int nAtom, const posVecType* position,
    int* neighborList)
{
  int totalPairs = 0;
  // Build Neighbor List
  // Find the nearest N atoms to each other atom, where N = maxNeighbors
  for (int i = 0; i < nAtom; i++)
  {
    // Current neighbor list for atom i, initialized to -1
    std::list<int>   currList(maxNeighbors, -1);
    // Distance to those neighbors.  We're populating this with the
    // closest neighbors, so initialize to FLT_MAX
    std::list<T> currDist(maxNeighbors, FLT_MAX);

    for (int j = 0; j < nAtom; j++)
    {
      if (i == j) continue; // An atom cannot be its own neighbor

      // Calculate distance and insert in order into the current lists
      T distIJ = distance<T, posVecType>(position, i, j);
      insertInOrder<T>(currDist, currList, j, distIJ, maxNeighbors);
    }
    // We should now have the closest maxNeighbors neighbors and their
    // distances to atom i. Populate the neighbor list data structure
    // for GPU coalesced reads.
    // The populate method returns how many of the maxNeighbors closest
    // neighbors are within the cutoff distance.  This will be used to
    // calculate GFLOPS later.
    totalPairs += populateNeighborList<T>(currDist, currList, i, nAtom,
        neighborList);
  }
  return totalPairs;
}


// ********************************************************
// Function: populateNeighborList
//
// Purpose:
//   Populates the neighbor list structure for a *single* atom for
//   GPU coalesced reads and counts the number of pairs within the cutoff
//   distance, (for current atom) so the benchmark gets an accurate FLOPS count
//
// Arguments:
//   currDist: distance between current atom and each of its maxNeighbors
//             neighbors
//   currList: current list of neighbors
//   i:        current atom
//   nAtom:    total number of atoms
//   neighborList: pointer to neighbor list data structure
//
// Returns:  number of pairs of atoms within cutoff distance
//
// Programmer: Kyle Spafford
// Creation: July 26, 2010
//
// Modifications:
//
// ********************************************************
template <class T>
inline int populateNeighborList(
    std::list<T>& currDist,
    std::list<int>& currList,
    const int i,
    const int nAtom,
    int* neighborList)
{
  int idx = 0;
  int validPairs = 0; // Pairs of atoms closer together than the cutoff

  // Iterate across distance and neighbor list
  typename std::list<T>::iterator distanceIter = currDist.begin();
  for (std::list<int>::iterator neighborIter = currList.begin();
      neighborIter != currList.end(); neighborIter++)
  {
    // Populate packed neighbor list
    neighborList[(idx * nAtom) + i] = *neighborIter;

    // If the distance is less than cutoff, increment valid counter
    if (*distanceIter < cutsq)
      validPairs++;

    // Increment idx and distance iterator
    idx++;
    distanceIter++;
  }
  return validPairs;
}
