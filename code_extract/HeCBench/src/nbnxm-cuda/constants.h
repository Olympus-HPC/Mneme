
template<std::uint64_t n>
struct StaticLog2
{
  //!< Variable value used for recursive static calculation of Log2(int)
  static const int value = StaticLog2<n / 2>::value + 1;
};
template<>
struct StaticLog2<1>
{
  //!< Base value for recursive static calculation of Log2(int)
  static const int value = 0;
};
template<>
struct StaticLog2<0>
{
  //!< Base value for recursive static calculation of Log2(int)
  static const int value = -1;
};

constexpr unsigned int NBNXN_INTERACTION_MASK_ALL = 0xffffffffU;

static constexpr int c_nbnxnGpuClusterpairSplit = 2;
static constexpr int c_nbnxnGpuNumClusterPerSupercluster = 8;
static constexpr int c_nbnxnGpuClusterSize = 8;
static constexpr int c_clSize = c_nbnxnGpuClusterSize;
constexpr int c_nbnxnGpuJgroupSize = (32 / c_nbnxnGpuNumClusterPerSupercluster);
static constexpr int c_nbnxnGpuExclSize = c_nbnxnGpuClusterSize * c_nbnxnGpuClusterSize 
                                          / c_nbnxnGpuClusterpairSplit;
static constexpr int c_splitClSize = c_clSize / c_nbnxnGpuClusterpairSplit;

constexpr int c_dBoxZ = 1;
constexpr int c_dBoxY = 1;
constexpr int c_dBoxX = 2;
constexpr int c_nBoxZ = 2 * c_dBoxZ + 1;
constexpr int c_nBoxY = 2 * c_dBoxY + 1;
constexpr int c_nBoxX = 2 * c_dBoxX + 1;
constexpr int c_numIvecs = c_nBoxZ * c_nBoxY * c_nBoxX;

constexpr int c_centralShiftIndex = c_numIvecs / 2;
static constexpr unsigned superClInteractionMask = ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);
constexpr float c_nbnxnMinDistanceSquared = 3.82e-07F; // r > 6.2e-4


// atom index is computed as shown in the following code snippet
//  for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i += c_clSize)
//  {
//     const int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj + i;
//     const int ai = ci * c_clSize + tidxi;
//
// As c_nbnxnGpuNumClusterPerSupercluster equals c_nbnxnGpuClusterSize, and c_clSize
// equals c_nbnxnGpuClusterSize, i is always 0 for the specific case
//
// grid size in the z dim
constexpr int grid_z = 3199;
// thread block size in the x and y dims
constexpr int block_x = 8;
constexpr int block_y = 8;
constexpr int NUM_ATOMS = (grid_z * c_nbnxnGpuNumClusterPerSupercluster + block_x) * c_clSize + block_y;

struct nbnxn_im_ei_t
{
  //! The i-cluster interactions mask for 1 warp
  unsigned int imask = 0U;
  //! Index into the exclusion array for 1 warp, default index 0 which means no exclusions
  int excl_ind = 0;
};

typedef struct
{
  //! The 4 j-clusters
  int cj[c_nbnxnGpuJgroupSize];
  //! The i-cluster mask data for 2 warps
  nbnxn_im_ei_t imei[c_nbnxnGpuClusterpairSplit];
} nbnxn_cj4_t;

typedef struct nbnxn_sci
{
  //! Returns the number of j-cluster groups in this entry
  int numJClusterGroups() const { return cj4_ind_end - cj4_ind_start; }

  //! i-super-cluster
  int sci;
  //! Shift vector index plus possible flags
  int shift;
  //! Start index into cj4
  int cj4_ind_start;
  //! End index into cj4
  int cj4_ind_end;
} nbnxn_sci_t;

struct nbnxn_excl_t
{
  //! Constructor, sets no exclusions, so all atom pairs interacting
  nbnxn_excl_t()
  {
    for (unsigned int& pairEntry : pair)
    {
      pairEntry = NBNXN_INTERACTION_MASK_ALL;
    }
  }

  //! Topology exclusion interaction bits per warp
  unsigned int pair[c_nbnxnGpuExclSize];
};
