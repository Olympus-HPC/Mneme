#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void mergeSortPass(const sycl::float4 *input, sycl::float4 *result,
                   const int *constStartAddr, const int threadsPerDiv,
                   const int nrElems, sycl::nd_item<3> item_ct1)
{

  const int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  // The division to work on
  int division = gid / threadsPerDiv;
  if(division >= DIVISIONS) return;
  // The block within the division
  int int_gid = gid - division * threadsPerDiv;
  int Astart = constStartAddr[division] + int_gid * nrElems;

  int Bstart = Astart + nrElems/2;
  sycl::float4 *resStart = &(result[Astart]);

  if(Astart >= constStartAddr[division + 1])
    return;
  if(Bstart >= constStartAddr[division + 1]){
    for(int i=0; i<(constStartAddr[division + 1] - Astart); i++)
    {
      resStart[i] = input[Astart + i];
    }
    return;
  }

  int aidx = 0;
  int bidx = 0;
  int outidx = 0;
  sycl::float4 a, b;
  a = input[Astart + aidx];
  b = input[Bstart + bidx];

  while(true)//aidx < nrElems/2)// || (bidx < nrElems/2  && (Bstart + bidx < constEndAddr[division])))
  {
    /**
     * For some reason, it's faster to do the texture fetches here than
     * after the merge
     */
    sycl::float4 nextA = input[Astart + aidx + 1];
    sycl::float4 nextB = input[Bstart + bidx + 1];

    sycl::float4 na = getLowest(a, b);
    sycl::float4 nb = getHighest(a, b);
    a = sortElem(na);
    b = sortElem(nb);
    // Now, a contains the lowest four elements, sorted
    resStart[outidx++] = a;

    bool elemsLeftInA;
    bool elemsLeftInB;

    elemsLeftInA = (aidx + 1 < nrElems/2); // Astart + aidx + 1 is allways less than division border
    elemsLeftInB = (bidx + 1 < nrElems/2) && (Bstart + bidx + 1 < constStartAddr[division + 1]);

    if(elemsLeftInA){
      if(elemsLeftInB){
        float nextA_t = nextA.x();
        float nextB_t = nextB.x();
        if(nextA_t < nextB_t) { aidx += 1; a = nextA; }
        else { bidx += 1;  a = nextB; }
      }
      else {
        aidx += 1; a = nextA;
      }
    }
    else {
      if(elemsLeftInB){
        bidx += 1;  a = nextB;
      }
      else {
        break;
      }
    }

  }
  resStart[outidx++] = b;
}

