__global__ void sneaky_snake(
  const uint*__restrict__ F_ReadSeq,
  const uint*__restrict__ F_RefSeq, 
  int*__restrict__ Ftest_Results, 
  const int NumReads,
  const int F_ErrorThreshold)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= NumReads) return;

  uint ReadsPerThread[NBytes];
  uint RefsPerThread[NBytes];

#pragma unroll
  for (int i = 0; i < NBytes; i++)
  {
    ReadsPerThread[i] = F_ReadSeq[tid*8 + i];
    RefsPerThread[i] = F_RefSeq[tid*8 + i];
  }

  /////////////////////////////////////////////////////////////////////////////
  Ftest_Results[tid] = 1;

  uint ReadCompTmp = 0;
  uint RefCompTmp = 0;
  uint DiagonalResult = 0;

  uint ReadTmp1 = 0;
  uint ReadTmp2 = 0;

  uint RefTmp1 = 0;
  uint RefTmp2 = 0;

  uint CornerCase = 0;

  int localCounter= 0;
  int localCounterMax=0;
  int globalCounter = 0;
  int Max_leading_zeros = 0;
  int AccumulatedErrs = 0;

  int Diagonal = 0;
  int ShiftValue = 0;

  int j = 0; //specifying the j-th int that we are reading in each read-ref comparison (can be from 0 to 7)

  while ( (j < 7) && (globalCounter < 200))
  {
    Diagonal = 0;
    RefTmp1 = lsl(RefsPerThread[j], ShiftValue);
    RefTmp2 = lsr(RefsPerThread[j + 1], 32 - ShiftValue);
    ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue);
    ReadTmp2 = lsr(ReadsPerThread[j + 1], 32 - ShiftValue);

    ReadCompTmp = ReadTmp1 | ReadTmp2;
    RefCompTmp = RefTmp1 | RefTmp2;
    DiagonalResult = ReadCompTmp ^ RefCompTmp;
    localCounterMax = __clz(DiagonalResult);

    //////////////////// Upper diagonals /////////////////////

    for(int e = 1; e <= F_ErrorThreshold; e++)
    {
      Diagonal += 1;
      CornerCase = 0;
      if ( (j == 0) && ( (ShiftValue - (2*e)) < 0 ) )
      {
        ReadTmp1 = lsr(ReadsPerThread[j], 2*e - ShiftValue);
        ReadTmp2 = 0;

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;

        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        CornerCase = 0;
        for(int Ci = 0; Ci < (2*e) - ShiftValue; Ci++)
        {
          set_bit(CornerCase, 31 - Ci);
        }

        DiagonalResult  = DiagonalResult | CornerCase;
        localCounter = __clz(DiagonalResult);

      }
      else if ( (ShiftValue - (2*e) ) < 0 )
      {
        ReadTmp1 = lsl(ReadsPerThread[j-1], 32 - (2*e - ShiftValue));
        ReadTmp2 = lsr(ReadsPerThread[j], 2*e - ShiftValue);

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;

        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        localCounter = __clz(DiagonalResult);
      }
      else
      {
        ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue - 2*e);
        ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue - 2*e)) ;

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;

        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        localCounter = __clz(DiagonalResult);
      }
      if (localCounter>localCounterMax)
        localCounterMax=localCounter;
    }

    /*
       sh = shift
       up = upper diagonal
       RC = ReadCompTmp
       FC = RefCompTmp
       D = DiagonalResult
       DN = diagonal
       LC = localCounter
       */

    //////////////////// Lower diagonals /////////////////////

    for(int e = 1; e <= F_ErrorThreshold; e++)
    {
      Diagonal += 1;
      CornerCase = 0;
      if (j<5)
      {
        if ((ShiftValue + 2*e) < 32)
        {
          ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue + 2*e);
          ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue + 2*e));

          ReadCompTmp = ReadTmp1 | ReadTmp2;
          RefCompTmp = RefTmp1 | RefTmp2;

          DiagonalResult = ReadCompTmp ^ RefCompTmp;
          localCounter = __clz(DiagonalResult);
        }
        else
        {
          ReadTmp1 = lsl(ReadsPerThread[j+1], (ShiftValue + 2*e) % 32);
          ReadTmp2 = lsr(ReadsPerThread[j+2], 32 - (ShiftValue + 2*e) % 32);

          ReadCompTmp = ReadTmp1 | ReadTmp2;
          RefCompTmp = RefTmp1 | RefTmp2;

          DiagonalResult = 0xffffffff;//ReadCompTmp ^ RefCompTmp;

          DiagonalResult = ReadCompTmp ^ RefCompTmp;

          localCounter = __clz(DiagonalResult);
        }
      }
      else
      {
        ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue + 2*e);
        ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue + 2*e));

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;
        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        CornerCase = 0;
        if ((globalCounter+32)>200) {

          for(int Ci = globalCounter+32-200; Ci < globalCounter+32-200+2*e; Ci++)
          {
            set_bit(CornerCase, Ci);
          }
        }

        else if ((globalCounter+32)>=(200- (2*e))){

          for(int Ci = 0; Ci < (2*e); Ci++)
          {
            set_bit(CornerCase, Ci);
          }
        }
        DiagonalResult = DiagonalResult | CornerCase;

        localCounter = __clz(DiagonalResult);
      }

      if (localCounter>localCounterMax)
        localCounterMax=localCounter;
    }

    /*
       CC = CornerCase
       sh = shift
       up = upper diagonal
       RC = ReadCompTmp
       FC = RefCompTmp
       D = DiagonalResult
       DN = diagonal
       LC = localCounter
       */

    Max_leading_zeros = 0;

    if ( (j == 6) && ( ((localCounterMax/2)*2) >= 8) )
    {
      Max_leading_zeros = 8;
      break;
    }
    else if((localCounterMax/2*2) > Max_leading_zeros)
    {
      Max_leading_zeros = ((localCounterMax/2)*2);
    }

    if (((Max_leading_zeros/2) < 16) && (j < 5))
    {
      AccumulatedErrs += 1;
    }
    else if ((j == 6) && ((Max_leading_zeros/2) < 4))
    {
      AccumulatedErrs += 1;
    }

    if(AccumulatedErrs > F_ErrorThreshold)
    {
      Ftest_Results[tid] = 0;
      break;
    }

    if(ShiftValue + Max_leading_zeros + 2 >= 32)
    {
      j += 1;
    }

    // ShiftValue_2Ref = (ShiftValue_2Ref + Max_leading_zeros + 2) %32;
    if (Max_leading_zeros == 32)
    {
      globalCounter += Max_leading_zeros;
    }
    else
    {
      ShiftValue = ((ShiftValue + Max_leading_zeros + 2) % 32);
      globalCounter += (Max_leading_zeros + 2);
    }
  }
}

