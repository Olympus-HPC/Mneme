#include <algorithm>

// write the function alternatively
int maximum3(int a, int b, int c) {
  return std::max(std::max(a,b),c);  
}

// the results are saved in input_itemsets
void nw_host(int *input_itemsets, int *referrence, int max_cols, int penalty)
{
  for(int blk = 1; blk <= (max_cols-1)/BLOCK_SIZE; blk++ )
  {
    for(int b_index_x = 0; b_index_x < blk; ++b_index_x)
    {
      int b_index_y = blk - 1 - b_index_x;
      int input_itemsets_l[(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)] __attribute__ ((aligned (64)));
      int reference_l[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));

      // Copy referrence to local memory
      for (int i = 0; i < BLOCK_SIZE; ++i )
      {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          reference_l[i*BLOCK_SIZE + j] = referrence[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1];
        }
      }

      // Copy input_itemsets to local memory
      for (int i = 0; i < BLOCK_SIZE + 1; ++i )
      {
        for (int j = 0; j < BLOCK_SIZE + 1; ++j)
        {
          input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i) + b_index_x*BLOCK_SIZE +  j];
        }
      }

      // Compute
      for (int i = 1; i < BLOCK_SIZE + 1; ++i )
      {
        for (int j = 1; j < BLOCK_SIZE + 1; ++j)
        {
          input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = maximum3(input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_l[(i - 1)*BLOCK_SIZE + j - 1],
              input_itemsets_l[i*(BLOCK_SIZE + 1) + j - 1] - penalty,
              input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j] - penalty);
        }
      }

      // Copy results to global memory
      for (int i = 0; i < BLOCK_SIZE; ++i )
      {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1] = input_itemsets_l[(i + 1)*(BLOCK_SIZE+1) + j + 1];
        }
      }
    }
  }    

  for (int blk = 2; blk <= (max_cols-1)/BLOCK_SIZE; blk++ )
  {
    for(int b_index_x = blk - 1; b_index_x < (max_cols-1)/BLOCK_SIZE; ++b_index_x)
    {
      int b_index_y = (max_cols-1)/BLOCK_SIZE + blk - 2 - b_index_x;

      int input_itemsets_l[(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)] __attribute__ ((aligned (64)));
      int reference_l[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));

      // Copy referrence to local memory
      for (int i = 0; i < BLOCK_SIZE; ++i )
      {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          reference_l[i*BLOCK_SIZE + j] = referrence[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1];
        }
      }

      // Copy input_itemsets to local memory
      for (int i = 0; i < BLOCK_SIZE + 1; ++i )
      {
        for (int j = 0; j < BLOCK_SIZE + 1; ++j)
        {
          input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i) + b_index_x*BLOCK_SIZE +  j];
        }
      }

      // Compute
      for (int i = 1; i < BLOCK_SIZE + 1; ++i )
      {
        for (int j = 1; j < BLOCK_SIZE + 1; ++j)
        {
          input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = maximum3(input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_l[(i - 1)*BLOCK_SIZE + j - 1],
              input_itemsets_l[i*(BLOCK_SIZE + 1) + j - 1] - penalty,
              input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j] - penalty);
        }
      }

      // Copy results to global memory
      for (int i = 0; i < BLOCK_SIZE; ++i )
      {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1] = input_itemsets_l[(i + 1)*(BLOCK_SIZE+1) + j +1];
        }
      }
    }
  }
}



