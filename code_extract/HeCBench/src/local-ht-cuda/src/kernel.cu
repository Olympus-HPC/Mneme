#include "kernel.hpp"


__device__ int bcast_warp(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    if (laneId == 0)        // Note unused variable for
        value = arg;        // all threads except lane 0
    value = __shfl_sync(0xffffffff, value, 0);   // Synchronize all threads in warp, and get "value" from lane 0
    if (value != arg && laneId == 0)
        printf("Thread %d failed. with val:%d, arg:%d \n", threadIdx.x, value, arg);
    return value;
}

__device__ void print_mer(cstr_type& mer){
   // if(threadIdx.x%32 == 0){
    for(int i = 0; i < mer.length; i++){
        printf("%c",mer.start_ptr[i]);
    }
    printf("\n");
   // }
}

__device__ void cstr_copy(cstr_type& str1, cstr_type& str2){

    for(int i = 0; i < str2.length; i++){
        str1.start_ptr[i] = str2.start_ptr[i];
    }
    str1.length = str2.length;
}

__device__ unsigned hash_func(cstr_type key, uint32_t max_size){
    unsigned hash, i;
    for(hash = i = 0; i < key.length; ++i)
    {
        hash += key.start_ptr[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash%max_size;//(hash & (HT_SIZE - 1));
}
#define MIX(h,k,m) { k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; }

__device__
uint32_t MurmurHashAligned2 (cstr_type key_in, uint32_t max_size)
{
  int len = key_in.length;
  char* key = key_in.start_ptr;
  const uint32_t m = 0x5bd1e995;
  const int r = 24;
  uint32_t seed = 0x3FB0BB5F;

  const unsigned char * data = (const unsigned char *)key;

  uint32_t h = seed ^ len;

  int align = (uint64_t)data & 3;

  if(align && (len >= 4))
  {
    /* Pre-load the temp registers  */

    uint32_t t = 0, d = 0;

    switch(align)
    {
      case 1: t |= data[2] << 16;
      case 2: t |= data[1] << 8;
      case 3: t |= data[0];
    }

    t <<= (8 * align);

    data += 4-align;
    len -= 4-align;

    int sl = 8 * (4-align);
    int sr = 8 * align;

    /* Mix */

    while(len >= 4)
    {
      d = *(uint32_t *)data;
      t = (t >> sr) | (d << sl);

      uint32_t k = t;

      MIX(h,k,m);

      t = d;

      data += 4;
      len -= 4;
    }

    /* Handle leftover data in temp registers  */

    d = 0;

    if(len >= align)
    {
      switch(align)
      {
      case 3: d |= data[2] << 16;
      case 2: d |= data[1] << 8;
      case 1: d |= data[0];
      }

      uint32_t k = (t >> sr) | (d << sl);
      MIX(h,k,m);

      data += align;
      len -= align;

      /* Handle tail bytes  */

      switch(len)
      {
      case 3: h ^= data[2] << 16;
      case 2: h ^= data[1] << 8;
      case 1: h ^= data[0];
          h *= m;
      };
    }
    else
    {
      switch(len)
      {
      case 3: d |= data[2] << 16;
      case 2: d |= data[1] << 8;
      case 1: d |= data[0];
      case 0: h ^= (t >> sr) | (d << sl);
          h *= m;
      }
    }

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h%max_size;
  }
  else
  {
    while(len >= 4)
    {
      uint32_t k = *(uint32_t *)data;

      MIX(h,k,m);

      data += 4;
      len -= 4;
    }

    /* Handle tail bytes  */

    switch(len)
    {
    case 3: h ^= data[2] << 16;
    case 2: h ^= data[1] << 8;
    case 1: h ^= data[0];
        h *= m;
    };

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h%max_size;
  }
}


__device__ void ht_insert(loc_ht* thread_ht, cstr_type kmer_key, MerFreqs mer_val, uint32_t max_size){
    unsigned hash_val = hash_func(kmer_key, max_size);
    //int count = 0; // for debugging
    while(true){
        int if_empty = thread_ht[hash_val].key.length; // length is set to some unimaginable number to indicate if its empty
        if(if_empty == EMPTY){ //the case where there is a key but no val, will not happen

            thread_ht[hash_val].key = kmer_key;
            thread_ht[hash_val].val = mer_val;
            return;
        }
        hash_val = (hash_val +1 ) % max_size;//(hash_val + 1) & (HT_SIZE-1);

    }
}
//overload for bool vals
__device__ void ht_insert(loc_ht_bool* thread_ht, cstr_type kmer_key, bool bool_val, uint32_t max_size){
    unsigned hash_val = hash_func(kmer_key, max_size);
    //int count = 0; // for debugging
    while(true){
        int if_empty = thread_ht[hash_val].key.length; // length is set to some unimaginable number to indicate if its empty
        if(if_empty == EMPTY){ //the case where there is a key but no val, will not happen
            thread_ht[hash_val].key = kmer_key;
            thread_ht[hash_val].val = bool_val;
            return;
        }
        hash_val = (hash_val +1 ) % max_size;//(hash_val + 1) & (HT_SIZE-1);
        //count++; //for debugging

    }
}


__device__ 
loc_ht& ht_get(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size){
    unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
    unsigned orig_hash = hash_val;
    
    while(true){
        if(thread_ht[hash_val].key.length == EMPTY){
            return thread_ht[hash_val];
        }
        else if(thread_ht[hash_val].key == kmer_key){
            //printf("key found, returning\n");// keep this for debugging
            return thread_ht[hash_val];
        }
        hash_val = (hash_val +1 ) %max_size;//hash_val = (hash_val + 1) & (HT_SIZE -1);
        if(hash_val == orig_hash){ // loop till you reach the same starting positions and then return error
            printf("*****end reached, hashtable full*****\n"); // for debugging
            printf("*****end reached, hashtable full*****\n");
            printf("*****end reached, hashtable full*****\n");
           // return loc_ht(cstr_type(NULL,-1), MerFreqs());
        }
    }

}

//overload for bool vals
__device__ 
loc_ht_bool& ht_get(loc_ht_bool* thread_ht, cstr_type kmer_key, uint32_t max_size){
    unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
    unsigned orig_hash = hash_val;
    
    while(true){
        if(thread_ht[hash_val].key.length == EMPTY){
            return thread_ht[hash_val];
        }
        else if(thread_ht[hash_val].key == kmer_key){
            //printf("key found, returning\n");// keep this for debugging
            return thread_ht[hash_val];
        }
        hash_val = (hash_val +1 ) %max_size;//hash_val = (hash_val + 1) & (HT_SIZE -1);
        if(hash_val == orig_hash){ // loop till you reach the same starting positions and then return error
            printf("*****end reached, hashtable full*****\n"); // for debugging
            printf("*****end reached, hashtable full*****\n");
            printf("*****end reached, hashtable full*****\n");
           // return loc_ht(cstr_type(NULL,-1), MerFreqs());
        }
    }

}

__device__ 
loc_ht& ht_get_atomic(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size){
    unsigned hash_val = MurmurHashAligned2(kmer_key, max_size);
    unsigned orig_hash = hash_val;

    while(true){
        int prev = atomicCAS(&thread_ht[hash_val].key.length, EMPTY, kmer_key.length);
        int mask = __match_any_sync(__activemask(), (unsigned long long)&thread_ht[hash_val]); // all the threads in the warp which have same address
        
        if(prev == EMPTY){
            thread_ht[hash_val].key.start_ptr = kmer_key.start_ptr;
            thread_ht[hash_val].val = {.hi_q_exts = {0}, .low_q_exts = {0}, .ext = 0, .count = 0};
        }
        __syncwarp(mask);
        if(prev != EMPTY && thread_ht[hash_val].key == kmer_key){
            //printf("key found, returning\n");// keep this for debugging
            return thread_ht[hash_val];
        }else if (prev == EMPTY){
            return thread_ht[hash_val];
        }
        hash_val = (hash_val +1 ) %max_size;//hash_val = (hash_val + 1) & (HT_SIZE -1);
        if(hash_val == orig_hash){ // loop till you reach the same starting positions and then return error
            printf("*****end reached, hashtable full*****\n"); // for debugging
            printf("*****end reached, hashtable full*****\n");
            printf("*****end reached, hashtable full*****\n");
        }
    }

}
__device__ char walk_mers(loc_ht* thrd_loc_ht, loc_ht_bool* thrd_ht_bool, uint32_t max_ht_size, int& mer_len, cstr_type& mer_walk_temp, cstr_type& longest_walk, cstr_type& walk, const int idx, int max_walk_len){
    char walk_result = 'X';
    #ifdef DEBUG_PRINT_GPU
    int test = 1;
    #endif
    for( int nsteps = 0; nsteps < max_walk_len; nsteps++){
        //check if there is a cycle in graph
        loc_ht_bool &temp_mer_loop = ht_get(thrd_ht_bool, mer_walk_temp, max_walk_len);
        if(temp_mer_loop.key.length == EMPTY){ // if the mer has not been visited, add it to the table and mark visited
            temp_mer_loop.key = mer_walk_temp;
            temp_mer_loop.val = true;
        }else{
            walk_result = 'R'; // if the table already contains this mer then a cycle exits, return the walk with repeat.
            #ifdef DEBUG_PRINT_GPU
            if(idx == test)
                printf("breaking at cycle found, res: %c\n", walk_result);
            #endif
            break;
        }

        loc_ht &temp_mer = ht_get(thrd_loc_ht, mer_walk_temp, max_ht_size);
        if(temp_mer.key.length == EMPTY){//if mer is not found then dead end reached, terminate the walk
            walk_result = 'X';
            #ifdef DEBUG_PRINT_GPU
            if(idx == test)
               printf("breaking at mer not found,res: %c\n", walk_result);
            #endif
            break;
        }
        char ext = temp_mer.val.ext;
        if(ext == 'F' || ext == 'X'){ // if the table points that ext is fork or dead end the terminate the walk
            walk_result = ext;
            #ifdef DEBUG_PRINT_GPU
                if(idx == test){
                    printf("breaking at dead end, res: %c\n", walk_result);
                    printf("Mer Looked up:\n");
                    print_mer(mer_walk_temp);
                    printf("ext:%c\n",temp_mer.val.ext);
                    printf("walk with mer_len:%d\n", mer_len);
                    print_mer(walk);
                }
            #endif
            break;
        }

         #ifdef DEBUG_PRINT_GPU
         if(test == idx){
             printf("Mer Looked up:\n");
             print_mer(mer_walk_temp);
             printf("ext:%c\n",temp_mer.val.ext);
             printf("walk with mer_len:%d\n", mer_len);
            // print_mer(walk);
         }
         #endif

        mer_walk_temp.start_ptr = mer_walk_temp.start_ptr + 1; // increment the mer pointer and append the ext
        mer_walk_temp.start_ptr[mer_walk_temp.length-1] = ext; // walk pointer points at the end of initial mer point.
        if (ext != 0) walk.length++;

        
    }
    
    #ifdef DEBUG_PRINT_GPU
    if(idx == test)
        for (int k = 0; k < max_walk_len; k++) {
            if(thrd_ht_bool[k].key.length != EMPTY){
                printf("from bool ht:\n");
                print_mer(thrd_ht_bool[k].key);
                printf("Bool:%d\n",thrd_ht_bool[k].val);
            }
        }
    #endif
    return walk_result;
}

__device__ 
void count_mers(loc_ht* thrd_loc_ht, char* loc_r_reads, uint32_t max_ht_size, char* loc_r_quals, uint32_t* reads_r_offset, uint32_t& r_rds_cnt, 
uint32_t* rds_count_r_sum, double& loc_ctg_depth, int& mer_len, uint32_t& qual_offset, int64_t& excess_reads, const long int idx){
    const int lane_id = threadIdx.x%32;
    cstr_type read;
    cstr_type qual;
    uint32_t running_sum_len = 0;
    // #ifdef DEBUG_PRINT_GPU
    // int test = 1;
    // if(DEBUG_PRINT_GPU && idx == test)
    //     printf("inside_count_mers, hash size:%d \n", max_ht_size);
    // #endif
    for(int i = 0; i < r_rds_cnt; i++){
        // #ifdef DEBUG_PRINT_GPU
        // if(DEBUG_PRINT_GPU && idx == test)
        //     printf("read loop iter:%d, thread:%d, loop max:%d\n",i, threadIdx.x, r_rds_cnt);
        // #endif
        read.start_ptr = loc_r_reads + running_sum_len;
        qual.start_ptr = loc_r_quals + running_sum_len;
        if(i == 0){
            if(idx == 0){
                read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];
                qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];
                // #ifdef DEBUG_PRINT_GPU
                // if(DEBUG_PRINT_GPU && idx == test)
                //     printf("rds_count_r_sum[idx]:%d, rds_cnt:%d, read_length:%d, thread_id:%d\n",rds_count_r_sum[idx], r_rds_cnt, read.length, threadIdx.x);
                // #endif
                }
            else{  
                // printf("idx:%d, r_rdsx_cnt: %d, i: %d \n", idx, r_rds_cnt, i); 
                // printf("i:%d, rds_count:%d, idx:%d, thread:%d, blk:%d\n", i, r_rds_cnt, idx, threadIdx.x, blockIdx.x); 
                // #ifdef DEBUG_PRINT_GPU
                // if(DEBUG_PRINT_GPU && idx == test)
                //     printf("rds_count_r_sum[idx]:%d,rds_count_r_sum[idx-1]:%d,i:%d, rds_cnt:%d, reads_offset_0:%d, thread:%d\n",rds_count_r_sum[idx], rds_count_r_sum[idx-1],i, r_rds_cnt, read.length, threadIdx.x);
                // #endif
                if(rds_count_r_sum[idx - 1] == 0){
                    read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];
                    qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i];                    
                }else{
                    read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx - 1] -1)];
                    qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx - 1] -1)];
                 }
            }
        }
        else{
            read.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + (i-1)];
            qual.length = reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + i] - reads_r_offset[(rds_count_r_sum[idx] - r_rds_cnt) + (i-1)];
            // #ifdef DEBUG_PRINT_GPU
            // if(DEBUG_PRINT_GPU && idx == test)
            //      printf("rds_count_r_sum[idx]:%d, rds_cnt:%d, reads_offset_0:%d, thread:%d\n",rds_count_r_sum[idx], r_rds_cnt, read.length, threadIdx.x);
            // #endif
                
            }
        // #ifdef DEBUG_PRINT_GPU
        // if(DEBUG_PRINT_GPU && idx == test){
        //     printf("mer_len:%d, read_len:%d\n",mer_len, read.length);
        //     printf("read from idx:%d, thread:%d\n", idx, threadIdx.x);
        //     print_mer(read);
        //   }
        // #endif
        if (mer_len > read.length) // skip the read which is smaller than merlen
            continue;
        int num_mers = read.length - mer_len;
        for( int start = lane_id; start < num_mers; start+=32){
            //TODO: on cpu side add a check that if a certain read contains 'N', 
            cstr_type mer(read.start_ptr + start, mer_len);
            loc_ht &temp_Mer = ht_get_atomic(thrd_loc_ht, mer, max_ht_size);
            
            int ext_pos = start + mer_len;
          //  assert(ext_pos < (int)read.length); // TODO: verify that assert works on gpu, for now commenting it out and replacing with printf
          if(ext_pos >= (int) read.length)
            printf("*********ASSERTION FAILURE IN COUNT_MERS****");
            char ext = read.start_ptr[ext_pos];
            if (ext == 'N') continue; // TODO: why the redundant check?
            int qual_diff = qual.start_ptr[ext_pos] - qual_offset;
            if (qual_diff >= LASSM_MIN_QUAL) temp_Mer.val.low_q_exts.inc(ext, 1);
            if (qual_diff >= LASSM_MIN_HI_QUAL) temp_Mer.val.hi_q_exts.inc(ext, 1);

            //temp_Mer.val.set_ext(loc_ctg_depth);
        }
        __syncwarp();
       running_sum_len += read.length; // right before the for loop ends, update the prev_len to offset next read correctly
    }
    __syncwarp();
    for(auto k = lane_id; k < max_ht_size; k+=32){
        if(thrd_loc_ht[k].key.length != EMPTY){
            thrd_loc_ht[k].val.set_ext(loc_ctg_depth);
        }
    }
    __syncwarp();
    int test = 1;
    #ifdef DEBUG_PRINT_GPU
    if(idx == test){
        if(lane_id == 0)    
            for(int k = 0; k < max_ht_size; k++){
            //{
                if( thrd_loc_ht[k].key.length != EMPTY){
                printf("from ht:\n");
                print_mer(thrd_loc_ht[k].key);
                printf("MerFreq.ext:%c, MerFreq.count:%d\n",thrd_loc_ht[k].val.ext,thrd_loc_ht[k].val.count);
                thrd_loc_ht[k].val.hi_q_exts.print();
                thrd_loc_ht[k].val.low_q_exts.print();
                }
            //}
            }
    }
    #endif
    __syncwarp();
}
//same kernel will be used for right and left walks
__global__ void iterative_walks_kernel(uint32_t* cid, uint32_t* ctg_offsets, char* contigs, char* reads_r, char* quals_r,  uint32_t* reads_r_offset,  uint32_t* rds_count_r_sum, 
double* ctg_depth, loc_ht* global_ht,  uint32_t* prefix_ht, loc_ht_bool* global_ht_bool, int kmer_len, uint32_t max_mer_len_off, uint32_t *term_counts, int64_t num_walks, int64_t max_walk_len, 
int64_t sum_ext, int32_t max_read_size, int32_t max_read_count, uint32_t qual_offset, char* longest_walks, char* mer_walk_temp, uint32_t* final_walk_lens, int tot_ctgs)
{
    const long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const long int warp_id_glb = idx/32;
    const long int lane_id = threadIdx.x%32;
    if(warp_id_glb < tot_ctgs){ // so that no out of bound accesses 
    cstr_type loc_ctg;
    char *loc_r_reads, *loc_r_quals;
    uint32_t r_rds_cnt;
    loc_ht* loc_mer_map;
    uint32_t ht_loc_size;
    loc_ht_bool* loc_bool_map;
    double loc_ctg_depth;
    int64_t excess_reads;
    uint32_t max_ht_size = 0;
    char* longest_walk_loc;
    char* loc_mer_walk_temp;
    #ifdef DEBUG_PRINT_GPU
    int test = 1;
    #endif

    int min_mer_len = LASSM_MIN_KMER_LEN;
    int max_mer_len = LASSM_MAX_KMER_LEN;
    
    //the warp portion is for HT phase only so mapping only reads related data based on warp id
    if(warp_id_glb == 0){
        loc_ctg.start_ptr = contigs;
        loc_ctg.length = ctg_offsets[warp_id_glb];
        loc_bool_map = global_ht_bool + warp_id_glb * max_walk_len;
        longest_walk_loc = longest_walks + warp_id_glb * max_walk_len;
        loc_mer_walk_temp = mer_walk_temp + warp_id_glb * (max_walk_len + max_mer_len_off);
        r_rds_cnt = rds_count_r_sum[warp_id_glb];
        loc_r_reads = reads_r;
        loc_r_quals = quals_r;
        loc_mer_map = global_ht;
        ht_loc_size = prefix_ht[warp_id_glb];
        loc_ctg_depth = ctg_depth[warp_id_glb];
    }else{
        loc_ctg.start_ptr = contigs + ctg_offsets[warp_id_glb-1];
        loc_ctg.length = ctg_offsets[warp_id_glb] - ctg_offsets[warp_id_glb - 1];
        loc_bool_map = global_ht_bool + warp_id_glb * max_walk_len;
        longest_walk_loc = longest_walks + warp_id_glb * max_walk_len;
        loc_mer_walk_temp = mer_walk_temp + warp_id_glb * (max_walk_len + max_mer_len_off);
        loc_ctg_depth = ctg_depth[warp_id_glb];
        r_rds_cnt = rds_count_r_sum[warp_id_glb] - rds_count_r_sum[warp_id_glb - 1];
        if (rds_count_r_sum[warp_id_glb - 1] == 0)
            loc_r_reads = reads_r;
        else
            loc_r_reads = reads_r + reads_r_offset[rds_count_r_sum[warp_id_glb - 1] - 1]; // you want to start from where previous contigs, last read ends.

        if (rds_count_r_sum[warp_id_glb - 1] == 0)
            loc_r_quals = quals_r;
        else
            loc_r_quals = quals_r + reads_r_offset[rds_count_r_sum[warp_id_glb - 1] - 1]; // you want to start from where previous contigs, last read ends.
       
        loc_mer_map = global_ht + prefix_ht[warp_id_glb - 1];
        ht_loc_size = prefix_ht[warp_id_glb] - prefix_ht[warp_id_glb - 1];
    }

    max_ht_size = ht_loc_size;
    max_mer_len = min(max_mer_len, loc_ctg.length);

    cstr_type longest_walk_thread(longest_walk_loc,0);

    //main for loop
    int shift = 0;

    for(int mer_len = kmer_len; mer_len >= min_mer_len && mer_len <= max_mer_len; mer_len += shift){
            // #ifdef DEBUG_PRINT_GPU
            // if(warp_id_glb == test){
            //    printf("GPU: shift:%d, mer_len:%d, min_mer_len:%d, idx:%d, max_mer_len:%d \n", shift, mer_len, min_mer_len, threadIdx.x, max_mer_len);
            //    printf("contig:\n");
            //    print_mer(loc_ctg);
            //    }
            // #endif

    if(warp_id_glb < tot_ctgs){ // all warps within this range can go in execute count mers, for walk_mers only the lane 0 of each warp does the work
            for(uint32_t k = lane_id; k < max_ht_size; k+=32){ // resetting hash table in parallel with warps
                loc_mer_map[k].key.length = EMPTY;
            }
            count_mers(loc_mer_map, loc_r_reads, max_ht_size, loc_r_quals, reads_r_offset, r_rds_cnt, rds_count_r_sum, loc_ctg_depth, mer_len, qual_offset, excess_reads, warp_id_glb);//passing warp_id instead of idx now
            for(uint32_t k = lane_id; k < max_walk_len; k+=32){ // resetting bool map for next go
                loc_bool_map[k].key.length = EMPTY;
            }
        if(lane_id == 0){ // this phase is processed by single thread of a warp
            cstr_type ctg_mer(loc_ctg.start_ptr + (loc_ctg.length - mer_len), mer_len);
            cstr_type loc_mer_walk(loc_mer_walk_temp, 0);
            cstr_copy(loc_mer_walk, ctg_mer);
            cstr_type walk(loc_mer_walk.start_ptr + mer_len, 0);

            // #ifdef DEBUG_PRINT_GPU
            // if(warp_id_glb == test){
            //     printf("read_count:%d, idx:%d\n",r_rds_cnt, warp_id_glb);
            //     printf("mer ctg len:%d mer_walk before:\n",loc_mer_walk.length);
            //     print_mer(loc_mer_walk);

            //    printf("ctg mer:\n");
            //    print_mer(ctg_mer);
            // }
            // #endif

            char walk_res = walk_mers(loc_mer_map, loc_bool_map, max_ht_size, mer_len, loc_mer_walk, longest_walk_thread, walk, warp_id_glb, max_walk_len);
            // #ifdef DEBUG_PRINT_GPU
            // if(warp_id_glb == test){
            //     printf("walk_res:%c, idx:%d\n",walk_res, warp_id_glb);
            //     printf("GPU: walk.len:%d, longest.len:%d, idx:%d\n", walk.length, longest_walk_thread.length, warp_id_glb);
            // }
            // #endif
            //int walk_len = walk.length
            if (walk.length > longest_walk_thread.length){ // this walk is longer than longest then copy it to longest walk
                cstr_copy(longest_walk_thread, walk);
            }
            if (walk_res == 'X') {
               // atomicAdd(&term_counts[0], 1);
                // walk reaches a dead-end, downshift, unless we were upshifting
                if (shift == LASSM_SHIFT_SIZE) 
                    break;
                shift = -LASSM_SHIFT_SIZE;
            }else {
                //if (walk_res == 'F') 
                   // atomicAdd(&term_counts[1], 1);
                //else 
                    //atomicAdd(&term_counts[2], 1);
                // otherwise walk must end with a fork or repeat, so upshift
                if (shift == -LASSM_SHIFT_SIZE){
                    // #ifdef DEBUG_PRINT_GPU
                    // printf("breaking at shift neg:%d\n", shift);
                    // #endif
                    break;
                    }
                if (mer_len > loc_ctg.length){
                    // #ifdef DEBUG_PRINT_GPU
                    // printf("breaking at mer_len too large:%d\n", mer_len);
                    // #endif
                    break;
                }
                shift = LASSM_SHIFT_SIZE;
            }

        }// lane id cond ended
        __syncwarp(FULL_MASK);
        unsigned mask = __activemask();
        unsigned active = mask & 1; // zero if lane 0 has returned
        if(active == 0) break; // return if lane 0 has returned
        shift = bcast_warp(shift);
        }//warp id cond end

    }
    if(lane_id == 0){
    if(longest_walk_thread.length > 0){
        final_walk_lens[warp_id_glb] = longest_walk_thread.length;
        // printf("final longest walk len:%d/n", longest_walk_thread.length);
        // print_mer(longest_walk_thread);
       // atomicAdd(num_walks, 1);
     //   atomicAdd(sum_ext, longest_walk_thread.length);
    }else{
        final_walk_lens[warp_id_glb] = 0;
    }
    }

    #ifdef DEBUG_PRINT_GPU
    if(idx == test){
       // printf("walk:\n");
       // print_mer(walk);
       // printf("walk len:%d\n", walk.length);
       // printf("mer_walk after:\n");
       // print_mer(loc_mer_walk);
       // printf("mer_walk after, len:%d\n", loc_mer_walk.length);
        }
        //printf("walk result:%c\n", walk_res);
    #endif
}//end if to check if idx exceeds contigs
}
