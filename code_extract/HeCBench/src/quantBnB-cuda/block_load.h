/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Operations for reading linear tiles of data into the GPU thread block.
 */

#pragma once

#include <iterator>
#include <type_traits>
#include "utils.h"

template <
    typename        InputT,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorT>
__device__ __forceinline__ void LoadDirectBlocked(
    int             linear_tid,
    InputIteratorT  block_itr,
    InputT          (&items)[ITEMS_PER_THREAD])
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        items[ITEM] = block_itr[(linear_tid * ITEMS_PER_THREAD) + ITEM];
    }
}


template <
    typename        InputT,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorT>
__device__ __forceinline__ void LoadDirectBlocked(
    int             linear_tid,
    InputIteratorT  block_itr,
    InputT          (&items)[ITEMS_PER_THREAD],
    int             valid_items)                ///< [in] Number of valid items to load
{

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if ((linear_tid * ITEMS_PER_THREAD) + ITEM < valid_items)
        {
            items[ITEM] = block_itr[(linear_tid * ITEMS_PER_THREAD) + ITEM];
        }
    }
}


template <
    typename        InputT,
    typename        DefaultT,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorT>
__device__ __forceinline__ void LoadDirectBlocked(
    int             linear_tid,
    InputIteratorT  block_itr,
    InputT          (&items)[ITEMS_PER_THREAD],
    int             valid_items,
    DefaultT        oob_default)
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        items[ITEM] = oob_default;

    LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
}


//-----------------------------------------------------------------------------
// Generic BlockLoad abstraction
//-----------------------------------------------------------------------------

enum BlockLoadAlgorithm
{
    BLOCK_LOAD_DIRECT
};


template <
    typename            InputT,
    int                 BLOCK_DIM_X,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  ALGORITHM           = BLOCK_LOAD_DIRECT,
    int                 BLOCK_DIM_Y         = 1,
    int                 BLOCK_DIM_Z         = 1>
class BlockLoad
{
private:

    /******************************************************************************
     * Constants and typed definitions
     ******************************************************************************/

    enum
    {
        /// The thread block size in threads
        BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
    };


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /// Load helper
    template <BlockLoadAlgorithm _POLICY, int DUMMY>
    struct LoadInternal;

    /**
     * BLOCK_LOAD_DIRECT specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType TempStorage;

        /// Linear thread-id
        int linear_tid;

        /// Constructor
        __device__ __forceinline__ LoadInternal(
            TempStorage &/*temp_storage*/,
            int linear_tid)
        :
            linear_tid(linear_tid)
        {}

        /// Load a linear segment of items from memory
        template <typename InputIteratorT>
        __device__ __forceinline__ void Load(
            InputIteratorT  block_itr,
            InputT          (&items)[ITEMS_PER_THREAD])
        {
            LoadDirectBlocked(linear_tid, block_itr, items);
        }

        /// Load a linear segment of items from memory, guarded by range
        template <typename InputIteratorT>
        __device__ __forceinline__ void Load(
            InputIteratorT  block_itr,
            InputT          (&items)[ITEMS_PER_THREAD],
            int             valid_items)
        {
            LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
        }

        /// Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
        template <typename InputIteratorT, typename DefaultT>
        __device__ __forceinline__ void Load(
            InputIteratorT  block_itr,
            InputT          (&items)[ITEMS_PER_THREAD],
            int             valid_items,
            DefaultT        oob_default)
        {
            LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
        }

    };


    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal load implementation to use
    typedef LoadInternal<ALGORITHM, 0> InternalLoad;


    /// Shared memory storage layout type
    typedef typename InternalLoad::TempStorage _TempStorage;


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Thread reference to shared storage
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;

public:

    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    __device__ __forceinline__ BlockLoad()
    :
        temp_storage(PrivateStorage()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    __device__ __forceinline__ BlockLoad(
        TempStorage &temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}




    /******************************************************************//**
     * \name Data movement
     *********************************************************************/


    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
        InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
        InputT          (&items)[ITEMS_PER_THREAD]) ///< [out] Data to load
    {
        InternalLoad(temp_storage, linear_tid).Load(block_itr, items);
    }


    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
        InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
        InputT          (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
        int             valid_items)                ///< [in] Number of valid items to load
    {
        InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items);
    }


    template <typename InputIteratorT, typename DefaultT>
    __device__ __forceinline__ void Load(
        InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
        InputT          (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
        int             valid_items,                ///< [in] Number of valid items to load
        DefaultT        oob_default)                ///< [in] Default value to assign out-of-bound items
    {
        InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items, oob_default);
    }

};

