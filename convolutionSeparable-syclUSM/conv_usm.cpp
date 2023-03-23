/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
#include "common.h"
#include "conv.h"

using namespace ext::oneapi::experimental::cuda;

// #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>

#define ROWS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define COLUMNS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 8
#define COLUMNS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1
#define COLUMNS_HALO_STEPS 1

class ConvRows {
    public:

        ConvRows(
            float* __restrict__ d_D,
            const float *__restrict__ d_S,   
            const float *__restrict__ d_K, 
            accessor<float, 2, sycl_read_write, access::target::local> l_D,
            const int iW,
            const int iH,
            const int p
        ): d_Dst{d_D}, d_Src{d_S}, d_Kernel{d_K}, imageW{iW}, imageH{iH}, pitch{p}, l_Data{l_D} {}
    
    private:
            float* __restrict__ d_Dst;          // buffer<float,1> &d_Dst,
            const float *__restrict__ d_Src;    // buffer<float,1> &d_Src,
            const float *__restrict__ d_Kernel; // buffer<float,1> &d_Kernel,
            accessor<float, 2, sycl_read_write, access::target::local> l_Data;
            const int imageW;
            const int imageH;
            const int pitch;

    public:
    
    void operator()(nd_item<2> item) const {
        int gidX = item.get_group(1); 
        int gidY = item.get_group(0); 
        int lidX = item.get_local_id(1); 
        int lidY = item.get_local_id(0); 

        //Offset to the left halo edge
        const int baseX = (gidX * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + lidX;
        const int baseY = gidY * ROWS_BLOCKDIM_Y + lidY;

        const float* __restrict__ d_Src_new = d_Src + baseY * pitch + baseX;
        float* __restrict__ d_Dst_new = d_Dst + baseY * pitch + baseX;

        // Load main data
        for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
          l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = d_Src_new[i * ROWS_BLOCKDIM_X];
        }

        // Load left halo
        for (int i = 0; i < ROWS_HALO_STEPS; i++)
          l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] =
              (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src_new[i * ROWS_BLOCKDIM_X] : 0;

        // Load right halo
        for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
             i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
          l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] =
              (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src_new[i * ROWS_BLOCKDIM_X]
                                                     : 0;

        // Compute and store results
        item.barrier(access::fence_space::local_space);
        for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
          float sum = 0;

          for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += d_Kernel[KERNEL_RADIUS - j] *
                   l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X + j];

          d_Dst_new[i * ROWS_BLOCKDIM_X] = sum;
        }
    }
};

void convolutionRows(
    queue &q,
    float* __restrict__ d_Dst,          // buffer<float,1> &d_Dst,
    const float *__restrict__ d_Src,    // buffer<float,1> &d_Src,
    const float *__restrict__ d_Kernel, // buffer<float,1> &d_Kernel,
    const int imageW,
    const int imageH,
    const int pitch)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    range<2> lws(ROWS_BLOCKDIM_Y, ROWS_BLOCKDIM_X);
    range<2> gws(imageH, imageW / ROWS_RESULT_STEPS);

    q.submit([&](handler &cgh) {
      accessor<float, 2, sycl_read_write, access::target::local> 
      l_Data({ROWS_BLOCKDIM_Y, (ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X}, cgh);
      cgh.parallel_for<class conv_rows>(nd_range<2>(gws,lws), ConvRows{d_Dst, d_Src, d_Kernel, l_Data, imageW, imageH, pitch});
    });
}

void convolutionColumns(
    queue &q,
    float *__restrict__ d_Dst,          // buffer<float,1> &d_Dst,
    const float *__restrict__ d_Src,    // buffer<float,1> &d_Src,
    const float *__restrict__ d_Kernel, // buffer<float,1> &d_Kernel,
    const int imageW,
    const int imageH,
    const int pitch)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    range<2> lws(COLUMNS_BLOCKDIM_Y, COLUMNS_BLOCKDIM_X);
    range<2> gws(imageH / COLUMNS_RESULT_STEPS, imageW);


    q.submit([&](handler &cgh)
             {
      accessor<float, 2, sycl_read_write, access::target::local> 
      l_Data({COLUMNS_BLOCKDIM_X, (COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1}, cgh);

      cgh.parallel_for<class conv_cols>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {

        int gidX = item.get_group(1); 
        int gidY = item.get_group(0); 
        int lidX = item.get_local_id(1); 
        int lidY = item.get_local_id(0); 

        //Offset to the upper halo edge
        const int baseX = gidX * COLUMNS_BLOCKDIM_X + lidX;
        const int baseY = (gidY * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + lidY;

        const float *__restrict__ d_Src_new = d_Src + baseY * pitch + baseX;
        float *__restrict__ d_Dst_new = d_Dst + baseY * pitch + baseX;

        //Load main data
        for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = d_Src_new[i * COLUMNS_BLOCKDIM_Y * pitch];

        //Load upper halo
        for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src_new[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

        //Load lower halo
        for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
            l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y]  = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src_new[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

        //Compute and store results
        item.barrier(access::fence_space::local_space);
        for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
            float sum = 0;

            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += d_Kernel[KERNEL_RADIUS - j] * l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y + j];

            d_Dst_new[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
        }
      });
    });
}