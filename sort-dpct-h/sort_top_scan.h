#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void top_scan(T *isums, const size_t num_work_groups, sycl::nd_item<3> item_ct1,
              T *lmem, T *s_seed)
{

  int lid = item_ct1.get_local_id(2);
  int local_range = item_ct1.get_local_range().get(2);

    if (lid == 0) *s_seed = 0;
  item_ct1.barrier();

  // Decide if this is the last thread that needs to
  // propagate the seed value
  int last_thread = (lid < num_work_groups &&
      (lid+1) == num_work_groups) ? 1 : 0;

  for (int d = 0; d < 16; d++)
  {
    T val = 0;
    // Load each block's count for digit d
    if (lid < num_work_groups)
    {
      val = isums[(num_work_groups * d) + lid];
    }
    // Exclusive scan the counts in local memory
    //FPTYPE res = scanLocalMem(val, lmem, 1);
    int idx = lid;
    lmem[idx] = 0;
    idx += local_range;
    lmem[idx] = val;
    item_ct1.barrier();
    for (int i = 1; i < local_range; i *= 2)
    {
      T t = lmem[idx -  i];
      item_ct1.barrier();
      lmem[idx] += t;
      item_ct1.barrier();
    }
    T res = lmem[idx-1];

    // Write scanned value out to global
    if (lid < num_work_groups)
    {
      isums[(num_work_groups * d) + lid] = res + *s_seed;
    }
    item_ct1.barrier();

    if (last_thread)
    {
      *s_seed += res + val;
    }
    item_ct1.barrier();
  }
}
