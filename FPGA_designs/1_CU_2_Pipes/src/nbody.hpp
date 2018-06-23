//
//  nbody.hpp
//  N-Body
//
//  Authors:
//	Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//	{emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#ifndef __KERNEL__HPP__
#define __KERNEL__HPP__

#include "ap_int.h"

typedef float my_type;
#define AP_ELEM 16
#define PARALLEL_BODY 2
#define BODY_PACK (32*PARALLEL_BODY)
#define AP_ELEM_PACK (AP_ELEM/PARALLEL_BODY)
//#define PIPES 2
#define RED_0 16
#define RED_1 AP_ELEM
//#define ELEM_RATIO (ELEM/AP_ELEM)
#define N_INPUT (N/AP_ELEM)

#define MY_REDUCTION2(num) reduction_##num
#define MY_REDUCTION(num) MY_REDUCTION2(num)

#define TILE_ELEM 120000

#define TILE_SIZE (TILE_ELEM/AP_ELEM)
//#define TILE_STEP (TILE_SIZE/PIPES)
#define FOR_INDEX_TILE (TILE_SIZE*AP_ELEM)
#define FOR_INDEX_TILE_PACK (FOR_INDEX_TILE/PARALLEL_BODY)

//typedef struct {
//	my_type val[ELEM];
//}my_type_48;

typedef ap_uint<512> my_ap_type;
typedef ap_uint<BODY_PACK> my_ap_pack_type;

extern "C" void nbody(ap_uint<512> *p_x, ap_uint<512> *p_y, ap_uint<512> *p_z, ap_uint<512> *a_x, ap_uint<512> *a_y, ap_uint<512> *a_z, ap_uint<512> *c, my_type *EPS_ptr, unsigned int *tiling_factor_ptr);

#endif
