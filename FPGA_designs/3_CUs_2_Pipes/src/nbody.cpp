//
//  nbody.cpp
//  N-Body
//
//  Authors:
//	Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//	{emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#include "string.h"
#include "math.h"
#include "utils.hpp"
#include "nbody.hpp"

void core(hls::stream<my_ap_type> &stream_x, hls::stream<my_ap_type> &stream_y, hls::stream<my_ap_type> &stream_z,
		hls::stream<my_ap_type> &stream_c, my_type x_val, my_type y_val, my_type z_val, my_type EPS,
		hls::stream<my_type> &out_x, hls::stream<my_type> &out_y, hls::stream<my_type> &out_z){

	my_type acc_x[RED_1];
	my_type acc_y[RED_1];
	my_type acc_z[RED_1];

	my_type tot_acc_x[RED_0];// = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	my_type tot_acc_y[RED_0];// = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	my_type tot_acc_z[RED_0];// = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	set_zero:for(int i = 0; i < RED_0; i++){
#pragma HLS PIPELINE
		tot_acc_x[i] = 0;
		tot_acc_y[i] = 0;
		tot_acc_z[i] = 0;
	}

	central:for(int j = 0; j < TILE_SIZE; j++){
	#pragma HLS PIPELINE
		my_ap_type tmp_x_val = stream_x.read();
		my_ap_type tmp_y_val = stream_y.read();
		my_ap_type tmp_z_val = stream_z.read();
		my_ap_type tmp_c_val = stream_c.read();
		unsigned int idx = j % RED_0;
	parallel0:for(int k = 0; k < AP_ELEM; k++){

			unsigned int lower_range = 32*k;
			unsigned int upper_range = 32*(k + 1)-1;
			unsigned int stream_x_val_tmp = tmp_x_val.range(upper_range, lower_range);
			my_type stream_x_val = *((my_type *)&stream_x_val_tmp);
			unsigned int stream_y_val_tmp = tmp_y_val.range(upper_range, lower_range);
			my_type stream_y_val = *((my_type *)&stream_y_val_tmp);
			unsigned int stream_z_val_tmp = tmp_z_val.range(upper_range, lower_range);
			my_type stream_z_val = *((my_type *)&stream_z_val_tmp);
			unsigned int stream_c_val_tmp = tmp_c_val.range(upper_range, lower_range);
			my_type stream_c_val = *((my_type *)&stream_c_val_tmp);


			my_type rx = stream_x_val - x_val;
			my_type ry = stream_y_val - y_val;
			my_type rz = stream_z_val - z_val;

			my_type dd = (rx*rx + ry*ry) + (rz*rz + EPS);

			my_type d = dd * sqrtf(dd);

			my_type s = stream_c_val / d;
			acc_x[k] = rx * s;
			acc_y[k] = ry * s;
			acc_z[k] = rz * s;

		}

		tot_acc_x[idx] += MY_REDUCTION(AP_ELEM)<my_type, RED_1>(acc_x);
		tot_acc_y[idx] += MY_REDUCTION(AP_ELEM)<my_type, RED_1>(acc_y);
		tot_acc_z[idx] += MY_REDUCTION(AP_ELEM)<my_type, RED_1>(acc_z);
	}

	reduction_16<my_type, RED_0>(out_x, tot_acc_x);
	reduction_16<my_type, RED_0>(out_y, tot_acc_y);
	reduction_16<my_type, RED_0>(out_z, tot_acc_z);


}

void core_start(ap_uint<512> p_x[TILE_SIZE], ap_uint<512> p_y[TILE_SIZE], ap_uint<512> p_z[TILE_SIZE], ap_uint<512> c[TILE_SIZE],
		my_ap_pack_type x_val, my_ap_pack_type y_val, my_ap_pack_type z_val, my_type EPS, hls::stream<my_ap_pack_type> &out_x, hls::stream<my_ap_pack_type> &out_y, hls::stream<my_ap_pack_type> &out_z){

#pragma HLS DATAFLOW

	hls::stream<my_ap_type> stream_x[PARALLEL_BODY];
#pragma HLS STREAM variable=stream_x depth=1 dim=1
	hls::stream<my_ap_type> stream_y[PARALLEL_BODY];
#pragma HLS STREAM variable=stream_y depth=1 dim=1
	hls::stream<my_ap_type> stream_z[PARALLEL_BODY];
#pragma HLS STREAM variable=stream_z depth=1 dim=1
	hls::stream<my_ap_type> stream_c[PARALLEL_BODY];
#pragma HLS STREAM variable=stream_c depth=1 dim=1

	hls::stream<my_type> stream_out_x[PARALLEL_BODY];
#pragma HLS STREAM variable=stream_out_x depth=1 dim=1
	hls::stream<my_type> stream_out_y[PARALLEL_BODY];
#pragma HLS STREAM variable=stream_out_y depth=1 dim=1
	hls::stream<my_type> stream_out_z[PARALLEL_BODY];
#pragma HLS STREAM variable=stream_out_z depth=1 dim=1

	Axi2_MultiStreams<my_ap_type, TILE_SIZE, PARALLEL_BODY>(stream_x, p_x);
	Axi2_MultiStreams<my_ap_type, TILE_SIZE, PARALLEL_BODY>(stream_y, p_y);
	Axi2_MultiStreams<my_ap_type, TILE_SIZE, PARALLEL_BODY>(stream_z, p_z);
	Axi2_MultiStreams<my_ap_type, TILE_SIZE, PARALLEL_BODY>(stream_c, c);

	my_type x_val_vect[PARALLEL_BODY];
#pragma HLS ARRAY_PARTITION variable=x_val_vect complete dim=1
	my_type y_val_vect[PARALLEL_BODY];
#pragma HLS ARRAY_PARTITION variable=y_val_vect complete dim=1
	my_type z_val_vect[PARALLEL_BODY];
#pragma HLS ARRAY_PARTITION variable=z_val_vect complete dim=1

	for(int i = 0; i < PARALLEL_BODY; i++){
#pragma HLS UNROLL

		unsigned int lower_range = 32*i;
		unsigned int upper_range = 32*(i + 1)-1;

		unsigned int x_val_tmp_uint = x_val.range(upper_range, lower_range);
		x_val_vect[i] = *((my_type *)&x_val_tmp_uint);
		unsigned int y_val_tmp_uint = y_val.range(upper_range, lower_range);
		y_val_vect[i] = *((my_type *)&y_val_tmp_uint);
		unsigned int z_val_tmp_uint = z_val.range(upper_range, lower_range);
		z_val_vect[i] = *((my_type *)&z_val_tmp_uint);

	}

	for(int i = 0; i < PARALLEL_BODY; i++){
#pragma HLS UNROLL
		core(stream_x[i], stream_y[i], stream_z[i], stream_c[i], x_val_vect[i], y_val_vect[i], z_val_vect[i], EPS, stream_out_x[i], stream_out_y[i], stream_out_z[i]);
	}

	my_type out_val_x_vect[PARALLEL_BODY];
	my_type out_val_y_vect[PARALLEL_BODY];
	my_type out_val_z_vect[PARALLEL_BODY];

	for(int i = 0; i < PARALLEL_BODY; i++){
	#pragma HLS UNROLL

		out_val_x_vect[i] = stream_out_x[i].read();
		out_val_y_vect[i] = stream_out_y[i].read();
		out_val_z_vect[i] = stream_out_z[i].read();
	}

	my_ap_pack_type out_val_x;
	my_ap_pack_type out_val_y;
	my_ap_pack_type out_val_z;

	for(int i = 0; i < PARALLEL_BODY; i++){
#pragma HLS UNROLL

		unsigned int lower_range = 32*i;
		unsigned int upper_range = 32*(i + 1)-1;

		my_type out_val_x_tmp = out_val_x_vect[i];
		my_type out_val_y_tmp = out_val_y_vect[i];
		my_type out_val_z_tmp = out_val_z_vect[i];

		out_val_x.range(upper_range, lower_range) = *((unsigned int *)&out_val_x_tmp);
		out_val_y.range(upper_range, lower_range) = *((unsigned int *)&out_val_y_tmp);
		out_val_z.range(upper_range, lower_range) = *((unsigned int *)&out_val_z_tmp);

	}

	out_x.write(out_val_x);
	out_y.write(out_val_y);
	out_z.write(out_val_z);

}

void n_body_cu(unsigned int tiling_factor_val,
		unsigned int* outer_tile_start_ptr, unsigned int* outer_tile_end_ptr,
		my_type EPS_val, ap_uint<512>* p_x, ap_uint<512>* p_y,
		ap_uint<512>* p_z, ap_uint<512>* c, ap_uint<512>* a_x,
		ap_uint<512>* a_y, ap_uint<512>* a_z) {

	ap_uint<512> outer_x[TILE_SIZE];
	ap_uint<512> outer_y[TILE_SIZE];
	ap_uint<512> outer_z[TILE_SIZE];
	//	const int blocks = PIPES;
	ap_uint<512> inner_x[TILE_SIZE];
	//#pragma HLS ARRAY_PARTITION variable=inner_x block factor=blocks dim=1
	ap_uint<512> inner_y[TILE_SIZE];
	//#pragma HLS ARRAY_PARTITION variable=inner_y block factor=blocks dim=1
	ap_uint<512> inner_z[TILE_SIZE];
	//#pragma HLS ARRAY_PARTITION variable=inner_z block factor=blocks dim=1
	ap_uint<512> inner_c[TILE_SIZE];
	//#pragma HLS ARRAY_PARTITION variable=inner_c block factor=blocks dim=1
	ap_uint<512> local_a_x[TILE_SIZE];
	ap_uint<512> local_a_y[TILE_SIZE];
	ap_uint<512> local_a_z[TILE_SIZE];
	unsigned int tiling_factor = tiling_factor_val;
	unsigned int outer_tile_start = outer_tile_start_ptr[0];
	unsigned int outer_tile_end = outer_tile_end_ptr[0];
	my_type EPS = EPS_val;
	//tiling_factor = 1;
	outerTile: for (int t0 = outer_tile_start; t0 < outer_tile_end; t0++) {
		unsigned int outer_tile = t0 * TILE_SIZE;
		unsigned int outer_tile_acc = (t0 - outer_tile_start) * TILE_SIZE;
		memcpy(outer_x, (const ap_uint<512>*) (p_x) + outer_tile,
				(TILE_SIZE) * sizeof(ap_uint<512> ));
		memcpy(outer_y, (const ap_uint<512>*) (p_y) + outer_tile,
				(TILE_SIZE) * sizeof(ap_uint<512> ));
		memcpy(outer_z, (const ap_uint<512>*) (p_z) + outer_tile,
				(TILE_SIZE) * sizeof(ap_uint<512> ));
		myMemset<ap_uint<512>, TILE_SIZE>(local_a_x, 0);
		myMemset<ap_uint<512>, TILE_SIZE>(local_a_y, 0);
		myMemset<ap_uint<512>, TILE_SIZE>(local_a_z, 0);
		innerTile: for (int t1 = 0; t1 < tiling_factor; t1++) {
			unsigned int inner_tile = t1 * TILE_SIZE;
			memcpy(inner_x, (const ap_uint<512>*) (p_x) + inner_tile,
					(TILE_SIZE) * sizeof(ap_uint<512> ));
			memcpy(inner_y, (const ap_uint<512>*) (p_y) + inner_tile,
					(TILE_SIZE) * sizeof(ap_uint<512> ));
			memcpy(inner_z, (const ap_uint<512>*) (p_z) + inner_tile,
					(TILE_SIZE) * sizeof(ap_uint<512> ));
			memcpy(inner_c, (const ap_uint<512>*) (c) + inner_tile,
					(TILE_SIZE) * sizeof(ap_uint<512> ));
			computation: for (int i = 0; i < FOR_INDEX_TILE_PACK; i++) {
				unsigned int i_index = i / AP_ELEM_PACK;
				unsigned int e_index = i % AP_ELEM_PACK;
				unsigned int lower_range = BODY_PACK * e_index;
				unsigned int upper_range = BODY_PACK * (e_index + 1) - 1;
				my_ap_pack_type x_val = outer_x[i_index].range(upper_range,
						lower_range);
				my_ap_pack_type y_val = outer_y[i_index].range(upper_range,
						lower_range);
				my_ap_pack_type z_val = outer_z[i_index].range(upper_range,
						lower_range);
				my_ap_pack_type acc_x_tmp = local_a_x[i_index].range(
						upper_range, lower_range);
				my_ap_pack_type acc_y_tmp = local_a_y[i_index].range(
						upper_range, lower_range);
				my_ap_pack_type acc_z_tmp = local_a_z[i_index].range(
						upper_range, lower_range);
				hls::stream<my_ap_pack_type> out_x;
#pragma HLS STREAM variable=out_x depth=1 dim=1
				hls::stream<my_ap_pack_type> out_y;
#pragma HLS STREAM variable=out_y depth=1 dim=1
				hls::stream<my_ap_pack_type> out_z;
#pragma HLS STREAM variable=out_z depth=1 dim=1
				core_start(inner_x, inner_y, inner_z, inner_c, x_val, y_val,
						z_val, EPS, out_x, out_y, out_z);
				my_ap_pack_type x_out_tmp = out_x.read();
				my_ap_pack_type y_out_tmp = out_y.read();
				my_ap_pack_type z_out_tmp = out_z.read();
				my_ap_pack_type x_out_val;
				my_ap_pack_type y_out_val;
				my_ap_pack_type z_out_val;
				for (int b = 0; b < PARALLEL_BODY; b++) {
					unsigned int lower_range = 32 * b;
					unsigned int upper_range = 32 * (b + 1) - 1;
					unsigned int x_out_tmp_uint = x_out_tmp.range(upper_range,
							lower_range);
					my_type x_out_tmp_my_type = *((my_type*) (&x_out_tmp_uint));
					unsigned int y_out_tmp_uint = y_out_tmp.range(upper_range,
							lower_range);
					my_type y_out_tmp_my_type = *((my_type*) (&y_out_tmp_uint));
					unsigned int z_out_tmp_uint = z_out_tmp.range(upper_range,
							lower_range);
					my_type z_out_tmp_my_type = *((my_type*) (&z_out_tmp_uint));
					unsigned int acc_x_tmp_uint = acc_x_tmp.range(upper_range,
							lower_range);
					my_type acc_x_tmp_my_type = *((my_type*) (&acc_x_tmp_uint));
					unsigned int acc_y_tmp_uint = acc_y_tmp.range(upper_range,
							lower_range);
					my_type acc_y_tmp_my_type = *((my_type*) (&acc_y_tmp_uint));
					unsigned int acc_z_tmp_uint = acc_z_tmp.range(upper_range,
							lower_range);
					my_type acc_z_tmp_my_type = *((my_type*) (&acc_z_tmp_uint));
					my_type x_output_acc = x_out_tmp_my_type
							+ acc_x_tmp_my_type;
					my_type y_output_acc = y_out_tmp_my_type
							+ acc_y_tmp_my_type;
					my_type z_output_acc = z_out_tmp_my_type
							+ acc_z_tmp_my_type;
					x_out_val.range(upper_range, lower_range) =
							*((unsigned int*) (&x_output_acc));
					y_out_val.range(upper_range, lower_range) =
							*((unsigned int*) (&y_output_acc));
					z_out_val.range(upper_range, lower_range) =
							*((unsigned int*) (&z_output_acc));
				}
				//				my_type x_out_val = x_out_tmp + acc_x_val;
				//				my_type y_out_val = y_out_tmp + acc_y_val;
				//				my_type z_out_val = z_out_tmp + acc_z_val;
				//				local_a_x[i_index].range(upper_range, lower_range) = *((unsigned int *)&x_out_val);
				//				local_a_y[i_index].range(upper_range, lower_range) = *((unsigned int *)&y_out_val);
				//				local_a_z[i_index].range(upper_range, lower_range) = *((unsigned int *)&z_out_val);
				local_a_x[i_index].range(upper_range, lower_range) = x_out_val;
				local_a_y[i_index].range(upper_range, lower_range) = y_out_val;
				local_a_z[i_index].range(upper_range, lower_range) = z_out_val;
			}
		}
		memcpy(a_x + outer_tile_acc, (const ap_uint<512>*) (local_a_x),
				(TILE_SIZE) * sizeof(ap_uint<512> ));
		memcpy(a_y + outer_tile_acc, (const ap_uint<512>*) (local_a_y),
				(TILE_SIZE) * sizeof(ap_uint<512> ));
		memcpy(a_z + outer_tile_acc, (const ap_uint<512>*) (local_a_z),
				(TILE_SIZE) * sizeof(ap_uint<512> ));
	}
}

void nbody(ap_uint<512> *p_x_0, ap_uint<512> *p_y_0, ap_uint<512> *p_z_0, ap_uint<512> *a_x_0, ap_uint<512> *a_y_0, ap_uint<512> *a_z_0, ap_uint<512> *c_0, unsigned int *outer_tile_start_ptr_0, unsigned int *outer_tile_end_ptr_0,
		ap_uint<512> *p_x_1, ap_uint<512> *p_y_1, ap_uint<512> *p_z_1, ap_uint<512> *a_x_1, ap_uint<512> *a_y_1, ap_uint<512> *a_z_1, ap_uint<512> *c_1, unsigned int *outer_tile_start_ptr_1, unsigned int *outer_tile_end_ptr_1,
		ap_uint<512> *p_x_2, ap_uint<512> *p_y_2, ap_uint<512> *p_z_2, ap_uint<512> *a_x_2, ap_uint<512> *a_y_2, ap_uint<512> *a_z_2, ap_uint<512> *c_2, unsigned int *outer_tile_start_ptr_2, unsigned int *outer_tile_end_ptr_2,
		my_type *EPS_ptr, unsigned int *tiling_factor_ptr){

#pragma HLS INTERFACE m_axi port=p_x_0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=p_y_0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=p_z_0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=a_x_0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=a_y_0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=a_z_0 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=c_0   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=outer_tile_start_ptr_0 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=outer_tile_end_ptr_0 offset=slave bundle=gmem1

#pragma HLS INTERFACE m_axi port=p_x_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=p_y_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=p_z_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=a_x_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=a_y_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=a_z_1 offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=c_1   offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=outer_tile_start_ptr_1 offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=outer_tile_end_ptr_1 offset=slave bundle=gmem3

#pragma HLS INTERFACE m_axi port=p_x_2 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=p_y_2 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=p_z_2 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=a_x_2 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=a_y_2 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=a_z_2 offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=c_2   offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=outer_tile_start_ptr_2 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=outer_tile_end_ptr_2 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=EPS_ptr offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=tiling_factor_ptr offset=slave bundle=gmem7

#pragma HLS INTERFACE s_axilite register port=p_x_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=p_y_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=p_z_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_x_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_y_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_z_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=c_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=outer_tile_start_ptr_0 bundle=control
#pragma HLS INTERFACE s_axilite register port=outer_tile_end_ptr_0 bundle=control

#pragma HLS INTERFACE s_axilite register port=p_x_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=p_y_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=p_z_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_x_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_y_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_z_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=c_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=outer_tile_start_ptr_1 bundle=control
#pragma HLS INTERFACE s_axilite register port=outer_tile_end_ptr_1 bundle=control

#pragma HLS INTERFACE s_axilite register port=p_x_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=p_y_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=p_z_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_x_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_y_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=a_z_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=c_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=outer_tile_start_ptr_2 bundle=control
#pragma HLS INTERFACE s_axilite register port=outer_tile_end_ptr_2 bundle=control

#pragma HLS INTERFACE s_axilite register port=EPS_ptr bundle=control
#pragma HLS INTERFACE s_axilite register port=tiling_factor_ptr bundle=control

#pragma HLS INTERFACE s_axilite register port=return bundle=control

	my_type EPS = EPS_ptr[0];
	unsigned int tiling_factor = tiling_factor_ptr[0];

	n_body_cu(tiling_factor, outer_tile_start_ptr_0, outer_tile_end_ptr_0,
					EPS, p_x_0, p_y_0, p_z_0, c_0, a_x_0, a_y_0, a_z_0);
	n_body_cu(tiling_factor, outer_tile_start_ptr_1, outer_tile_end_ptr_1,
					EPS, p_x_1, p_y_1, p_z_1, c_1, a_x_1, a_y_1, a_z_1);
	n_body_cu(tiling_factor, outer_tile_start_ptr_2, outer_tile_end_ptr_2,
				EPS, p_x_2, p_y_2, p_z_2, c_2, a_x_2, a_y_2, a_z_2);
}

