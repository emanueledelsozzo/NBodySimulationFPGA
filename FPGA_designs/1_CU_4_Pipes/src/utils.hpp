//
//  utils.hpp
//  N-Body
//
//  Authors:
//	Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//	{emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#pragma once

#include "hls_stream.h"

#define BIT_WIDTH 32
#define BIT_WIDTH_1 (BIT_WIDTH - 1)


template <
	typename data_type,
	int N
>
void Axi2Stream(hls::stream<data_type> &streamOut, data_type input[N]){
	for(int i = 0; i < N; i++){
#pragma HLS PIPELINE
		data_type val = input[i];
		streamOut.write(val);
	}
}

template <
	typename data_type,
	int N
>
void Stream2Axi(hls::stream<data_type> &streamIn, data_type output[N]){
	for(int i = 0; i < N; i++){
#pragma HLS PIPELINE
		data_type val = streamIn.read();
		output[i] = val;
	}
}

template <
	typename data_type,
	int N
>
void Axi2_2Streams(hls::stream<data_type> &streamOut0, hls::stream<data_type> &streamOut1, data_type input[N]){
	for(int i = 0; i < N; i++){
#pragma HLS PIPELINE
		data_type val = input[i];
		streamOut0.write(val);
		streamOut1.write(val);
	}
}

template <
	typename data_type,
	int N
>
void Axi2_4Streams(hls::stream<data_type> &streamOut0, hls::stream<data_type> &streamOut1, hls::stream<data_type> &streamOut2, hls::stream<data_type> &streamOut3, data_type input[N]){
	for(int i = 0; i < N; i++){
#pragma HLS PIPELINE
		data_type val = input[i];
		streamOut0.write(val);
		streamOut1.write(val);
		streamOut2.write(val);
		streamOut3.write(val);
	}
}

template <
	typename data_type,
	int N,
	int STREAM
>
void Axi2_MultiStreams(hls::stream<data_type> streamOut[STREAM], data_type input[N]){
	for(int i = 0; i < N; i++){
#pragma HLS PIPELINE
		data_type val = input[i];
		for(int j = 0; j < STREAM; j++){
			streamOut[j].write(val);
		}
	}
}

template <
	typename data_out,
	typename data_in,
	typename data_tmp,
	int N,
	int BLOCK,
	int LOOP,
	int AP_ELEM
>
void Axi2StreamMod(hls::stream<data_out> &streamOut, data_in input[N]){
	for(int i = 0; i < LOOP; i++){
	#pragma HLS PIPELINE
		data_out out;
		for(int j = 0; j < BLOCK; j++){
			data_in in = input[i*BLOCK + j];
			for(int k = 0; k < AP_ELEM; k++){
				unsigned int tmp = in.range(k*BIT_WIDTH + BIT_WIDTH_1, k*BIT_WIDTH);
				data_tmp f_val = *((data_tmp *)&tmp);
				out.val[j*AP_ELEM + k] = f_val;
			}
		}
		streamOut.write(out);
	}
}



template <
	typename data_t,
	unsigned int r_1,
	unsigned int r_2
>
void reduction(hls::stream<data_t> &streamOut, data_t input[r_1][r_2]){

	data_t acc_val = 0;

	for(int i = 0; i < r_1; i++){
		for(int j = 0; j < r_2; j++){
	#pragma HLS UNROLL
			acc_val += input[i][j];
		}
	}

	streamOut.write(acc_val);
}

template<
	typename data_t,
	int SIZE_0,
	int SIZE_1
>
void my_memset(data_t array[SIZE_0][SIZE_1], int val){
	for(int i = 0; i < SIZE_0; i++){
#pragma HLS PIPELINE
		for(int j = 0; j < SIZE_1; j++){
			array[i][j] = val;
		}
	}
}

template <
	typename data_t,
	unsigned int RED
>
data_t reduction_16(data_t input[RED]){
#pragma HLS INLINE

	data_t tmp0 = input[0] + input[1];
	data_t tmp1 = input[2] + input[3];
	data_t tmp2 = input[4] + input[5];
	data_t tmp3 = input[6] + input[7];
	data_t tmp4 = input[8] + input[9];
	data_t tmp5 = input[10] + input[11];
	data_t tmp6 = input[12] + input[13];
	data_t tmp7 = input[14] + input[15];

	data_t tmp8 = tmp0 + tmp1;
	data_t tmp9 = tmp2 + tmp3;
	data_t tmp10 = tmp4 + tmp5;
	data_t tmp11 = tmp6 + tmp7;

	data_t tmp12 = tmp8 + tmp9;
	data_t tmp13 = tmp10 + tmp11;

	data_t tmp14 = tmp12 + tmp13;

	return tmp14;
}

template <
	typename data_t,
	unsigned int RED
>
void reduction_6(hls::stream<data_t> &streamOut, data_t input[RED]){


	data_t tmp0 = input[0] + input[1];
	data_t tmp1 = input[2] + input[3];
	data_t tmp2 = input[4] + input[5];

	data_t tmp3 = tmp0 + tmp1;

	data_t tmp4 = tmp2 + tmp3;

	streamOut.write(tmp4);
}

template <
	typename data_t,
	unsigned int RED
>
void reduction_8(hls::stream<data_t> &streamOut, data_t input[RED]){


	data_t tmp0 = input[0] + input[1];
	data_t tmp1 = input[2] + input[3];
	data_t tmp2 = input[4] + input[5];
	data_t tmp3 = input[6] + input[7];

	data_t tmp4 = tmp0 + tmp1;
	data_t tmp5 = tmp2 + tmp3;

	data_t tmp6 = tmp4 + tmp5;

	streamOut.write(tmp6);
}

template <
	typename data_t,
	unsigned int RED
>
void reduction_12(hls::stream<data_t> &streamOut, data_t input[RED]){


	data_t tmp0 = input[0] + input[1];
	data_t tmp1 = input[2] + input[3];
	data_t tmp2 = input[4] + input[5];
	data_t tmp3 = input[6] + input[7];
	data_t tmp4 = input[8] + input[9];
	data_t tmp5 = input[10] + input[11];

	data_t tmp6 = tmp0 + tmp1;
	data_t tmp7 = tmp2 + tmp3;
	data_t tmp8 = tmp4 + tmp5;

	data_t tmp9 = tmp6 + tmp7;

	data_t tmp10 = tmp8 + tmp9;

	streamOut.write(tmp10);
}

template <
	typename data_t,
	unsigned int RED
>
void reduction_16(hls::stream<data_t> &streamOut, data_t input[RED]){
#pragma HLS INLINE

	data_t tmp0 = input[0] + input[1];
	data_t tmp1 = input[2] + input[3];
	data_t tmp2 = input[4] + input[5];
	data_t tmp3 = input[6] + input[7];
	data_t tmp4 = input[8] + input[9];
	data_t tmp5 = input[10] + input[11];
	data_t tmp6 = input[12] + input[13];
	data_t tmp7 = input[14] + input[15];

	data_t tmp8 = tmp0 + tmp1;
	data_t tmp9 = tmp2 + tmp3;
	data_t tmp10 = tmp4 + tmp5;
	data_t tmp11 = tmp6 + tmp7;

	data_t tmp12 = tmp8 + tmp9;
	data_t tmp13 = tmp10 + tmp11;

	data_t tmp14 = tmp12 + tmp13;

	streamOut.write(tmp14);
}

template <
	typename data_t,
	unsigned int RED
	>
data_t reduction_32(data_t input[RED]){
	data_t tmp_0 = input[0] + input[1];
	data_t tmp_1 = input[2] + input[3];
	data_t tmp_2 = input[4] + input[5];
	data_t tmp_3 = input[6] + input[7];
	data_t tmp_4 = input[8] + input[9];
	data_t tmp_5 = input[10] + input[11];
	data_t tmp_6 = input[12] + input[13];
	data_t tmp_7 = input[14] + input[15];
	data_t tmp_8 = input[16] + input[17];
	data_t tmp_9 = input[18] + input[19];
	data_t tmp_10 = input[20] + input[21];
	data_t tmp_11 = input[22] + input[23];
	data_t tmp_12 = input[24] + input[25];
	data_t tmp_13 = input[26] + input[27];
	data_t tmp_14 = input[28] + input[29];
	data_t tmp_15 = input[30] + input[31];

	data_t tmp_16 = tmp_0 + tmp_1;
	data_t tmp_17 = tmp_2 + tmp_3;
	data_t tmp_18 = tmp_4 + tmp_5;
	data_t tmp_19 = tmp_6 + tmp_7;
	data_t tmp_20 = tmp_8 + tmp_9;
	data_t tmp_21 = tmp_10 + tmp_11;
	data_t tmp_22 = tmp_12 + tmp_13;
	data_t tmp_23 = tmp_14 + tmp_15;

	data_t tmp_24 = tmp_16 + tmp_17;
	data_t tmp_25 = tmp_18 + tmp_19;
	data_t tmp_26 = tmp_20 + tmp_21;
	data_t tmp_27 = tmp_22 + tmp_23;

	data_t tmp_28 = tmp_24 + tmp_25;
	data_t tmp_29 = tmp_26 + tmp_27;

	data_t tmp_30 = tmp_28 + tmp_29;

	return tmp_30;

}

template <
	typename data_t,
	unsigned int RED
	>
data_t reduction_48(data_t input[RED]){
	data_t tmp_0 = input[0] + input[1];
	data_t tmp_1 = input[2] + input[3];
	data_t tmp_2 = input[4] + input[5];
	data_t tmp_3 = input[6] + input[7];
	data_t tmp_4 = input[8] + input[9];
	data_t tmp_5 = input[10] + input[11];
	data_t tmp_6 = input[12] + input[13];
	data_t tmp_7 = input[14] + input[15];
	data_t tmp_8 = input[16] + input[17];
	data_t tmp_9 = input[18] + input[19];
	data_t tmp_10 = input[20] + input[21];
	data_t tmp_11 = input[22] + input[23];
	data_t tmp_12 = input[24] + input[25];
	data_t tmp_13 = input[26] + input[27];
	data_t tmp_14 = input[28] + input[29];
	data_t tmp_15 = input[30] + input[31];
	data_t tmp_16 = input[32] + input[33];
	data_t tmp_17 = input[34] + input[35];
	data_t tmp_18 = input[36] + input[37];
	data_t tmp_19 = input[38] + input[39];
	data_t tmp_20 = input[40] + input[41];
	data_t tmp_21 = input[42] + input[43];
	data_t tmp_22 = input[44] + input[45];
	data_t tmp_23 = input[46] + input[47];

	data_t tmp_24 = tmp_0 + tmp_1;
	data_t tmp_25 = tmp_2 + tmp_3;
	data_t tmp_26 = tmp_4 + tmp_5;
	data_t tmp_27 = tmp_6 + tmp_7;
	data_t tmp_28 = tmp_8 + tmp_9;
	data_t tmp_29 = tmp_10 + tmp_11;
	data_t tmp_30 = tmp_12 + tmp_13;
	data_t tmp_31 = tmp_14 + tmp_15;
	data_t tmp_32 = tmp_16 + tmp_17;
	data_t tmp_33 = tmp_18 + tmp_19;
	data_t tmp_34 = tmp_20 + tmp_21;
	data_t tmp_35 = tmp_22 + tmp_23;

	data_t tmp_36 = tmp_24 + tmp_25;
	data_t tmp_37 = tmp_26 + tmp_27;
	data_t tmp_38 = tmp_28 + tmp_29;
	data_t tmp_39 = tmp_30 + tmp_31;
	data_t tmp_40 = tmp_32 + tmp_33;
	data_t tmp_41 = tmp_34 + tmp_35;

	data_t tmp_42 = tmp_36 + tmp_37;
	data_t tmp_43 = tmp_38 + tmp_39;
	data_t tmp_44 = tmp_40 + tmp_41;

	data_t tmp_45 = tmp_42 + tmp_43;

	data_t tmp_46 = tmp_44 + tmp_45;

	return tmp_46;

}

template <
	typename data,
	int N
>
void myMemset(data input[N], int value){
	for(int i = 0; i < N; i++){
#pragma HLS PIPELINE
		input[i] = value;
	}
}
