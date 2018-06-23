//
//  support.h
//  N-Body
//
//  Authors:
//	Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//	{emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#ifndef SUPPORT_H_
#define SUPPORT_H_


typedef struct {
    float x;
    float y;
    float z;
} coord3d_t;

typedef struct {
    coord3d_t p;
    coord3d_t v;
} particle_t;


double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#endif
