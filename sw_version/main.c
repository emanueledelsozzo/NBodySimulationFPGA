//
//  main.c
//  N-Body
//
//  Authors:
//  Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//  {emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include "support.h"
#include "parser.h"


/**
 * \brief Count the number of lines in a file
 * \param [in] fp       File pointer
 * \return The number of lines
 */
static int count_lines(FILE *fp)
{
    int nl = 0;
    int el = 0;
    char buf[BUFSIZ];
    while (fgets(buf, sizeof(buf), fp) != NULL) {
        if (strchr(buf, '\n')) {
            nl++;
            el = 0;
        } else {
            el = 1;
        }
    }
    return nl + el;
}

void final_computation(particle_t * p, coord3d_t *a, int N){
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
            p[i].p.x += p[i].v.x;
            p[i].p.y += p[i].v.y;
            p[i].p.z += p[i].v.z;
            p[i].v.x += a[i].x;
            p[i].v.y += a[i].y;
            p[i].v.z += a[i].z;
        }
}

void central_computation(particle_t * p, coord3d_t *a, int N, float EPS, const float *m){
    #pragma omp parallel for
    for (int q = 0; q < N; q++) {
            for (int j = 0; j < N; j++) {
                //if (j != q) {
                float rx = p[j].p.x - p[q].p.x;
                float ry = p[j].p.y - p[q].p.y;
                float rz = p[j].p.z - p[q].p.z;
                float dd = rx*rx + ry*ry + rz*rz + EPS;
                float d = 1/ (dd*sqrtf(dd));
                float s = m[j] * d;
                a[q].x += rx * s;
                a[q].y += ry * s;
                a[q].z += rz * s;
                //}
            }
        }
}


void data_generation(int N, particle_t **particles, float **m, params_t args_info){
    
    if (!args_info.random && !args_info.file) {
        print_usage();
        exit(EXIT_FAILURE);
    }
    
    if (args_info.random) {
        *particles = (particle_t *) calloc(N, sizeof(particle_t));
        *m = (float *) calloc(N, sizeof(float));
        
        srand(100);
        for (int i = 0; i < N; i++)
        {
            (*m)[i] = (float)rand()/100000;
            (*particles)[i].p.x = (float)rand()/100000;
            (*particles)[i].p.y = (float)rand()/100000;
            (*particles)[i].p.z = (float)rand()/100000;
            (*particles)[i].v.x = (float)rand()/100000;
            (*particles)[i].v.y = (float)rand()/100000;
            (*particles)[i].v.z = (float)rand()/100000;
            
            
        }
    } else {
        const char *filename = args_info.file_name;
        
        FILE *fp = fopen(args_info.file_name, "r");
        if (fp == NULL) {
            fprintf(stderr, "Failed to open input file: `%s'\n", filename);
            exit(EXIT_FAILURE);
        }
        
        N = count_lines(fp) - 1;
        
        if (args_info.num_particles < N) {
            N = args_info.num_particles;
        }
        
        *particles = (particle_t *) calloc(N, sizeof(particle_t));
        *m = (float *) calloc(N, sizeof(float));
        
        rewind(fp);
        
        fscanf(fp, "m,x,y,z,vx,vy,vz\n");
        for (int i = 0; i < N; i++) {
            fscanf(fp, "%g,%g,%g,%g,%g,%g,%g", &((*m)[i]),
                   &((*particles)[i]).p.x, &((*particles)[i]).p.y, &((*particles)[i]).p.z,
                   &((*particles)[i]).v.x, &((*particles)[i]).v.y, &((*particles)[i]).v.z);
        }
        
        fclose(fp);
    }

    
    
}


/**
 * \brief Run the N-body simulation on the CPU.
 * \param [in]  N               Number of particles
 * \param [in]  nt              Number of time-steps
 * \param [in]  EPS             Damping factor
 * \param [in]  m               Masses of the N particles
 * \param [in]  in_particles    Initial state of the N particles
 * \param [out] out_particles   Final state of the N particles after nt time-steps
 * \param [out] time            Execution time
 */
void run_cpu(int N, int nt, float EPS, const float *m,
                    const particle_t *in_particles, particle_t *out_particles,
                    double *time)
{
    particle_t *p = (particle_t *) malloc(N * sizeof(particle_t));
    memcpy(p, in_particles, N * sizeof(particle_t));
    
    coord3d_t *a = (coord3d_t *) malloc(N * sizeof(coord3d_t));
    
    double wall_time_start, wall_time_end;
    double time_it_start, time_it_end;
    double time_up_start, time_up_end;
    
    wall_time_start = get_time();
    
    outer_loop:for (int t = 0; t < nt; t++) {
        //printf("Iteration %d - ", t);
        
        memset(a, 0, N * sizeof(coord3d_t));
        
        time_it_start = get_time();
        central_computation(p,a,N, EPS, m);
       time_it_end = get_time();
        
        
        time_up_start = get_time();
        final_computation(p,a,N);
        time_up_end = get_time();
        
        //printf("time computation: %f - time update: %f\n", time_it_end - time_it_start, time_up_end - time_up_start);
        
    }
    
    wall_time_end = get_time();
    
    //*time = wall_time_end - wall_time_start;
    *time = time_it_end - time_it_start;
    
    memcpy(out_particles, p, N * sizeof(particle_t));
    
    free(p);
    free(a);
}

int main(int argc, char **argv)
{
    params_t args_info;
    
    if (parse_input(argc, argv, &args_info) != 0) {
        exit(EXIT_FAILURE);
    }
    
    int N = args_info.num_particles;
    int nt = args_info.num_timesteps;
    float EPS = args_info.EPS;
    
    if (EPS == 0) {
        fprintf(stderr, "EPS cannot be set to zero\n");
        exit(EXIT_FAILURE);
    }
    
    particle_t *particles;
    float *m;
    
    data_generation(N, &particles, &m, args_info);
    
    double cpuTime = 0;
    particle_t *cpu_particles = NULL;
    
    cpu_particles = (particle_t *) malloc(N * sizeof(particle_t));
        
    puts("Running on CPU...\n");
    run_cpu(N, nt, EPS, m, particles, cpu_particles, &cpuTime);
    printf("CPU execution time: %.3gs\n", cpuTime);
    
    
    free_params_t(&args_info);
    free(particles);
    free(m);
    free(cpu_particles);
    
    return 0;
}
