//
//  main-cl.cpp
//  N-Body
//
//  Authors:
//	Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//	{emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include "support.hpp"
#include "parser.hpp"

typedef float my_type;
#define TILE_ELEM 120000

/*
 * Given an event, this function returns the kernel execution time in ms
 */
float  getTimeDifference(cl_event event)
{
	cl_ulong	time_start = 0;
	cl_ulong	time_end = 0;
	float		total_time = 0.0;

	clGetEventProfilingInfo(event,
				CL_PROFILING_COMMAND_START,
				sizeof(time_start),
				&time_start,
				NULL);
	clGetEventProfilingInfo(event,
				CL_PROFILING_COMMAND_END,
				sizeof(time_end),
				&time_end,
				NULL);
	total_time = time_end - time_start;
	return total_time / 1000000.0;
	//To convert nanoseconds to milliseconds
}


int load_file_to_memory(const char *filename, char **result)
{
    size_t size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
        {
            *result = NULL;
            return -1; // -1 means file opening fail
        }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f))
        {
            free(*result);
            return -2; // -2 means file reading fail
        }
    fclose(f);
    (*result)[size] = 0;
    return size;
}

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

void final_computation(particle_t * p, coord3d_t *a, int N_loc){
    for (int i = 0; i < N_loc; i++) {
		p[i].p.x += p[i].v.x;
		p[i].p.y += p[i].v.y;
		p[i].p.z += p[i].v.z;
		p[i].v.x += a[i].x;
		p[i].v.y += a[i].y;
		p[i].v.z += a[i].z;
        }
}

void central_computation(particle_t * p, coord3d_t *a, int N_loc, float EPS, const float *m){
    for (int q = 0; q < N_loc; q++) {
    	for (int j = 0; j < N_loc; j++) {
			float rx = p[j].p.x - p[q].p.x;
			float ry = p[j].p.y - p[q].p.y;
			float rz = p[j].p.z - p[q].p.z;
			float dd = rx*rx + ry*ry + rz*rz + EPS;
			float d = 1/ (dd*sqrtf(dd));
			float s = m[j] * d;
			a[q].x += rx * s;
			a[q].y += ry * s;
			a[q].z += rz * s;
    	}
	}
}

/**
 * \brief Run the N-body simulation on the CPU.
 * \param [in]  N_loc               Number of particles
 * \param [in]  nt              Number of time-steps
 * \param [in]  EPS             Damping factor
 * \param [in]  m               Masses of the N_loc particles
 * \param [in]  in_particles    Initial state of the N_loc particles
 * \param [out] out_particles   Final state of the N_loc particles after nt time-steps
 * \param [out] time            Execution time
 */
void run_cpu(int N_loc, int nt, float EPS, const float *m,
                    const particle_t *in_particles, particle_t *out_particles,
                    double *time)
{
    particle_t *p = (particle_t *) malloc(N_loc * sizeof(particle_t));
    memcpy(p, in_particles, N_loc * sizeof(particle_t));
    
    coord3d_t *a = (coord3d_t *) malloc(N_loc * sizeof(coord3d_t));
    
    double wall_time_start, wall_time_end;
    double time_it_start, time_it_end;
    double time_up_start, time_up_end;
    
    //wall_time_start = get_time();
    
    for (int t = 0; t < nt; t++) {
        
        memset(a, 0, N_loc * sizeof(coord3d_t));
        
       // time_it_start = get_time();
        central_computation(p, a, N_loc, EPS, m);
       //time_it_end = get_time();

        //time_up_start = get_time();
        final_computation(p, a, N_loc);
        //time_up_end = get_time();

    }
    
    //wall_time_end = get_time();
    
    //*time = wall_time_end - wall_time_start;
    
    memcpy(out_particles, p, N_loc * sizeof(particle_t));
    
    free(p);
    free(a);
}

void run_FPGA(int N_loc, int nt, float EPS, float *m, const particle_t *in_particles, particle_t *out_particles,
		cl_context context, cl_command_queue commands, cl_program program, cl_kernel kernel){

	cl_mem     p_x_buff, p_y_buff, p_z_buff, c_buff, EPS_buff, tiling_factor_buff;
	cl_mem 	   a_x_buff, a_y_buff, a_z_buff;

	unsigned int tiling_factor = N_loc / TILE_ELEM;

//	if(N_loc % TILE_ELEM != 0){
//		tiling_factor++;
//	}
//
//	int new_N_loc = TILE_ELEM * tiling_factor;

	particle_t *p = (particle_t *) malloc(N_loc * sizeof(particle_t));
    memcpy(p, in_particles, N_loc * sizeof(particle_t));

    my_type *m_local = (my_type *) calloc(N_loc, sizeof(my_type));
    memcpy(m_local, m, N_loc * sizeof(my_type));

    my_type *p_x = (my_type *)malloc(N_loc * sizeof(my_type));
    my_type *p_y = (my_type *)malloc(N_loc * sizeof(my_type));
    my_type *p_z = (my_type *)malloc(N_loc * sizeof(my_type));

    my_type *v_x = (my_type *)malloc(N_loc * sizeof(my_type));
    my_type *v_y = (my_type *)malloc(N_loc * sizeof(my_type));
    my_type *v_z = (my_type *)malloc(N_loc * sizeof(my_type));

    my_type *a_x = (my_type *)malloc(N_loc * sizeof(my_type));
    my_type *a_y = (my_type *)malloc(N_loc * sizeof(my_type));
    my_type *a_z = (my_type *)malloc(N_loc * sizeof(my_type));


    for(int i = 0; i < N_loc; i++){
        p_x[i] = p[i].p.x;
    	p_y[i] = p[i].p.y;
        p_z[i] = p[i].p.z;
        v_x[i] = p[i].v.x;
        v_y[i] = p[i].v.y;
        v_z[i] = p[i].v.z;
        a_x[i] = -1000;
        a_y[i] = -1000;
        a_z[i] = -1000;

    }

	p_x_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(my_type) * N_loc, NULL, NULL);
	p_y_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(my_type) * N_loc, NULL, NULL);
	p_z_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(my_type) * N_loc, NULL, NULL);
	a_x_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(my_type) * N_loc, NULL, NULL);
	a_y_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(my_type) * N_loc, NULL, NULL);
	a_z_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(my_type) * N_loc, NULL, NULL);
	c_buff 	 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(my_type) * N_loc, NULL, NULL);
    EPS_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), NULL, NULL);
    tiling_factor_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, NULL);



    for (int t = 0; t < nt; t++) {

		int		error;
		//write buffers for the	kernel

		error = clEnqueueWriteBuffer(commands, p_x_buff, CL_TRUE, 0, sizeof(my_type) * N_loc, p_x, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			printf("error while writing the buffer!! p_x\n");
			exit(EXIT_FAILURE);
		}
		error = clEnqueueWriteBuffer(commands, p_y_buff, CL_TRUE, 0, sizeof(my_type) * N_loc, p_y, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			printf("error while writing the buffer!! p_y\n");
			exit(EXIT_FAILURE);
		}
		error = clEnqueueWriteBuffer(commands, p_z_buff, CL_TRUE, 0, sizeof(my_type) * N_loc, p_z, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			printf("error while writing the buffer!! p_z\n");
			exit(EXIT_FAILURE);
		}
		error = clEnqueueWriteBuffer(commands, c_buff, CL_TRUE, 0, sizeof(my_type) * N_loc, m_local, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			printf("error while writing the buffer!! m\n");
			exit(EXIT_FAILURE);
		}
		error = clEnqueueWriteBuffer(commands, EPS_buff, CL_TRUE, 0, sizeof(float), &EPS, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			printf("error while writing the buffer!! EPS\n");
			exit(EXIT_FAILURE);
		}
		error = clEnqueueWriteBuffer(commands, tiling_factor_buff, CL_TRUE, 0, sizeof(unsigned int), &tiling_factor, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			printf("error while writing the buffer!! tiling_factor\n");
			exit(EXIT_FAILURE);
		}



		//set the arguments for the kernel
		error = 0;


		error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &p_x_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &p_y_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &p_z_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_x_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &a_y_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 5, sizeof(cl_mem), &a_z_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 6, sizeof(cl_mem), &c_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 7, sizeof(cl_mem), &EPS_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
		error = clSetKernelArg(kernel, 8, sizeof(cl_mem), &tiling_factor_buff);
		if (error != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments 0! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}

		//Execute the kernel over the entire range of our 1 d input data set
		// using the maximum number of work group items for this device

		error = 1;
		cl_event enqueue_kernel;

		double start_time, end_time;

		start_time = get_time();

#ifdef C_KERNEL
		error = clEnqueueTask(commands, kernel, 0, NULL, &enqueue_kernel);
#endif
		if (error){
			printf("Error: Failed to execute kernel! %d\n", error);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}


		clWaitForEvents(1, &enqueue_kernel);

		end_time = get_time();

		float executionTime = getTimeDifference(enqueue_kernel);
		printf(" execution time is %f ms \n", executionTime);


		printf("Global execution time is %f s \n", end_time - start_time);

		//Read back the results from the device to verify the output
		cl_event readEvent_0, readEvent_1, readEvent_2;
		error = clEnqueueReadBuffer(commands,  a_x_buff, CL_TRUE, 0, sizeof(my_type) * N_loc, a_x, 0, NULL, &readEvent_0);
		error |= clEnqueueReadBuffer(commands, a_y_buff, CL_TRUE, 0, sizeof(my_type) * N_loc, a_y, 0, NULL, &readEvent_1);
		error |= clEnqueueReadBuffer(commands, a_z_buff, CL_TRUE, 0, sizeof(my_type) * N_loc, a_z, 0, NULL, &readEvent_2);

		if (error != CL_SUCCESS) {
			printf("error in reading the output!! %d \n", error);
			fflush(stdout);
			//return EXIT_FAILURE;
		}
		clWaitForEvents(1, &readEvent_0);
		clWaitForEvents(1, &readEvent_1);
		clWaitForEvents(1, &readEvent_2);


        for(int kk = 0; kk < N_loc; kk++){
        	p_x[kk] += v_x[kk];
        	p_y[kk] += v_y[kk];
        	p_z[kk] += v_z[kk];
        	v_x[kk] += a_x[kk];
        	v_y[kk] += a_y[kk];
        	v_z[kk] += a_z[kk];
        }

    }

    for(int i = 0; i < N_loc; i++){
    	p[i].p.x = p_x[i];
    	p[i].p.y = p_y[i];
    	p[i].p.z = p_z[i];
    	p[i].v.x = v_x[i];
    	p[i].v.y = v_y[i];
    	p[i].v.z = v_z[i];
    }

    memcpy(out_particles, p, N_loc * sizeof(particle_t));

    free(p);
    free(m_local);
    free(p_x);
    free(p_y);
    free(p_z);
    free(v_x);
    free(v_y);
    free(v_z);
    free(a_x);
    free(a_y);
    free(a_z);
}


void data_generation(int N_loc, particle_t **particles, float **m, params_t args_info){
    
    if (!args_info.random && !args_info.file) {
        print_usage();
        exit(EXIT_FAILURE);
    }
    
    if (args_info.random) {
        *particles = (particle_t *) calloc(N_loc, sizeof(particle_t));
        *m = (my_type *) calloc(N_loc, sizeof(my_type));
        
        srand(0);
        for (int i = 0; i < N_loc; i++)
        {
            (*m)[i] = (my_type)rand()/1000;
            (*particles)[i].p.x = (my_type)rand()/1000;
            (*particles)[i].p.y = (my_type)rand()/1000;
            (*particles)[i].p.z = (my_type)rand()/1000;
            (*particles)[i].v.x = (my_type)rand()/1000;
            (*particles)[i].v.y = (my_type)rand()/1000;
            (*particles)[i].v.z = (my_type)rand()/1000;
            
        }
    } else {
        const char *filename = args_info.file_name;
        
        FILE *fp = fopen(args_info.file_name, "r");
        if (fp == NULL) {
            fprintf(stderr, "Failed to open input file: `%s'\n", filename);
            exit(EXIT_FAILURE);
        }
        
        N_loc = count_lines(fp) - 1;
        
        if (args_info.num_particles < N_loc) {
            N_loc = args_info.num_particles;
        }
        
        *particles = (particle_t *) calloc(N_loc, sizeof(particle_t));
        *m = (float *) calloc(N_loc, sizeof(float));
        
        rewind(fp);
        
        fscanf(fp, "m,x,y,z,vx,vy,vz\n");
        for (int i = 0; i < N_loc; i++) {
            fscanf(fp, "%g,%g,%g,%g,%g,%g,%g", &((*m)[i]),
                   &((*particles)[i]).p.x, &((*particles)[i]).p.y, &((*particles)[i]).p.z,
                   &((*particles)[i]).v.x, &((*particles)[i]).v.y, &((*particles)[i]).v.z);
        }
        
        fclose(fp);
    }

    
    
}



int 
main(int argc, char **argv)
{

#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
  #define STR_VALUE(arg)      #arg
  #define GET_STRING(name) STR_VALUE(name)
  #define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif

	printf("starting HOST code \n");
	fflush(stdout);
	int		err;
	//error code returned from api calls

	//input parsing
	params_t args_info;

	printf("Parsing input...\n");

	if (parse_input(argc, argv, &args_info) != 0) {
		printf("Error in parsing input\n");
		print_usage();
		exit(EXIT_FAILURE);
	}

	printf("Done!\n");

	int N_loc = args_info.num_particles;
	int nt = args_info.num_timesteps;
	float EPS = args_info.EPS;
	float threshold = 0.001;

	if (EPS == 0) {
		fprintf(stderr, "EPS cannot be set to zero\n");
		exit(EXIT_FAILURE);
	}

	particle_t *particles;
	float *m;

	data_generation(N_loc, &particles, &m, args_info);

	particle_t *cpu_particles = NULL;
	particle_t *FPGA_particles = NULL;

	cpu_particles = (particle_t *) malloc(N_loc * sizeof(particle_t));
	FPGA_particles = (particle_t *)malloc(N_loc * sizeof(particle_t));

    cl_platform_id platform_id;         // platform id
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;           // compute kernel

    char cl_platform_vendor[1001];
	char cl_platform_name[1001];

    // Connect to first platform
	  //
		printf("GET platform \n");
	  err = clGetPlatformIDs(1,&platform_id,NULL);
	  if (err != CL_SUCCESS)
	  {
	    printf("Error: Failed to find an OpenCL platform!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	  }
		printf("GET platform vendor \n");
	  err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
	  if (err != CL_SUCCESS)
	  {
	    printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	  }
	  printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
		printf("GET platform name \n");
	  err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
	  if (err != CL_SUCCESS)
	  {
	    printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	  }
	  printf("CL_PLATFORM_NAME %s\n",cl_platform_name);

	  // Connect to a compute device
	  //
	  int fpga = 0;
	//#if defined (FPGA_DEVICE)
	  fpga = 1;
	//#endif
		printf("get device \n");
	  err = clGetDeviceIDs(platform_id, fpga ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_CPU,
	                       1, &device_id, NULL);
	  if (err != CL_SUCCESS)
	  {
	    printf("Error: Failed to create a device group!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	  }

	  // Create a compute context
	  //
		printf("create context \n");
	  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	  if (!context)
	  {
	    printf("Error: Failed to create a compute context!\n");
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	  }

	  // Create a command commands
	  //
		printf("create queue \n");
	  commands = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
	  if (!commands)
	  {
	    printf("Error: Failed to create a command commands!\n");
	    printf("Error: code %i\n",err);
	    printf("Test failed\n");
	    return EXIT_FAILURE;
	  }


	  int status;

	// Create Program Objects
	  //

	  // Load binary from disk
	  unsigned char *kernelbinary;
	  char *xclbin = (char *)malloc(sizeof(char) * (strlen(args_info.xclbin_name) + 1));
	  strcpy(xclbin, args_info.xclbin_name);
	  printf("loading %s\n", xclbin);
	  int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
	  if (n_i < 0) {
		printf("failed to load kernel from xclbin: %s\n", xclbin);
		printf("Test failed\n");
		return EXIT_FAILURE;
	  }
	  size_t n = n_i;
	  // Create the compute program from offline
		printf("create program with binary \n");
	  program = clCreateProgramWithBinary(context, 1, &device_id, &n,
										  (const unsigned char **) &kernelbinary, &status, &err);
	  if ((!program) || (err!=CL_SUCCESS)) {
		printf("Error: Failed to create compute program from binary %d!\n", err);
		printf("Test failed\n");
		return EXIT_FAILURE;
	  }

	  // Build the program executable
	  //
		printf("build program \n");
	  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	  if (err != CL_SUCCESS)
	  {
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		printf("Test failed\n");
		return EXIT_FAILURE;
	  }

	  // Create the compute kernel in the program we wish to run
	  //
	  printf("create kernel \n");

	  kernel = clCreateKernel(program, "nbody", &err);
	  if (!kernel || err != CL_SUCCESS)
	  {
		printf("Error: Failed to create compute kernel!\n");
		printf("Test failed\n");
		return EXIT_FAILURE;
	  }


	/* run FPGA */
	FPGA_particles = (particle_t *) malloc(N_loc * sizeof(particle_t));

	printf("Running on FPGA...\n");
	run_FPGA(N_loc, nt, EPS, m, particles, FPGA_particles, context, commands, program, kernel);

	printf("Done!\n");

	/* run CPU */
	cpu_particles = (particle_t *) malloc(N_loc * sizeof(particle_t));
	printf("Running on CPU...\n");

    double cpu_time = 0;
    run_cpu(N_loc, nt, EPS, m, particles, cpu_particles, &cpu_time);
	
    printf("Done!\n");

    int mismatches = 0;

    for(int i =0; i < N_loc; i++){
        if( abs(FPGA_particles[i].p.x - cpu_particles[i].p.x) > threshold ||
                abs(FPGA_particles[i].p.y - cpu_particles[i].p.y) > threshold ||
                abs(FPGA_particles[i].p.z - cpu_particles[i].p.z) > threshold ||
                abs(FPGA_particles[i].v.x - cpu_particles[i].v.x) > threshold ||
                abs(FPGA_particles[i].v.y - cpu_particles[i].v.y) > threshold ||
                abs(FPGA_particles[i].v.z - cpu_particles[i].v.z) > threshold){
//            printf("ERROR \n");
//            printf(" FPGA %f - CPU %f \n", FPGA_particles[i].p.x, cpu_particles[i].p.x);
//            printf(" FPGA %f - CPU %f \n", FPGA_particles[i].p.y, cpu_particles[i].p.y);
//            printf(" FPGA %f - CPU %f \n", FPGA_particles[i].p.z, cpu_particles[i].p.z);
//            printf(" FPGA %f - CPU %f \n", FPGA_particles[i].v.x, cpu_particles[i].v.x);
//            printf(" FPGA %f - CPU %f \n", FPGA_particles[i].v.y, cpu_particles[i].v.y);
//            printf(" FPGA %f - CPU %f \n", FPGA_particles[i].v.z, cpu_particles[i].v.z);

            mismatches++;

        }
	//printf(" %f \n", p_x[i]);
    }
    
    printf("Results Checked! \n");

    printf("Mismatches rate = %f\n", 100.0*mismatches/N_loc);


	//Shutdown and cleanup
		//
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	return EXIT_SUCCESS;
}
