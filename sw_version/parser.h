//
//  parser.h
//  N-Body
//
//  Authors:
//  Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//  {emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#ifndef parser_h
#define parser_h

typedef struct params {
    int num_particles;
    int num_timesteps;
    float EPS;
    int random;
    int file;
    char *file_name;
} params_t;

struct options {
    char *param_short;
    char *param_long;
    int step;
    char param_val;
};

void print_usage();

int parse_input(int argc, char *argv[], params_t *args_info);

void free_params_t(params_t *args_info);


#endif /* parser_h */
