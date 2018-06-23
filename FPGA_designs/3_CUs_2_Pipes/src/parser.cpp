//
//  parser.cpp
//  N-Body
//
//  Authors:
//  Emanuele Del Sozzo, Marco Rabozzi, Lorenzo Di Tucci
//  {emanuele.delsozzo, marco.rabozzi, lorenzo.ditucci}@polimi.it
//

#include "parser.hpp"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


const char *help[] = {
    "  -h, --help               Print help and exit",
	"  -K, --kernel=FILE        xclbin file name",
    "  -N, --num-particles=INT  Maximum number of particles. The default value is \n                             only used\n                             	 when the input is random.  (default=`384')",
    "  -t, --num-timesteps=INT  Number of time-steps  (default=`1')",
    "  -e, --EPS=FLOAT          Damping factor  (default=`100')",
    "\n Group: input",
    "  -r, --random             Generate random input data",
    "  -f, --file=FILE          Read input data from file",
    0
};



void print_usage(){
    int i = 0;
    while (help[i]){
        printf("%s\n", help[i++]);
    }
    return;
}

char get_opt(int argc, char *argv[], int *option_index){
    
    int index = *option_index;
    int i;
    
    static struct options options_list[] = {
        { (char *)"-h", (char *)"--help",           0, 'h' },
        { (char *)"-N", (char *)"--num-particles",  1, 'N' },
        { (char *)"-t", (char *)"--num-timesteps",  1, 't' },
        { (char *)"-e", (char *)"--EPS",            1, 'e' },
        { (char *)"-r", (char *)"--random",         0, 'r' },
        { (char *)"-f", (char *)"--file",           1, 'f' },
		{ (char *)"-K", (char *)"--kernel",  		1, 'K' },
        { NULL,			NULL,               		0, '?' }
    };
    
    
    if (index >= argc){
        return -1;
    }
    
    for(i = 0; options_list[i].param_short; i++){
        if(!strcmp(options_list[i].param_long, argv[index]) ||
           !strcmp(options_list[i].param_short, argv[index])){
            *option_index = index + options_list[i].step;
            return options_list[i].param_val;
        }
    }
    return options_list[i].param_val;
    
    
}


int parse_input(int argc, char *argv[], params_t *args_info){
    int c;	/* Character of the parsed option.  */
    int option_index = 1;
    int int_value;
    float float_value;
    
    params_t local_params = {384, 1, 100.0, 0, 0, 0};
     
    if(argc <= 1){
        print_usage();
        return 1;
    }
    
    while (1)
    {
        
        c = get_opt (argc, argv, &option_index);
        
        if (c == -1) break;	/* Exit from `while (1)' loop.  */
        
        switch (c)
        {
            case 'h':	/* Print help and exit.  */
                
                print_usage();
                exit (EXIT_SUCCESS);
                
            case 'N':	/* Maximum number of particles. The default value is only used
                         when the input is random..  */
                int_value = atoi(argv[option_index]);
                if(int_value > 0){
                    local_params.num_particles = int_value;
                }else{
                    puts("Please enter a positive value for num-particles parameter\n");
                    return 1;
                }
                break;
                
            case 't':	/* Number of time-steps.  */
                
                int_value = atoi(argv[option_index]);
                if(int_value > 0){
                    local_params.num_timesteps = int_value;
                }else{
                    puts("Please enter a positive value for num-timesteps parameter\n");
                    return 1;
                }
                break;
                
            case 'e':	/* Damping factor.  */
                
                float_value = atof(argv[option_index]);
                if(float_value > 0){
                    local_params.EPS = float_value;
                }else{
                    puts("Please enter a positive value for EPS parameter\n");
                    return 1;
                }
                break;
                
                
            case 'r':	/* Generate random input data.  */
                
                local_params.random = 1;
                
                break;
            case 'f':	/* Read input data from file.  */
                
                local_params.file = 1;
                local_params.file_name = (char *)malloc(sizeof(char) * (strlen(argv[option_index]) + 1));
                strcpy(local_params.file_name, argv[option_index]);
                break;

            case 'K':	/* Read xclbin from user-specified file.  */

            	local_params.xclbin = 1;
            	local_params.xclbin_name = (char *)malloc(sizeof(char) * (strlen(argv[option_index]) + 1));
				strcpy(local_params.xclbin_name, argv[option_index]);
				break;
                
            default:	/* bug: option not considered.  */
                fprintf (stderr, "%s: option unknown\n", argv[option_index]);
                print_usage();
                return 1;
        } /* switch */
        option_index += 1;
    } /* while */
    
    if(local_params.xclbin == 0){
    	printf("Missing xclbin file!\n");
    	return 1;
    }

    
    memcpy(args_info, &local_params, sizeof(params_t));
    
    return 0;
}

void free_params_t(params_t *args_info){
    
    if(args_info->file){
        free(args_info->file_name);
    }
}
