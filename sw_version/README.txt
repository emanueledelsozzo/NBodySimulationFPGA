This version of N-Body is parallelized by means of OpenMP framework

Compile as:
make o3

Run as:
OMP_NUM_THREADS=numThread ./N-Body [flags]

List of supported flags:
-h, --help               Print help and exit
-N, --num-particles=INT  Maximum number of particles. The default value is only used when the input is random (default="384")
-t, --num-timesteps=INT  Number of time-steps (default="1")
-e, --EPS=FLOAT          Damping factor (default="100")
-r, --random             Generate random input data
-f, --file=FILE          Read input data from file

