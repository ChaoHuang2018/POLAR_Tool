# make &&\
# ./reachnn_benchmark_5 0.01 10 4 6 0 sigmoid 0.005 &&\
#./reachnn_benchmark_5 0.01 10 4 6 0 tanh 0.005 &&\
#./reachnn_benchmark_5 0.01 10 4 6 0 relu 0.005 &&\
#./reachnn_benchmark_5 0.01 10 4 6 0 relu_tanh 0.005 &&\
./reachnn_benchmark_5 0.01 10 2 2 1 relu &&\
./reachnn_benchmark_5 0.01 10 2 2 1 sigmoid &&\
./reachnn_benchmark_5 0.01 10 2 3 1 tanh &&\
./reachnn_benchmark_5 0.01 10 2 2 1 relu_tanh 
