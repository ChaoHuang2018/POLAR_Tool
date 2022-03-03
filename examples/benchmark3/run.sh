make &&\
./reachnn_benchmark_3 0.05 60 4 6 0 sigmoid 0.02 &&\
./reachnn_benchmark_3 0.05 60 4 6 0 tanh 0.02 &&\
./reachnn_benchmark_3 0.05 60 4 6 0 relu 0.02 &&\
./reachnn_benchmark_3 0.05 60 4 6 0 relu_sigmoid 0.02 &&\
./reachnn_benchmark_3 0.05 60 4 6 1 sigmoid 0.02 &&\
./reachnn_benchmark_3 0.05 60 4 6 1 tanh 0.02 &&\
./reachnn_benchmark_3 0.05 60 4 6 1 relu 0.02 &&\
./reachnn_benchmark_3 0.05 60 4 6 1 relu_sigmoid 0.02
