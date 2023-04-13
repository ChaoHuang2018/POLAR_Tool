CXX = g++-8
HOME= /usr/local/include
POLAR_HOME = ../../POLAR
FLOWSTAR_HOME = ../../flowstar/flowstar-toolbox
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g  -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L/usr/local/lib


all: reachnn_benchmark_1 flowstar_1step flowstar_1step_v1 nncs_reachnn_benchmark_1

reachnn_benchmark_1: reachnn_benchmark_1.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

flowstar_1step: flowstar_1step.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

flowstar_1step_v1: flowstar_1step_v1.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nncs_reachnn_benchmark_1: nncs_reachnn_benchmark_1.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o reachnn_benchmark_1 flowstar_1step flowstar_1step_v1 nncs_reachnn_benchmark_1

