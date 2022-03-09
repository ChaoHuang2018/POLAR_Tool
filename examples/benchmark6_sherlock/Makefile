CXX = g++
HOME= /usr/local/include
POLAR_HOME = ../../POLAR
FLOWSTAR_HOME = ../../flowstar
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g  -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L/usr/local/lib


all: reachnn_benchmark_6

reachnn_benchmark_6: reachnn_benchmark_6.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)



%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o reachnn_benchmark_6

