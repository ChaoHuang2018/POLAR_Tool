CXX = g++-8
HOME= /usr/local/include
POLAR_HOME = ../../POLAR
FLOWSTAR_HOME = ../../flowstar/flowstar-toolbox
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g  -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L/usr/local/lib

all: nn_docking_v4 flowstar_1step_v1

nn_docking_v1: nn_docking_v1.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_docking_v2: nn_docking_v2.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

nn_docking_v3: nn_docking_v3.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS) 

nn_docking_v4: nn_docking_v4.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS) 

flowstar_1step_v1: flowstar_1step_v1.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)



%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o nn_docking_v1 nn_docking_v4 flowstar_1step_v1  
