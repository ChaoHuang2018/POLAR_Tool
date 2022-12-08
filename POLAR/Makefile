CXX = g++
HOME= /usr/local/include
LIB_HOME = ../flowstar/flowstar-toolbox
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g -L$(LIB_HOME) -L/usr/local/lib
OBJS = NeuralNetwork.o BernsteinPoly.o Neuron.o PolarSetting.o System.o Specification.o Polar.o

all: polar_lib polar_tool
# all: benchmark6_tanh

polar_tool: Polar_tool.o $(OBJS)
	g++ -O3 -w $(LINK_FLAGS) -L./ -o $@ $^ $(LIBS)

polar_lib: $(OBJS)
	ar rcs libpolar.a $^

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o libpolar.a polar_tool

