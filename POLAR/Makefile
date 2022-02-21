CXX = g++
HOME= /usr/local/include
LIB_HOME = ../flowstar
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g -L$(LIB_HOME) -L/usr/local/lib
OBJS = NeuralNetwork.o BernsteinPoly.o Neuron.o PolarSetting.o

all: polar_lib
# all: benchmark6_tanh

polar_lib: $(OBJS)
	ar rcs libpolar.a $^

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o libpolar.a

