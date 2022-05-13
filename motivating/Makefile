CXX = g++-8
HOME= /usr/local/include
POLAR_HOME = ../../POLAR
FLOWSTAR_HOME = ../../flowstar
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g  -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L/usr/local/lib


all: motivating_berns motivating_taylor motivating_berns_only

motivating_berns: motivating_berns.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

motivating_berns_only: motivating_berns_only.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

motivating_taylor: motivating_taylor.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o motivating_berns motivating_taylor motivating_berns_only
