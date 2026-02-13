CXX = g++
HOME = /usr/local/include

POLAR_HOME = ../../POLAR
FLOWSTAR_HOME = ../../flowstar/flowstar-toolbox

PREFIX ?= $(CONDA_PREFIX)
ifeq ($(PREFIX),)
PREFIX = /usr/local
endif

CFLAGS = -I . -I $(HOME) -I$(PREFIX)/include -g -O3 -std=c++17
CFLAGS += -I$(POLAR_HOME) -I$(FLOWSTAR_HOME)
CFLAGS += -I$(POLAR_HOME)/third_party/onnx_include -I$(POLAR_HOME)/build/onnx_pb
CFLAGS += -DUSE_ONNX_PROTO -DONNX_ML=1 -DONNX_NAMESPACE=onnx

LINK_FLAGS = -g -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L/usr/local/lib -L$(PREFIX)/lib \
             -Wl,-rpath,$(POLAR_HOME) -Wl,-rpath,$(FLOWSTAR_HOME) -Wl,-rpath,$(PREFIX)/lib

ABSL_LIBS ?= -labsl_log_internal_check_op -labsl_log_internal_message -labsl_log_internal_format \
            -labsl_log_internal_conditions -labsl_log_internal_globals -labsl_log_severity \
            -labsl_raw_logging_internal -labsl_base -labsl_strings -labsl_throw_delegate

LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread -lprotobuf $(ABSL_LIBS)

ONNX_PB_OBJ ?= $(POLAR_HOME)/build/onnx_pb/onnx/onnx-ml.pb.o

.PHONY: all clean check_onnx_pb
all: reachnn_benchmark_1 flowstar_1step flowstar_1step_v1

check_onnx_pb:
	@test -f $(ONNX_PB_OBJ) || (echo "Missing $(ONNX_PB_OBJ). Build POLAR first: make -C $(POLAR_HOME) PREFIX=..."; false)

reachnn_benchmark_1: reachnn_benchmark_1.o check_onnx_pb
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $< $(ONNX_PB_OBJ) $(LIBS)

flowstar_1step: flowstar_1step.o check_onnx_pb
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $< $(ONNX_PB_OBJ) $(LIBS)

flowstar_1step_v1: flowstar_1step_v1.o check_onnx_pb
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $< $(ONNX_PB_OBJ) $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<

clean:
	rm -f *.o reachnn_benchmark_1 flowstar_1step flowstar_1step_v1
