CXX = g++
HOME = /usr/local/include
LIB_HOME = ../flowstar/flowstar-toolbox

# ---- Prefix (conda/system) ----
PREFIX ?= $(CONDA_PREFIX)
ifeq ($(PREFIX),)
PREFIX = /usr/local
endif

# ---- Protobuf compiler ----
PROTOC ?= $(PREFIX)/bin/protoc
ifeq ($(wildcard $(PROTOC)),)
PROTOC = protoc
endif

# ---- Local ONNX proto + bridge header (in repo) ----
ONNX_PROTO_DIR ?= $(abspath third_party/onnx_proto)
ONNX_BRIDGE_INC ?= $(abspath third_party/onnx_include)

# ---- Generated protobuf outputs ----
GEN_DIR    = build/onnx_pb
PROTO_WRAP = $(GEN_DIR)/proto
WRAP_ONNX  = $(PROTO_WRAP)/onnx

ONNX_PROTO_ML      = $(ONNX_PROTO_DIR)/onnx-ml.proto
WRAP_ONNX_PROTO_ML = $(WRAP_ONNX)/onnx-ml.proto

GEN_ONNX_ML_H  = $(GEN_DIR)/onnx/onnx-ml.pb.h
GEN_ONNX_ML_CC = $(GEN_DIR)/onnx/onnx-ml.pb.cc
GEN_ONNX_ML_O  = $(GEN_DIR)/onnx/onnx-ml.pb.o

# ---- Abseil libs (override if needed) ----
ABSL_LIBS ?= \
  -labsl_log_internal_check_op \
  -labsl_log_internal_message \
  -labsl_log_internal_format \
  -labsl_log_internal_conditions \
  -labsl_log_internal_globals \
  -labsl_log_severity \
  -labsl_raw_logging_internal \
  -labsl_base \
  -labsl_strings \
  -labsl_throw_delegate

# ---- Libraries ----
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread -lprotobuf $(ABSL_LIBS)

# ---- Flags ----
CFLAGS = -I . -I$(ONNX_BRIDGE_INC) -I$(GEN_DIR) -I$(PREFIX)/include -I $(HOME) -g -O3 -std=c++17
CFLAGS += -DUSE_ONNX_PROTO -DONNX_ML=1 -DONNX_NAMESPACE=onnx

LINK_FLAGS = -g -L$(LIB_HOME) -L/usr/local/lib -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib

OBJS = NeuralNetwork.o BernsteinPoly.o Neuron.o PolarSetting.o System.o Specification.o Polar.o

all: polar_lib polar_tool

NeuralNetwork.o: onnx_pb_gen

polar_tool: Polar_tool.o $(OBJS) $(GEN_ONNX_ML_O)
	$(CXX) -O3 -w $(LINK_FLAGS) -L./ -o $@ $^ $(LIBS)

polar_lib: $(OBJS)
	ar rcs libpolar.a $^

onnx_pb_gen: $(GEN_ONNX_ML_H)

$(WRAP_ONNX_PROTO_ML): $(ONNX_PROTO_ML)
	@mkdir -p $(WRAP_ONNX)
	@ln -sf $(ONNX_PROTO_ML) $@

$(GEN_ONNX_ML_H) $(GEN_ONNX_ML_CC): $(WRAP_ONNX_PROTO_ML)
	@mkdir -p $(GEN_DIR)
	$(PROTOC) -I $(PROTO_WRAP) --cpp_out=$(GEN_DIR) $(WRAP_ONNX_PROTO_ML)

$(GEN_ONNX_ML_O): $(GEN_ONNX_ML_CC) $(GEN_ONNX_ML_H)
	$(CXX) -O3 -c $(CFLAGS) -o $@ $(GEN_ONNX_ML_CC)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<

clean:
	rm -rf build
	rm -f *.o libpolar.a polar_tool
