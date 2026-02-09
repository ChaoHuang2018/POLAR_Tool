# ===== Compiler =====
CXX         ?= g++-15
STD         ?= -std=gnu++17

# ===== Homebrew prefix (Apple Silicon 默认 /opt/homebrew) =====
BREW_PREFIX ?= /opt/homebrew

# ===== ONNX / Protobuf / Abseil homes =====
ONNX_HOME      ?= $(BREW_PREFIX)
PROTOBUF_HOME  ?= $(BREW_PREFIX)
ABSL_HOME      ?= $(BREW_PREFIX)/opt/abseil

# ===== flowstar 静态库位置 =====
FLOWSTAR_HOME ?= $(abspath ../flowstar/flowstar-toolbox)
FLOWSTAR_LIB  ?= $(FLOWSTAR_HOME)

# ===== 其他依赖（Homebrew 常见路径）=====
MPFR_HOME  ?= $(BREW_PREFIX)/opt/mpfr
GMP_HOME   ?= $(BREW_PREFIX)/opt/gmp
GSL_HOME   ?= $(BREW_PREFIX)/opt/gsl
GLPK_HOME  ?= $(BREW_PREFIX)/opt/glpk

# ===== Flags =====
CXXFLAGS   ?= -O3 -g $(STD) -I.
CXXFLAGS   += -I$(MPFR_HOME)/include -I$(GMP_HOME)/include -I$(GSL_HOME)/include -I$(GLPK_HOME)/include
CXXFLAGS   += -I$(FLOWSTAR_HOME) -I..
# ONNX/Protobuf/Abseil 头
CXXFLAGS   += -I$(ONNX_HOME)/include -I$(PROTOBUF_HOME)/include -I$(ABSL_HOME)/include
# 开启 ONNX 直读，并兼容 onnx-ml 头文件
CXXFLAGS   += -DUSE_ONNX_PROTO -DONNX_ML=1 -DONNX_NAMESPACE=onnx

LDFLAGS    ?= -g -L$(FLOWSTAR_LIB)
LDFLAGS    += -L$(MPFR_HOME)/lib -L$(GMP_HOME)/lib -L$(GSL_HOME)/lib -L$(GLPK_HOME)/lib
# ONNX/Protobuf/Abseil 库路径 + 运行时查找
LDFLAGS    += -L$(ONNX_HOME)/lib -L$(PROTOBUF_HOME)/lib -L$(ABSL_HOME)/lib
LDFLAGS    += -Wl,-rpath,$(ONNX_HOME)/lib -Wl,-rpath,$(PROTOBUF_HOME)/lib -Wl,-rpath,$(ABSL_HOME)/lib

# ===== Libraries =====
LIBS  = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lglpk -lpthread -lm

# ONNX + Protobuf（任选其一：先试 full；不行再换 lite）
# 如果 full 通过，就用 full；如果 full 仍然报 absl 符号，可保留 full，关键在于把 absl logging 链接齐
LIBS += -lonnx -lonnx_proto -lprotobuf
# 如需尝试 lite，把上一行的 -lprotobuf 改为 -lprotobuf-lite

# ===== Abseil: 你机器上实际存在的 logging 相关 + 依赖基础件 =====
# --- logging 内核 ---
LIBS += \
  -labsl_log_internal_message \
  -labsl_log_internal_check_op \
  -labsl_log_internal_conditions \
  -labsl_log_internal_format \
  -labsl_log_internal_globals \
  -labsl_log_internal_log_sink_set \
  -labsl_log_internal_fnmatch \
  -labsl_log_internal_nullguard \
  -labsl_log_internal_proto \
  -labsl_log_internal_structured_proto \
  -labsl_log_severity \
  -labsl_log_sink \
  -labsl_log_globals \
  -labsl_log_initialize \
  -labsl_log_flags \
  -labsl_log_entry

# --- 常见基础依赖（strings/format/base/int128/alloc 等）---
LIBS += \
  -labsl_strings \
  -labsl_strings_internal \
  -labsl_str_format_internal \
  -labsl_base \
  -labsl_int128 \
  -labsl_spinlock_wait \
  -labsl_malloc_internal \
  -labsl_raw_logging_internal \
  -labsl_throw_delegate \
  -labsl_die_if_null

# --- 与 time / timezone / 调试栈 相关（被 logging 及 onnx 侧引用）---
LIBS += \
  -labsl_time \
  -labsl_civil_time \
  -labsl_time_zone \
  -labsl_debugging_internal \
  -labsl_demangle_internal \
  -labsl_stacktrace \
  -labsl_symbolize \
  -labsl_cord \
  -labsl_cord_internal \
  -labsl_graphcycles_internal


# ===== Objects =====
OBJS = NeuralNetwork.o BernsteinPoly.o Neuron.o PolarSetting.o System.o Specification.o Polar.o

# ===== Targets =====
.PHONY: all polar_lib polar_tool clean
all: polar_lib polar_tool

polar_tool: Polar_tool.o $(OBJS)
	$(CXX) -w $(LDFLAGS) -L./ -o $@ $^ $(LIBS)

polar_lib: $(OBJS)
	ar rcs libpolar.a $^

# ===== Build rules =====
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
%.o: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o libpolar.a polar_tool
