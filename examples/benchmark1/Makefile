# ===== Compiler =====
CXX         ?= g++-15
STD         ?= -std=gnu++17

# ===== Homebrew prefix =====
BREW_PREFIX ?= /opt/homebrew

# ===== Third-party libs (Homebrew) =====
MPFR_HOME  ?= $(BREW_PREFIX)/opt/mpfr
GMP_HOME   ?= $(BREW_PREFIX)/opt/gmp
GSL_HOME   ?= $(BREW_PREFIX)/opt/gsl
GLPK_HOME  ?= $(BREW_PREFIX)/opt/glpk

# ===== ONNX / Protobuf / Abseil =====
ONNX_HOME      ?= $(BREW_PREFIX)
PROTOBUF_HOME  ?= $(BREW_PREFIX)
ABSL_HOME      ?= $(BREW_PREFIX)/opt/abseil

# ===== Your libs =====
POLAR_HOME     ?= ../../POLAR
FLOWSTAR_HOME  ?= ../../flowstar/flowstar-toolbox

# ===== Flags =====
CXXFLAGS  ?= -O3 -g $(STD) -I.
CXXFLAGS  += -I../../
# 如需头文件（若源码中直接 include 了 polar/flowstar 的 .h），再开启下面两行
# CXXFLAGS  += -I$(POLAR_HOME) -I$(FLOWSTAR_HOME)

CXXFLAGS  += -I$(MPFR_HOME)/include -I$(GMP_HOME)/include -I$(GSL_HOME)/include -I$(GLPK_HOME)/include
# 与主库一致：开启 ONNX 直读宏（可选；示例源码本身一般不需要）
CXXFLAGS  += -DUSE_ONNX_PROTO -DONNX_ML=1 -DONNX_NAMESPACE=onnx
# 头文件路径（链接阶段只要库即可，但加上不影响）
CXXFLAGS  += -I$(ONNX_HOME)/include -I$(PROTOBUF_HOME)/include -I$(ABSL_HOME)/include

LDFLAGS   ?= -g -L$(POLAR_HOME) -L$(FLOWSTAR_HOME)
LDFLAGS   += -L$(MPFR_HOME)/lib -L$(GMP_HOME)/lib -L$(GSL_HOME)/lib -L$(GLPK_HOME)/lib
# ONNX/Protobuf/Abseil 库路径 + 运行时 rpath
LDFLAGS   += -L$(ONNX_HOME)/lib -L$(PROTOBUF_HOME)/lib -L$(ABSL_HOME)/lib
LDFLAGS   += -Wl,-rpath,$(POLAR_HOME) -Wl,-rpath,$(FLOWSTAR_HOME)
LDFLAGS   += -Wl,-rpath,$(ONNX_HOME)/lib -Wl,-rpath,$(PROTOBUF_HOME)/lib -Wl,-rpath,$(ABSL_HOME)/lib
LDFLAGS   += -Wl,-rpath,$(MPFR_HOME)/lib -Wl,-rpath,$(GMP_HOME)/lib -Wl,-rpath,$(GSL_HOME)/lib -Wl,-rpath,$(GLPK_HOME)/lib

# ===== Libraries =====
LIBS      ?= -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lglpk -lpthread -lm

# --- ONNX + Protobuf（任选其一：先试 full；不行再换 lite）---
LIBS      += -lonnx -lonnx_proto -lprotobuf
# 如要减少依赖，可改为：
# LIBS   += -lonnx -lonnx_proto -lprotobuf-lite

# --- Abseil：按主库做法带上 logging 相关 + 基础依赖 ---
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
  -labsl_log_entry \
  -labsl_strings \
  -labsl_strings_internal \
  -labsl_str_format_internal \
  -labsl_base \
  -labsl_int128 \
  -labsl_spinlock_wait \
  -labsl_malloc_internal \
  -labsl_raw_logging_internal \
  -labsl_throw_delegate \
  -labsl_die_if_null \
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

# ===== Targets =====
.PHONY: all clean
all: reachnn_benchmark_1 reachnn_benchmark_1_json reachnn_benchmark_1_onnx flowstar_1step flowstar_1step_v1 nncs_reachnn_benchmark_1

reachnn_benchmark_1_onnx: reachnn_benchmark_1_onnx.o
	$(CXX) -O3 -w $(LDFLAGS) -o $@ $^ $(LIBS)

reachnn_benchmark_1_json: reachnn_benchmark_1_json.o
	$(CXX) -O3 -w $(LDFLAGS) -o $@ $^ $(LIBS)

reachnn_benchmark_1: reachnn_benchmark_1.o
	$(CXX) -O3 -w $(LDFLAGS) -o $@ $^ $(LIBS)

flowstar_1step: flowstar_1step.o
	$(CXX) -O3 -w $(LDFLAGS) -o $@ $^ $(LIBS)

flowstar_1step_v1: flowstar_1step_v1.o
	$(CXX) -O3 -w $(LDFLAGS) -o $@ $^ $(LIBS)

nncs_reachnn_benchmark_1: nncs_reachnn_benchmark_1.o
	$(CXX) -O3 -w $(LDFLAGS) -o $@ $^ $(LIBS)

# ===== Build rules =====
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
%.o: %.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o reachnn_benchmark_1 reachnn_benchmark_1_json flowstar_1step flowstar_1step_v1 nncs_reachnn_benchmark_1

