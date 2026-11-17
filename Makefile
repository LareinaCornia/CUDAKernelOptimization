NVCC ?= nvcc
HOST_CXX ?=
CUDA_ARCHS ?= 75 86 89
NVCCFLAGS ?= -O3 --compiler-options="-g3 -Wall -Wextra -Werror -Wno-unused-function"

ARCH_FLAGS = $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
COMPILE_CUDA = $(NVCC) $(if $(HOST_CXX),-ccbin $(HOST_CXX)) $(NVCCFLAGS) $(ARCH_FLAGS) -Iinclude
define build_cuda
tmp=$$(mktemp "$@.XXXXXX") || exit 1; \
trap 'rm -f "$$tmp"' EXIT; \
$(COMPILE_CUDA) "$<" -o "$$tmp" $(1) && mv -f "$$tmp" "$@"
endef

BUILD_DIR := build
PROGRAMS := saxpy sgemm conv1d query_config
EXECS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))
DEPS := Makefile include/cuda_error.cuh

.PHONY: all clean check $(PROGRAMS)

all: $(EXECS)

$(PROGRAMS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p "$@"

$(BUILD_DIR)/saxpy: kernels/vector_addition/saxpy.cu $(DEPS) | $(BUILD_DIR)
	$(call build_cuda)

$(BUILD_DIR)/sgemm: kernels/matrix_multiply/sgemm.cu kernels/matrix_multiply/kernels.cuh $(DEPS) | $(BUILD_DIR)
	$(call build_cuda,-lcublas)

$(BUILD_DIR)/conv1d: kernels/depth_conv1d/conv1d_cuda.cu kernels/depth_conv1d/conv1d_kernel.cuh $(DEPS) | $(BUILD_DIR)
	$(call build_cuda)

$(BUILD_DIR)/query_config: tools/query_config.cu $(DEPS) | $(BUILD_DIR)
	$(call build_cuda)

ifeq (check,$(firstword $(MAKECMDGOALS)))
CHECK_SUITES := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
KNOWN_SUITES := all saxpy_correct saxpy_perf sgemm_all sgemm_perf sgemm0 sgemm1 sgemm2 sgemm3 sgemm4 sgemm5 nondivisible conv1d_correct conv1d_perf
ifneq ($(strip $(filter-out $(KNOWN_SUITES),$(CHECK_SUITES))),)
$(error Unknown test suite: $(filter-out $(KNOWN_SUITES),$(CHECK_SUITES)))
endif
SUITE_GOALS := $(filter-out all,$(CHECK_SUITES))
ifneq ($(strip $(SUITE_GOALS)),)
.PHONY: $(SUITE_GOALS)
$(SUITE_GOALS): check
endif
endif

check: all
	uv run --locked python -u -m cuda_explore.run_tests $(CHECK_SUITES) $(CHECK_ARGS)

clean:
	rm -f $(EXECS)
