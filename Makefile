# Local default: RTX 3050 / CUDA 12.5. A800 overrides are documented in README.
CXX = mpicxx
CUDA_HOME ?= /usr/local/cuda-12.5
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_86
CXXFLAGS = -std=c++17 -O3 -ffp-contract=off -fPIE -Iinclude -I$(CUDA_HOME)/include
NVCCFLAGS = -std=c++17 -O3 --fmad=false -arch=$(CUDA_ARCH) -Iinclude
LDLIBS = -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart
CPP_SOURCES := $(wildcard src/*.cpp)
CU_SOURCES := $(wildcard src/*.cu)
OBJECTS := $(patsubst src/%.cpp,obj/%.o,$(CPP_SOURCES)) $(patsubst src/%.cu,obj/%.o,$(CU_SOURCES))
HEADERS := $(wildcard include/*.hpp include/*.h include/*.cuh)

.PHONY: all clean test-upload-reuse
all: EU3D

EU3D: $(OBJECTS)
	$(CXX) $^ $(LDLIBS) -o $@

obj/%.o: src/%.cpp $(HEADERS) Makefile | obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

obj/%.o: src/%.cu $(HEADERS) Makefile | obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

obj:
	mkdir -p $@

obj/test_upload_reuse: tests/test_upload_reuse.cu src/FlowfieldCuda.cu $(HEADERS) Makefile | obj
	$(NVCC) $(NVCCFLAGS) tests/test_upload_reuse.cu src/FlowfieldCuda.cu -o $@

test-upload-reuse: obj/test_upload_reuse
	./obj/test_upload_reuse

clean:
	rm -f $(OBJECTS) obj/test_upload_reuse EU3D
