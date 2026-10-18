compiler = mpicxx
CUDA_HOME = /usr/local/cuda-11.8
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_ARCH = sm_80

ifeq ($(mode),debug)
	cflags += -std=c++17 -O3 -ffp-contract=off -I ./include \
          -I $(CUDA_HOME)/include -fPIE
	linkflags += -L$(CUDA_HOME)/lib64 -lcudart
else
	cflags += -std=c++17 -O3 -ffp-contract=off -I ./include \
          -I $(CUDA_HOME)/include -fPIE
	linkflags 	+= -L$(CUDA_HOME)/lib64 -lcudart
endif

SrcDir = ./src
HeadDir = ./include
ObjDir = ./obj
OutDir = ./output/results ./output/time_record/compute ./output/time_record/total ./output/load ./output/grid

source = $(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp))
head = $(foreach dir,$(HeadDir),$(wildcard $(dir)/*.h $(dir)/*.hpp $(dir)/*.cuh))

object = $(patsubst %.cpp,$(ObjDir)/%.o,$(notdir $(source)))
object += $(ObjDir)/TimeAdvCuda.o
object += $(ObjDir)/FlowfieldCuda.o
object += $(ObjDir)/ReactionCuda.o

output = $(foreach dir,$(OutDir),$(wildcard $(dir)/*.txt $(dir)/*.dat))
target = EU3D

NO_COLOR		=\033[0m
OK_COLOR		=\033[32;01m

$(target):$(object) $(head)
	$(compiler) -o $(target) $(object) $(linkflags)
	@printf "$(OK_COLOR)Compiling Is Successful!\nExecutable File: $(target) $(NO_COLOR)\n"

$(ObjDir)/%.o:$(SrcDir)/%.cpp $(head)
	$(compiler) -c $(cflags) $< -o $@

$(ObjDir)/TimeAdvCuda.o: $(SrcDir)/TimeAdvCuda.cu $(head)
	$(NVCC)  -c -std=c++17 -O3 --fmad=false -arch=$(CUDA_ARCH) \
	    -I ./include $< -o $@
		
$(ObjDir)/FlowfieldCuda.o: $(SrcDir)/FlowfieldCuda.cu $(head)
	$(NVCC)  -c -std=c++17 -O3 --fmad=false -arch=$(CUDA_ARCH) \
	    -I ./include $< -o $@

$(ObjDir)/ReactionCuda.o: $(SrcDir)/ReactionCuda.cu $(head)
	$(NVCC)  -c -std=c++17 -O3 --fmad=false -arch=$(CUDA_ARCH) \
	    -I ./include $< -o $@

.PHONY:clean,cleano
clean:
	-rm $(object) $(target)
cleano:
	-rm $(output)

