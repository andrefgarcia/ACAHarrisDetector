
TARGETS = lib/libcutil_x86_64.a harrisDetector 

all: $(TARGETS)

harrisDetector: harrisDetector.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc harrisDetector.cu -Llib -lcutil_x86_64 -o harrisDetector

lib/libcutil_x86_64.a: 
	make -C common

clean:
	make -C common clean
	rm -f $(TARGETS) 
