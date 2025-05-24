# ---------------------------- V1 --------------------

V1_SRC = V1.c
V1_EXE = V1

V1_RUN_FLAGS = -O3
V1_PROFILE_FLAGS = -pg

V1_PROFILE_OUT = v1_gmon.out
V1_PROFILE_REPORT = v1_profile.txt

V1_CC = gcc

# Run build (optimized)
V1: $(V1_EXE)

$(V1_EXE): $(V1_SRC)
$(V1_CC) $(V1_RUN_FLAGS) -lm -o $@ $<

run_v1: $(V1_EXE)
./$(V1_EXE)

profile_v1:
$(V1_CC) $(V1_PROFILE_FLAGS) -lm -o $(V1_EXE) $(V1_SRC)
./$(V1_EXE)
gprof $(V1_EXE) gmon.out > $(V1_PROFILE_REPORT)
@echo "V1 profiling report saved to $(V1_PROFILE_REPORT)"

clean_v1:
rm -f $(V1_EXE) gmon.out $(V1_PROFILE_REPORT)

# ---------------------------- V2 --------------------

V2_SRC = V2.cu
V2_EXE = V2

V2_RUN_FLAGS = -O3
V2_PROFILE_FLAGS = -pg

V2_PROFILE_OUT = v2_gmon.out
V2_PROFILE_REPORT = v2_profile.txt

V2_CC = nvcc

V2: $(V2_EXE)

$(V2_EXE): $(V2_SRC)
$(V2_CC) $(V2_RUN_FLAGS) -o $@ $<

run_v2: $(V2_EXE)
./$(V2_EXE)

profile_v2:
$(V2_CC) $(V2_PROFILE_FLAGS) -o $(V2_EXE) $(V2_SRC)
./$(V2_EXE)
gprof $(V2_EXE) gmon.out > $(V2_PROFILE_REPORT)
@echo "V2 profiling report saved to $(V2_PROFILE_REPORT)"

clean_v2:
rm -f $(V2_EXE) gmon.out $(V2_PROFILE_REPORT)

# ---------------------------- V3 --------------------

V3_SRC = V3.cu
V3_EXE = V3

V3_RUN_FLAGS = -O3
V3_PROFILE_FLAGS = -pg

V3_PROFILE_OUT = v3_gmon.out
V3_PROFILE_REPORT = v3_profile.txt

V3_CC = nvcc

V3: $(V3_EXE)

$(V3_EXE): $(V3_SRC)
$(V3_CC) $(V3_RUN_FLAGS) -o $@ $<

run_v3: $(V3_EXE)
./$(V3_EXE)

profile_v3:
$(V3_CC) $(V3_PROFILE_FLAGS) -o $(V3_EXE) $(V3_SRC)
./$(V3_EXE)
gprof $(V3_EXE) gmon.out > $(V3_PROFILE_REPORT)
@echo "V3 profiling report saved to $(V3_PROFILE_REPORT)"

clean_v3:
rm -f $(V3_EXE) gmon.out $(V3_PROFILE_REPORT)

# ---------------------------- V4 --------------------

V4_SRC = V4.cu
V4_EXE = V4

V4_RUN_FLAGS = -lcublas -O3 -arch=sm_75 -lcudart -lcuda
V4_PROFILE_FLAGS = -pg -lcublas -arch=sm_75 -lcudart -lcuda

V4_PROFILE_OUT = v4_gmon.out
V4_PROFILE_REPORT = v4_profile.txt

V4_CC = nvcc

V4: $(V4_EXE)

$(V4_EXE): $(V4_SRC)
$(V4_CC) $(V4_RUN_FLAGS) -o $@ $<

run_v4: $(V4_EXE)
./$(V4_EXE)

profile_v4:
$(V4_CC) $(V4_PROFILE_FLAGS) -o $(V4_EXE) $(V4_SRC)
./$(V4_EXE)
gprof $(V4_EXE) gmon.out > $(V4_PROFILE_REPORT)
@echo "V4 profiling report saved to $(V4_PROFILE_REPORT)"

clean_v4:
rm -f $(V4_EXE) gmon.out $(V4_PROFILE_REPORT)
