# fast_logicae: standalone LogicAE (DDLGN) training + fast inference. Needs gcc with OpenMP, nothing else.
#   make                 fastlogic (everything, one file) + the split tools lt, fastlae, hardcheck
#   make check           selftest (gradient contract, packed == scalar, checkpoint round trip)
#   make gen MODEL=m.lth DATA=d.ids SEQ=64    fastgen: m compiled to straight-line C (T <= 64; ~2 min for 16x1024)
# lt is built -std=c11 on purpose: gnu11 lets gcc contract a*b+c into FMA, which changes training bits.
CC ?= gcc
OPT = -O3 -march=native -fopenmp -Wno-unknown-pragmas
all: fastlogic lt fastlae hardcheck
fastlogic: fastlogic.c
	$(CC) $(OPT) -std=c11 fastlogic.c -lm -o $@
lt: logic_text.c
	$(CC) $(OPT) -std=c11 logic_text.c -lm -o $@
fastlae: fastlae.c logic_text.c
	$(CC) $(OPT) -std=gnu11 -I. fastlae.c -lm -o $@
hardcheck: tools/hardcheck.c logic_text.c
	$(CC) $(OPT) -std=c11 -I. tools/hardcheck.c -lm -o $@
check: lt
	./lt selftest
gen: fastlae
	./fastlae gen $(MODEL) $(DATA) $(SEQ) model.inc
	$(CC) -O2 -march=native -fopenmp -Wno-unknown-pragmas -std=gnu11 -I. -DGEN_INC='"model.inc"' fastgen.c -lm -o fastgen
	./fastgen verify $(MODEL) $(DATA) $(SEQ)
clean:
	rm -f fastlogic lt fastlae hardcheck fastgen model.inc
.PHONY: all check gen clean
