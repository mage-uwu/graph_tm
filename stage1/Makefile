CC      ?= gcc
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -lpthread -lm

all: gtm gtm_portable

gtm: gtm.c main.c gtm.h
	$(CC) $(CFLAGS) -o $@ gtm.c main.c $(LDFLAGS)

# no AVX-512: exercises the scalar feedback path (must give identical results)
gtm_portable: gtm.c main.c gtm.h
	$(CC) -O3 -march=x86-64-v2 -std=c11 -Wall -Wextra -Wno-unused-parameter -o $@ gtm.c main.c $(LDFLAGS)

# sanitizer build for the parity suite
gtm_asan: gtm.c main.c gtm.h
	$(CC) -O1 -g -march=native -std=c11 -fsanitize=address,undefined -o $@ gtm.c main.c $(LDFLAGS)

clean:
	rm -f gtm gtm_portable gtm_asan
