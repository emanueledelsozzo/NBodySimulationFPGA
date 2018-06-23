
app := N-Body

sources := $(wildcard *.c)
headers := $(wildcard *.h)
src := $(sources) $(headers)
OBJECTS = $(sources:.c=.o)
OFLAGS = -O3 -fopenmp -std=c99

all: $(app)

CFLAGS := -c -fopenmp -std=c99

$(app).o: $(src)
	$(CC) $(sources) $(CFLAGS)

$(app): $(app).o
	$(CC) $(OBJECTS) -fopenmp -o $@ -lm

.PHONY: clean

o3:
	$(CC) $(sources) $(OFLAGS) -lm -o $(app)

clean:
	@rm *.o $(app) 2> /dev/null || true
