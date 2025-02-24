CC = gcc
CFLAGS = -Wall -Wextra -O2

SOURCES = q3.c
TARGETS = q3

q3: q3.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

clean:
	rm -f $(TARGETS)

