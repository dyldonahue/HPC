CC = gcc
CFLAGS = -Wall -Wextra -O2

SOURCES = q2.c
TARGETS = q2

q2: q2.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

clean:
	rm -f $(TARGETS)
