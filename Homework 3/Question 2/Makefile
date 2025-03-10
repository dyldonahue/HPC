CC = gcc
CFLAGS = -Wall -Wextra -O2 -mavx512f

SOURCES = q2.c
TARGETS = q3

q2: q2.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f $(TARGETS)