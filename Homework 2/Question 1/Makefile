CC = gcc
CFLAGS = -Wall -Wextra -O2

SOURCES = q1_a_mc.c q1_a_leibniz.c q1_b_mc.c q1_b_leibniz.c
TARGETS = am al bm bl

am: q1_a_mc.c
	$(CC) $(CFLAGS) -pthread -o $@ $<

al: q1_a_leibniz.c
	$(CC) $(CFLAGS) -pthread -o $@ $<

bm: q1_b_mc.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

bl: q1_b_leibniz.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

all: $(TARGETS)

run:
	@if [ -z "$(target)" ]; then \
		echo 'Error: Specify an executable with make run target="prog" args="num_threads problem_size"'; \
		exit 1; \
	fi; \
	if [ -x "$(target)" ]; then \
		./$(target) $(args); \
	else \
		echo "Error: Executable '$(target)' not found!"; \
		exit 1; \
	fi

clean:
	rm -f $(TARGETS)

help:
	@echo 'Usage:'
	@echo '  To compile all executables, run "make all"'
	@echo ''
	@echo '  To run an executable, use:'
	@echo '    make run target="prog" args="num_threads problem_size"'
	@echo ''
	@echo '  where "
