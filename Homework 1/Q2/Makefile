CC = gcc
CFLAGS = -Wall -O2 -pthread

# Define the target executable
TARGET = problem2

# Define the source file
SRC = problem2.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run