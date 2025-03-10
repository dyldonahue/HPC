To compile each of problems 1-3, cd into their directory, and run "make"

To run each of problems 1-3, run the executable directly, ie

./q1
./q2
./q3

Problem 4 does not have a makefile. To load OpenBLAS on Explorer, run, 

module load OpenBLAS

Then, compile manually via 

gcc -o q4 q4.c -lopenblas

and run with 

./q4

