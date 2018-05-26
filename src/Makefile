# Parallel Sorting by Regular Sampling (PSRS)
# Author: Jimmy Nguyen
# Email: jimmy@jimmyworks.net
#
# Problem Domain:
# Generate 128,000,000 32-bit numbers_size on each of 32 nodes
# and sort them using the PSRS algorithm. Sorting is verified
# by the parallel definition of sorted.  The program is timed
# on based on the sorting tasks performed.
#
# Program Description:
# This program uses the PSRS algorithm for parallel sorting.
# Sorting on a single process uses quicksort which is optimized
# using OpenMP.  The overall program is parallelized using MPI
# and multiple processors.  Operation can be performed in one
# of three modes: NORMAL, DEBUG, and TIMER.  NORMAL mode runs
# without any debug statements nor time statements except for
# the final reported run time.  DEBUG mode prints all debug
# statements and can only be done with the sample count specified.
# TIMER mode prints all the time intervals throughout the program
# allowing the program to be profiled for various bottlenecks.
# Set up info for C++ implicit rule
#
# Makefile Commands:
#
# make          make executable
# make test     test all test cases
# make debug    run in debug mode to show correct output
# make time     run program in profiling mode to show times
# make backup   backup files

CC = mpicc
CCFLAGS = -lm -fopenmp
CPPFLAGS =
LDFLAGS =
PROJECTNAME = PSRS
SRCS = psrs.c
EXEC = psrs

MPI = mpirun
MPIFLAGS = -n 6
ARGS =

# For Testing

N1  = -n 1
N2  = -n 2
N3  = -n 3
N4  = -n 4
N5  = -n 5
N6  = -n 6
N7  = -n 7
N8  = -n 8
N9  = -n 9
N10 = -n 10

M1  = 100000
M2  = 50000
M3  = 33333
M4  = 25000
M5  = 20000
M6  = 16667
M7  = 14285
M8  = 12500
M9  = 11111
M10 = 10000

# ======================================================
# ======================================================
OBJS = $(SRCS:c=o)

# make
all: $(EXEC) $(EXEO3)

# make test
test: all all_tests

# make time
time:
        $(MPI) $(N4) ./$(EXEC) $(M10) TIMER

# make debug
debug:
        clear
        $(MPI) $(N4) ./$(EXEC) 8 DEBUG

# make clean
clean:
        rm -f $(OBJS) *.d *~  $(EXEC)

Makefile: $(SRCS:.c=.d)

# Pattern for .d files.
%.d:%.c
        @echo Updating .d Dependency File
        @set -e; rm -f $@; \
        $(CC) -MM $(CPPFLAGS) $< > $@.$$$$; \
        sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
        rm -f $@.$$$$

#  Linker rule
$(EXEC) : $(OBJS)
        $(CC) -o $@ $(CCFLAGS) $^

# Backup Target
backup: clean
        @mkdir -p ~/backups; chmod 700 ~/backups
        @$(eval CURDIRNAME := $(shell basename `pwd`))
        @$(eval MKBKUPNAME := ~/backups/$(PROJECTNAME)-$(shell date +'%Y.%m.%d-%H:%M:%S').tar.gz)
        @echo
        @echo Writing Backup file to: $(MKBKUPNAME)
        @echo
        @-tar zcfv $(MKBKUPNAME) ../$(CURDIRNAME)
        @chmod 600 $(MKBKUPNAME)
        @echo
        @echo Done!


# Include the dependency files
-include $(SRCS:.c=.d)

# Test cases
all_tests:
        $(MPI) $(N10) ./$(EXEC) $(M10)
        $(MPI) $(N9)  ./$(EXEC) $(M9)
        $(MPI) $(N8)  ./$(EXEC) $(M8)
        $(MPI) $(N7)  ./$(EXEC) $(M7)
        $(MPI) $(N6)  ./$(EXEC) $(M6)
        $(MPI) $(N5)  ./$(EXEC) $(M5)
        $(MPI) $(N4)  ./$(EXEC) $(M4)
        $(MPI) $(N3)  ./$(EXEC) $(M3)
        $(MPI) $(N2)  ./$(EXEC) $(M2)
        $(MPI) $(N1)  ./$(EXEC) $(M1)

