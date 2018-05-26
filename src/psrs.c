// Parallel Sorting by Regular Sampling (PSRS)
// Author: Jimmy Nguyen
// Email: jimmy@jimmyworks.net
//
// Problem Domain:
// Generate 128,000,000 32-bit numbers on each of 32 nodes
// and sort them using the PSRS algorithm. Sorting is verified
// by the parallel definition of sorted.  The program is timed
// on based on the sorting tasks performed.
//
// Program Description:
// This program uses the PSRS algorithm for parallel sorting.
// Sorting on a single process uses quicksort which is optimized
// using OpenMP.  The overall program is parallelized using MPI
// and multiple processors.  Operation can be performed in one
// of three modes: NORMAL, DEBUG, and TIMER.  NORMAL mode runs
// without any debug statements nor time statements except for
// the final reported run time.  DEBUG mode prints all debug
// statements and can only be done with the sample count specified.
// TIMER mode prints all the time intervals throughout the program
// allowing the program to be profiled for various bottlenecks.
//
// NOTE: DEBUG mode also limits values to 0-29 for easy reading.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <limits.h>

#define DEFAULT_SAMPLE_COUNT 128000000 //Default sample size
#define ROOT 0 // Root process

// Enumeration for run modes
enum run_modes{ NORMAL,
                DEBUG,
                TIMER};

// Globals for process id and run mode
int p_id;
int mode = NORMAL;

// Function signatures
void swap (int *x, int *y); // Swap values
void quicksort(int *array, int low, int high); // Quicksort
int subsort(int *array, int low, int high); // Sub-quicksort algo
float getTimeLapse (struct timeval *last); // Get time elapsed since last

// Main
// (input) int argc     main argument count
// (input) char** argv  main argument array
//                        argv[0] = program
//                        argv[1] = number of numbers_size
//                        argv[2] = run mode:
//                        NORMAL - default run mode
//                        DEBUG - print debug statements
//                        TIMED - print time statements (profiling)
// (output) int         return code
// return code 0        success
//        code 1        invalid command line arg
int main(int argc, char** argv)
{
   // Recorded times
   // Checkpoint used for program profiling and seed for srand()
   struct timeval start, stop, checkpoint, seed;
   float elapsed;

   int p_count; // Number of processes

   // General use
   int i, j, offset, count;
   int flag = 1; // flag used to send messages

   // Local sample collection
   int *numbers;
   int numbers_size = DEFAULT_SAMPLE_COUNT;

   // Regular samples sent/received from root
   int *samples;
   int sample_count, sample_size;

   // Total regular samples collected (only allocated by root)
   int *collective;
   int collective_size;

   // Final sorted sample collection
   int finalSortedSize;
   int *finalSorted;

   // Send/Receive counts and displacement arrays
   int sDispl, rDispl; // count variables
   int *sendCount = malloc(p_count*sizeof(int)); // send count to each proc
   int *recvCount = malloc(p_count*sizeof(int)); // recv count from each proc
   int *sendDispl = malloc(p_count*sizeof(int)); // send displacement

   // MPI Housekeeping
   MPI_Init(&argc, &argv);                   // Initialize MPI Environment
   MPI_Comm_size(MPI_COMM_WORLD, &p_count);  // Get the number of processes
   MPI_Comm_rank(MPI_COMM_WORLD, &p_id);     // Get the rank of the process

   // Seed the RNG using process id
   gettimeofday( &seed, NULL );
   srand((seed.tv_sec*1000) + (seed.tv_usec/1000) + p_id);

   // Print the run details and initialize run settings
   if(p_id==ROOT)
   {
      printf("RUN CONFIGURATION\n");
      printf("==========================================\n");
   }
   // Check program args
   switch(argc)
   {
      case 1: // NORMAL RUN
             if(p_id==ROOT)
             {
                printf("Normal run. Processors = %d,\n", p_count);
                printf("            Samples/Processor = %d\n", numbers_size);
             }
             break;
      case 3: // DEBUG or TIMER RUN
             if(strcmp(argv[2],"DEBUG")==0)
             {
                if(p_id==ROOT) printf("DEBUG MODE\n");
                mode = DEBUG;
             }
             else if(strcmp(argv[2],"TIMER")==0)
             {
                if(p_id==ROOT) printf("TIMER MODE\n");
                mode = TIMER;
             }
             else
             {
                if(p_id==ROOT)
                {
                   printf("Invalid run args.  Second option must be \"DEBUG\" or \"TIMER\"\n");
                   printf("Usage: psrs [<sample count> [ DEBUG | TIMER ]]\n");
                }
                MPI_Finalize();
				return 1;
             }
      case 2: // Custom sample size
             numbers_size = atoi(argv[1]);
             // Valid size
             if(numbers_size < 1)
             {
                if(p_id==ROOT)
                {
                   printf("Invalid run args.  Sample size must be greater than 0.\n");
                   printf("Usage: psrs [<sample count> [ DEBUG | TIMER]]\n");
                   MPI_Finalize();
                   return 1;
                }

             }
             // Print run values
             if(p_id==ROOT)
             {
                printf("Custom run\nProcessors = %d,\n", p_count);
                printf("Samples/Processor = %d\n", numbers_size);
             }
             break;
      default:
             if(p_id==ROOT)
             {
                printf("Error: invalid run args.\n");
                printf("Usage: psrs [<sample count> [ DEBUG | TIMER ]]\n");
             }
             MPI_Finalize();
             return 1;
   }
   if(p_id==ROOT) printf("==========================================\n");

   // Allocate local collection
   numbers = malloc(numbers_size*sizeof(int));

   // Fill each local collection with random samples
   if(mode==DEBUG)
   {  // If DEBUG mode, make numbers range 1 - 30
      for(i = 0; i < numbers_size; i++)
      {
         numbers[i] = rand() % 30;
      }
   }
   else
   {  // Else, numbers are any 32-bit value
      for(i = 0; i < numbers_size; i++)
      {
         numbers[i] = rand();
      }
   }

 //// DEBUG PRINT - LOCAL COLLECTIONS
   if(mode==DEBUG)
   {
      // Print each process collection
      if(p_id==ROOT)
      {
         printf("\nINITIAL LOCAL COLLECTIONS\n");
         printf("==========================================\n");
      }
      if(p_id) MPI_Recv(&flag, 1, MPI_INT, p_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Process %d: ", p_id);
      for(i = 0; i < numbers_size; i++)
      {
         printf("%d ", numbers[i]);
      }
      printf("\n");
      if(p_id < (p_count-1))
         MPI_Send(&flag, 1, MPI_INT, p_id+1, 1, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Start Timer
   MPI_Barrier(MPI_COMM_WORLD);
   gettimeofday( &start, NULL );
   checkpoint = start;

   // Quicksort local collections
   quicksort(numbers, 0, numbers_size-1);

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Quicksort initial collection: \t%f\n", elapsed);
   }

 //// DEBUG PRINT - SORTED LOCAL COLLECTIONS
   if(mode==DEBUG)
   {
      // Print each process collection
      if(p_id==ROOT)
      {
         printf("\nSORTED LOCAL COLLECTIONS\n");
         printf("==========================================\n");
      }
      if(p_id) MPI_Recv(&flag, 1, MPI_INT, p_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Process %d: ", p_id);
      for(i = 0; i < numbers_size; i++)
      {
         printf("%d ", numbers[i]);
      }
      printf("\n");
      if(p_id < (p_count-1)) MPI_Send(&flag, 1, MPI_INT, p_id+1, 1, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Regular Sampling size -- handle when p_count less than number of samples
   sample_count = (p_count < numbers_size)? p_count : numbers_size;

   // Allocate space for regular samples and pick them out
   samples = malloc(sample_count*sizeof(int));
   offset = numbers_size/sample_count; // space out selections evenly
   j = offset-1; // index of first one
   for(i = 0; i < sample_count; i++)
   {
      samples[i] = numbers[j];
      j += offset;
   }

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Get regular samples: \t\t%f\n", elapsed);
   }

 //// DEBUG PRINT - REGULAR SAMPLES
   if(mode==DEBUG)
   {
      if(p_id==ROOT)
      {
         printf("\nREGULAR SAMPLES\n");
         printf("==========================================\n");
      }
      if(p_id) MPI_Recv(&flag, 1, MPI_INT, p_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // Samples Array
      printf("Process %d: ", p_id);
      for(i = 0; i < sample_count; i++)
      {
         printf("%d ", samples[i]);
      }

      printf("\n");
      if(p_id < (p_count-1)) MPI_Send(&flag, 1, MPI_INT, p_id+1, 1, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Allocate space for the total collection of all regular samples
   collective_size = p_count*sample_count; // total size of array
   if(p_id==ROOT) // only root process needs to make this
      collective = malloc(collective_size*sizeof(int));

   // Gather all regular samples to root process
   MPI_Gather(samples, sample_count, MPI_INT, collective, sample_count, MPI_INT, ROOT, MPI_COMM_WORLD);

   // Free samples array and re-allocate space for split values
   free(samples);
   sample_size = p_count; // size of array
   sample_count = sample_size-1; // only p_count - 1 split values received; last one is MPI_MAX
   samples = malloc(sample_size*sizeof(int));
   
   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Gather all reg samples: \t%f\n", elapsed);
   }

   // Quicksort all regular samples
   // Then, pick out split values
   if(p_id==ROOT)
   {
      quicksort(collective, 0, collective_size-1);

      // Create array of split values to broadcast back
      offset = collective_size/p_count; // pick values evenly spaced
      j = offset - 1; // index of first one
      for(i = 0; i < sample_count; i++)
      {
         samples[i] = collective[j];
         j += offset;
      }
      samples[sample_count] = INT_MAX-1; // last section is everything less than INT_MAX
   }

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Quicksort and select split values\n from all reg samples:\t\t%f\n", elapsed);
   }

   // Broadcast split values to all processes
   MPI_Bcast(samples, sample_count, MPI_INT, ROOT, MPI_COMM_WORLD);

 //// DEBUG PRINT - TOTAL REGULAR SAMPLES and BROADCASTED PIVOT VALUES
   if(mode==DEBUG)
   {
      if(p_id==ROOT)
      {
         printf("\nTOTAL SORTED REGULAR SAMPLES\n");
         printf("==========================================\n");
		 for(i = 0; i < collective_size; i++)
         {
            printf("%d ", collective[i]);
         }
         printf("\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);

      if(p_id==1)
      {
         printf("\nBROADCASTED SPLIT VALUES\n");
         printf("==========================================\n");
         printf("Process %d: ", p_id);
         for(i = 0; i < sample_count; i++)
         {
            printf("%d ", samples[i]);
         }
         printf("\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Broadcast split values: \t%f\n", elapsed);
   }

   // Calculate the send count to each processor
   count = 0;
   j = 0; // local sorted array index iterator
   for(i = 0; i < p_count; i++)
   {
      while(numbers[j] <= samples[i] && j < numbers_size)
      {
         count++;
         j++;
      }
      sendCount[i] = count;
      count = 0;
   }

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Calculate send counts: \t\t%f\n", elapsed);
   }

   // Send All-to-all exchange so all processes know what how much space
   // to allocate for final array
   // sendCount -> recvCount
   MPI_Alltoall(sendCount, 1, MPI_INT, recvCount, 1, MPI_INT, MPI_COMM_WORLD);

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Send all-to-all: \t\t%f\n", elapsed);
   }

   // Now all processes have sendCount and recvCount
   // Calculate the sendDispl and recvDispl arrays
   sDispl = rDispl = 0;
   for(i = 0; i < p_count; i++)
   {
      sendDispl[i] = sDispl; // assign displacement
      recvDispl[i] = rDispl;
      sDispl += sendCount[i]; // add counts for next displacement
      rDispl += recvCount[i];
   }

   // Allocate space for final sorted array
   finalSortedSize = rDispl; // Last rDispl is the full array size
   finalSorted = malloc(finalSortedSize*sizeof(int));

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Calc send/recv displ: \t\t%f\n", elapsed);
   }

   // Send final variable all-to-all message with the constructed arrays:
   // sendCount, sendDispl, recvCount, recvDispl
   // Along with the send and recv buffers
   MPI_Alltoallv(numbers, sendCount, sendDispl, MPI_INT,
                 finalSorted, recvCount, recvDispl, MPI_INT, MPI_COMM_WORLD);
				 
	// Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Send all-to-allv: \t\t%f\n", elapsed);
   }

   // Quicksort the final arrays
   quicksort(finalSorted, 0, finalSortedSize-1);

   // Time Interval
   if(mode==TIMER)
   {
      elapsed = getTimeLapse(&checkpoint);
      if(p_id==ROOT) printf("Final quicksort: \t\t%f\n", elapsed);
   }

   // BARRIER - Wait for all processes to finish before getting end time
   MPI_Barrier(MPI_COMM_WORLD);
   gettimeofday( &stop, NULL );   // Record end time

 //// DEBUG PRINT - SEND COUNTS, RECEIVE COUNTS, FINAL SORTED
   if(mode==DEBUG)
   {
      if(p_id==ROOT)
      {
         printf("\nSEND COUNTS\n");
         printf("==========================================\n");
      }
      // DEBUG Sorted Collections
      if(p_id) MPI_Recv(&flag, 1, MPI_INT, p_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      printf("Process %d: ", p_id);
      for(i = 0; i < p_count; i++)
      {
         printf("%d ", sendCount[i]);
      }
      printf("\n");
      if(p_id < (p_count-1)) MPI_Send(&flag, 1, MPI_INT, p_id+1, 1, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);


      if(p_id) MPI_Recv(&flag, 1, MPI_INT, p_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(p_id==ROOT)
	  {
         printf("\nRECEIVE COUNTS\n");
         printf("==========================================\n");
      }
      printf("Process %d: ", p_id);
      for(i = 0; i < p_count; i++)
      {
         printf("%d ", recvCount[i]);
      }
      printf("\n");
      if(p_id < (p_count-1)) MPI_Send(&flag, 1, MPI_INT, p_id+1, 1, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

      if(p_id) MPI_Recv(&flag, 1, MPI_INT, p_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(p_id==ROOT)
      {
         printf("\nFINAL SORTED\n");
         printf("==========================================\n");
      }
      printf("Process %d: ", p_id);
      for(i = 0; i < finalSortedSize; i++)
      {
         printf("%d ", finalSorted[i]);
      }
      printf("\n");

      sleep(1);
      if(p_id < (p_count-1)) MPI_Send(&flag, 1, MPI_INT, p_id+1, 1, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);

   }

   // PRINT FINAL TIME
   checkpoint = start;
   elapsed = getTimeLapse(&checkpoint);
   if(p_id==ROOT) printf("Final time: \t\t\t%f\n", elapsed);

   // Free all allocated memory
   free(sendCount);
   free(recvCount);
   free(sendDispl);
   free(recvDispl);
   free(numbers);
   free(samples);
   free(collective);
   free(finalSorted);

   MPI_Finalize(); // Cleanup MPI
   return 0; // exit
}

// Swap
// Simple swap routine which switches the values in the two references provided
// (input) int *x       first reference
// (input) int *y       second reference
// (output) void
void swap (int *x, int *y)
{
   int temp = *x; // temp holder
   *x = *y; // assign x to y value
   *y = temp; // assign y to held x value
}

// Subsort routine
// Supports quicksort and actually does the sorting
// Due to being the most time-consuming portion (see profiling run option),
// this routine utilizes OpenMP pragmas to assist in speedup.
// Uses last array value as the pivot which ends up in its final
// resting place before ending the routine.  Values that are less
// than or equal to the pivot are constantly pushed to the lower
// end of the array checking every item in the array.
// (input)  int *array          array to sort
// (input)  int low             lower index bound
// (input)  int high            upper index bound
// (output) int                 index of final pivot location
int subsort(int *array, int low, int high)
{
    int pivot = array[high];    // Pivot value
    int i = (low - 1);  // Index of last element found to be <= pivot
    // NOTE: Starting location of i is lower than array range
    int j; // Current iterator

    #pragma omp parallel for
    for (j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (array[j] <= pivot)
        {
            #pragma omp critical
            {
            i++;    // move to next insert location
            swap(&array[i], &array[j]); // swap to highest lower end
            }
        }
    }
    swap(&array[i + 1], &array[high]); // swap pivot to highest lower end
    return (i + 1); // send that location to quicksort
}

// Quicksort routine
// Pass in array and lower and upper index to sort
// Calls subsort which actually does the sorting sending
// back the index of the final resting place of the pivot
// value.  Partially sorted upper and lower array is then
// recursively sorted.
// (input)  int *array          array to sort
// (input)  int low             lower index bound
// (input)  int high            upper index bound
// (output) void
void quicksort(int *array, int low, int high)
{

   if (high > low)
    {
        // wall is the index where the pivot
        // value is placed at its final destination
        int wall = subsort(array, low, high);
		
        // Recurvisely call quicksort to fully
        // sort the partially sorted section
        // before and after the wall
        quicksort(array, low, wall - 1);
        quicksort(array, wall + 1, high);
    }
}

// Get time elapsed from last time recorded
// Must be called on all processes due to MPI_Barrier
// Returns the elapsed time and refreshes the checkpoint
// (input)  struct timeval *last     last time recorded (checkpoint)
// (output) float                    elapsed time
float getTimeLapse (struct timeval *last)
{
   struct timeval current; // current time
   float elapsed; // elapsed time

  // Wait for all processes
   MPI_Barrier(MPI_COMM_WORLD);

   // Get Current time
   gettimeofday( &current, NULL);
   elapsed = ((current.tv_sec-last->tv_sec) +
              (current.tv_usec-last->tv_usec)/(float)1000000 );

   // Update checkpoint value
   *last = current;
   // Return elapsed time
   return elapsed;
}
                            