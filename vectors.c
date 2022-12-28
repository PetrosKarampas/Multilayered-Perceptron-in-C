#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 4000

double x1, x2;

int main()
{
    srand(time(NULL));
   
    FILE *train = fopen("train_vectors.txt", "w");
    FILE *test = fopen("test_vectors.txt", "w");

    if (train == NULL || test == NULL)
    {
        printf("Error opening file!\n");
        return 1;
    }

    for (int i = 0; i < VECTOR_SIZE; i++)
    {   
        x1 = ((double)rand()/(double)(RAND_MAX)) * 2.0 - 1.0;
        x2 = ((double)rand()/(double)(RAND_MAX)) * 2.0 - 1.0;

        fprintf(train,"%f\t%f\n", x1, x2);

        x1 = ((double)rand()/(double)(RAND_MAX)) * 2.0 - 1.0;
        x1 = ((double)rand()/(double)(RAND_MAX)) * 2.0 - 1.0;
       
        fprintf(test,"%f\t%f\n", x1, x2);
    }

    fclose(train);
    fclose(test);

    return 0;
} 