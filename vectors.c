#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define K 3
#define VECTOR_SIZE 4000

typedef struct vector 
{
    double x[2];
}vector;

double randfrom(double min, double max) 
{
    return (double)rand()/RAND_MAX*2.0-1.0;
}

int main(int argc, char* argv[]) {
    vector train_vectors[VECTOR_SIZE];
    vector test_vectors[VECTOR_SIZE];

    //fill the train and test vectors with random numbers between -1 and 1
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        train_vectors[i].x[0] = randfrom(-1.0, 1.0);
        train_vectors[i].x[1] = randfrom(-1.0, 1.0);

        test_vectors[i].x[0] = randfrom(-1.0, 1.0);
        test_vectors[i].x[1] = randfrom(-1.0, 1.0);
    }

    FILE *train = fopen("train_vectors.txt", "w");
    FILE *test = fopen("test_vectors.txt", "w");
    if (train == NULL || test == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }


    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        fprintf(train,"%f\t%f\n", train_vectors[i].x[0], train_vectors[i].x[1]);

        fprintf(test,"%f\t%f\n", train_vectors[i].x[0], train_vectors[i].x[1]);
    }
    
    fclose(train);
    fclose(test);
} 