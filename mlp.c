#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define d   2   /* Number of inputs */
#define K   3   /* Number of categories */
#define H1  5   /* Number of neurons in the first layer */
#define H2  3   /* Number of neurons in the second layer*/
#define H3  3   /* Number of neurons in the third layer */
#define HL  3   /* Number of hidden layers */
#define f   0   /* type of activation function to be used (0 for logistic, 1 for tanh, 2 for relu) */

#define EPOCHS 700  /* Number of epochs before termination */
#define SET    4000 /* Number of inputs per set */


//Neuron struct for each neuron in the network

typedef struct Input
{
    double x1;
    double x2;
    int category[3]; // 1-out of-p encoding, C1 = {1, 0, 0} || C2 = {0, 1, 0} || C3 = {0, 0, 1}
}Input_t;

typedef struct Neuron_t
{
    double w;
}Neuron_t;

//  layer struct that contains the neurons for each layer;
typedef struct layer
{
    Neuron_t* neurons;
}layer;

void initialize();
void encode_input(double x1, double x2, Input_t *input, int i);
void categorize(char *category, Input_t *input, int i);

Input_t train_set[SET];
Input_t test_set[SET];


/*
 * This function encodes each input according to their respective category 
 * using the 1-out of-p encoding method
 */
void categorize(char *category, Input_t *input, int i)
{
    if(strcmp(category, "C1") == 0)
    {
        input[i].category[0] = 1;
        input[i].category[1] = 0;
        input[i].category[2] = 0;
    }
    else if (strcmp(category, "C2") == 0)
    {
        input[i].category[0] = 0;
        input[i].category[1] = 1;
        input[i].category[2] = 0;
    }
    else if (strcmp(category, "C3") == 0)
    {
        input[i].category[0] = 0;
        input[i].category[1] = 0;
        input[i].category[2] = 1;
    }
}

/* 
 *This function organizes each input in a category 
 */
void encode_input(double x1, double x2, Input_t *input, int i)
{
    if      (pow((x1 - 0.5),2) + pow((x2 - 0.5),2) < 0.2 && x2 > 0.5)  categorize("C1", input, i);  
    else if (pow((x1 - 0.5),2) + pow((x2 - 0.5),2) < 0.2 && x2 < 0.5)  categorize("C2", input, i);  
    else if (pow((x1 + 0.5),2) + pow((x2 + 0.5),2) < 0.2 && x2 > -0.5) categorize("C1", input, i);  
    else if (pow((x1 + 0.5),2) + pow((x2 + 0.5),2) < 0.2 && x2 < -0.5) categorize("C2", input, i);  
    else if (pow((x1 - 0.5),2) + pow((x2 + 0.5),2) < 0.2 && x2 > -0.5) categorize("C1", input, i);  
    else if (pow((x1 - 0.5),2) + pow((x2 + 0.5),2) < 0.2 && x2 < -0.5) categorize("C2", input, i);  
    else if (pow((x1 + 0.5),2) + pow((x2 - 0.5),2) < 0.2 && x2 > 0.5)  categorize("C1", input, i);  
    else if (pow((x1 + 0.5),2) + pow((x2 - 0.5),2) < 0.2 && x2 < 0.5)  categorize("C2", input, i);  
    else categorize("C3", input, i);
}

void initialize()
{
    FILE *train = fopen("train_vectors.txt", "r");
    FILE *test = fopen("test_vectors.txt", "r");

    if(train == NULL || test == NULL)
    {
        printf("file cold not be opened");
        exit(1);
    }

    double x1, x2;
    for (int i = 0; i < SET; i++) 
    {
        fscanf(train, "%lf", &x1);
        fscanf(train, "%lf", &x2);
        
        train_set[i].x1 = x1;
        train_set[i].x2 = x2;
        encode_input(x1, x2, train_set, i);

        fscanf(test, "%lf", &x1);
        fscanf(test, "%lf", &x2);
        
        test_set[i].x1 = x1;
        test_set[i].x2 = x2;
        encode_input(x1, x2, test_set, i);
    }

    fclose(train);
    fclose(test);
}

int main() {

    initialize();
}

