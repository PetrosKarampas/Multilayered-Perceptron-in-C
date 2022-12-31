#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define d   2       /* Number of inputs */
#define p   3       /* Number of outputs */
#define K   3       /* Number of categories */
#define H1  5       /* Number of neurons in the first layer */
#define H2  5       /* Number of neurons in the second layer*/
#define H3  4       /* Number of neurons in the third layer */
#define HL  3       /* Number of hidden layers */
#define H   4       /* Number of layers including the output layer */
#define f   0       /* Type of activation function to be used (0 for logistic, 1 for tanh, 2 for relu) */
#define n   0.005   /* Learning rate */
#define a   1       /* Gradient for the activation functions*/

#define EPOCHS 700  /* Number of epochs before termination */
#define SET    4000 /* Number of inputs per set */
#define WEIGHTS_NUM  H1*(d+1) + H2*(H1+1) + H3*(H2+1) + p*(H3+1)  /* Total number of weights in the network */
#define RANDOM_DOUBLE(A,B) ((double)rand()/(double)(RAND_MAX)) * (B-A) + A;


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
    double *weights;
    //double theta;
    double output;
    double delta;
}Neuron_t;

//  layer struct that contains the neurons for each layer;
typedef struct layer
{
    Neuron_t* neurons;
}Layer_t;

typedef struct Network
{
    Layer_t layers[H]; /* Hidden + output layers */
}Network_t;

void initialize();
void setup();
void encode_input(double x1, double x2, Input_t *input, int i);
void categorize(char *category, Input_t *input, int i);


/* ------------GLOBALS------------- */
Network_t network;
Input_t train_set[SET];
Input_t test_set[SET];
int neuronsPerLayer[4] = {H1, H2, H3, p}; // Hidden + output


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

double activation_function(double sum)
{
    switch(f)
    {
        case 0: // Logistic 
            return 1/(1+exp(-a*sum));
        case 1: // tanh
            return (exp(a*sum) - exp(-a*sum)) / (exp(a*sum) + exp(-a*sum));
        case 2: // relu
            return sum > 0.0 ? sum : 0.0;
    }
}

void forward_pass(Input_t x)
{
    for (int i = 0; i < H; i++)
    {
        if (i == 0)
        {
            for ( int j = 0; j < neuronsPerLayer[i]; j++)
            {
                double sum;
                sum += network.layers[i].neurons[j].weights[0] * 1.0;   // The bias input is always 1.
                sum += network.layers[i].neurons[j].weights[1] * x.x1;  
                sum += network.layers[i].neurons[j].weights[2] * x.x2;

                network.layers[i].neurons[j].output = activation_function(sum);
                printf("%lf \n", network.layers[i].neurons[j].output);
            }
        }
        else
        {
            for ( int j = 0; j < neuronsPerLayer[i]; j++)
            {
                double sum;
                sum += network.layers[i].neurons[j].weights[0] * 1.0;
                for (int k = 0; k < neuronsPerLayer[i-1]; k++)
                {
                    sum += network.layers[i-1].neurons[j].output * network.layers[i].neurons[j].weights[k];
                }
                network.layers[i].neurons[j].output = activation_function(sum);
                printf("Hidden layer %d output: %lf \n", i+1,  network.layers[i].neurons[j].output);
            }
        }
    }
}

void backprop()
{
    /*
    TODO
    */
}

/* This method allocates the necessary memory for the while network
 * layers, neurons etc. and initialize the weights of all the neurons (Hidden + output)
 * with a random number in the closed set [-1, 1].
 */
void initialize()
{
    srand(time(NULL)); //This is so the random generted weights are different every time.
    
    // Allocate memory for all the neurons in the network.
    for (int i = 0; i < H ; i++)
    {
        network.layers[i].neurons = (Neuron_t *) malloc(sizeof(Neuron_t) * neuronsPerLayer[i]);

        if(network.layers[i].neurons == NULL)
        {
            exit(1);
        }
    }

    //Allocate memory for the weights and initialize them
    int lastLayerNeurons;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < neuronsPerLayer[i]; j++)
        {
            lastLayerNeurons = (i == 0) ? 2 + 1 : neuronsPerLayer[i-1] + 1; // The +1 is for the Bias in both occations

            network.layers[i].neurons[j].weights = (double *) malloc(sizeof(double) * lastLayerNeurons); 
            if(network.layers[i].neurons[j].weights == NULL)
            {
                printf("Memory allocation for the weights failed!");
                exit(0);
            }

            for (int k = 0; k < lastLayerNeurons; k++)
            {
                network.layers[i].neurons[j].weights[k] = RANDOM_DOUBLE(-1.0, 1.0);
                printf("W: %lf\n", network.layers[i].neurons[j].weights[k]);
            }
        }
    }
}


void setup()
{
    FILE *train = fopen("train_vectors.txt", "r");
    FILE *test = fopen("test_vectors.txt", "r");

    if(train == NULL || test == NULL)
    {
        printf("file could not be opened");
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

    setup();
    initialize();
    forward_pass(train_set[1]);

    return 0;
}

