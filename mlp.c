#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define d   2       /* Number of inputs */
#define p   3       /* Number of outputs */
#define K   3       /* Number of categories */
#define H1  5       /* Number of neurons in the first layer */
#define H2  3       /* Number of neurons in the second layer*/
#define H3  2       /* Number of neurons in the third layer */
#define HL  3       /* Number of hidden layers */
#define H   4       /* Number of layers including the output layer */
#define f   1       /* Type of activation function to be used (0 for logistic, 1 for tanh, 2 for relu) */
#define n   0.005   /* Learning rate */
#define a   1       /* Gradient for the activation functions*/
#define B   300     /* Batch for gradient descent*/

#define EPOCHS 700  /* Number of epochs before termination */
#define SET    4000 /* Number of inputs per set */
#define WEIGHTS_NUM  H1*(d+1) + H2*(H1+1) + H3*(H2+1) + p*(H3+1)  /* Total number of weights in the network */
#define RANDOM_DOUBLE(A,B) ((double)rand()/(double)(RAND_MAX)) * (B-A) + A;


// Neuron struct for each neuron in the network
typedef struct Input
{
    double x1;
    double x2;
    int category[3]; // 1-out of-p encoding, C1 = {1, 0, 0} || C2 = {0, 1, 0} || C3 = {0, 0, 1}
}Input_t;

typedef struct Neuron_t
{
    double *w;
    double *error_derivative;
    double output;
    double delta;
}Neuron_t;

//  layer struct that contains the neurons for each layer;
typedef struct layer
{
    Neuron_t *neuron;
}Layer_t;

typedef struct Network
{
    Layer_t layers[H]; /* Hidden + output layers */
}Network_t;

void initialize();
void setup();
void encode_input(double x1, double x2, Input_t *input, int i);
void categorize(char *category, Input_t *input, int i);
void reverse_pass(Input_t x, int *t);
void gradient_descent();



/* ------------GLOBALS------------- */
Network_t network;
Input_t train_set[SET];
Input_t test_set[SET];

int neuronsPerLayer[4] = {H1, H2, H3, p}; // Hidden + output
double error_derivatives[WEIGHTS_NUM];
/* ------------GLOBALS------------- */

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
        case 0: return 1/(1+exp(-a*sum));                                       // Logistic 
        case 1: return (exp(a*sum) - exp(-a*sum)) / (exp(a*sum) + exp(-a*sum)); // tanh
        case 2: return sum > 0.0 ? sum : 0.0;                                   // relu
    }
}

void forward_pass(Input_t x)
{
    for (int i = 0; i < H; i++)
    {
        if (i == 0)
        {
            //This for loop is for the first layer only hence the x1 and x2 inputs. 
            for ( int j = 0; j < neuronsPerLayer[i]; j++)
            {
                double sum;
                sum += network.layers[i].neuron[j].w[0] * 1.0;   // The bias input is always 1. and w[0] is always the bias of each neuron
                sum += network.layers[i].neuron[j].w[1] * x.x1;  
                sum += network.layers[i].neuron[j].w[2] * x.x2;

                network.layers[i].neuron[j].output = activation_function(sum);
            }
        }
        else
        {
            for ( int j = 0; j < neuronsPerLayer[i]; j++)
            {
                double sum;
                sum += network.layers[i].neuron[j].w[0] * 1.0; 
                for (int k = 0; k < neuronsPerLayer[i-1]; k++)
                {
                    sum += network.layers[i-1].neuron[j].output * network.layers[i].neuron[j].w[k];
                }
                if(i == H-1) // Output layer
                {
                    network.layers[i].neuron[j].output = 1/(1+exp(-a*sum)); // Always use logistic function because it gives result from 0 to 1
                }                                                            // which is what we need in this case
                else
                {
                    network.layers[i].neuron[j].output = activation_function(sum);
                }
            }
        }
    }
}

double calculate_delta_derivative(double y)
{
    switch(f)
    {
        case 0: return y * (1 - y);
        case 1: return (1 - pow(y, 2));
        case 2: return (y <= 0) ? 0 : 1;
    }
}

void backpropagation(Input_t x) // vector t={1, 0, 0} || {0, 1, 0} || {0, 0, 1} target output
{
    forward_pass(x);
    //Calculate deltas and new weights for every neuron.
    for (int i = HL; i >= 0; i--)
    {
        if (i == HL) // Output layer
        {
            // Calculate the partial derivative for the bias for every neuron in the output layer
            for(int j = 0; j < p; j++)
            {
                double y = network.layers[i].neuron[j].output;
                network.layers[i].neuron[j].delta = y * (1 - y) * (y - x.category[j]);
                network.layers[i].neuron[j].error_derivative[0] = network.layers[i].neuron[j].delta; //Bias 
                
                for (int l = 0; l < neuronsPerLayer[i-1]; l++)
                {
                    network.layers[i].neuron[j].error_derivative[l+1] = network.layers[i].neuron[j].delta * network.layers[i-1].neuron[l].output;
                }
                    
            }
        }   
        else //Hidden layers
        {
            for(int j = 0; j < neuronsPerLayer[i]; j++) // Hidden layers
            {
                double weight_delta_sum;

                for (int k = 0; k < neuronsPerLayer[i+1]; k++) // for each neuron of the next layer    
                {                                                                                                       /*                       */ 
                    weight_delta_sum += network.layers[i+1].neuron[k].w[j] * network.layers[i+1].neuron[k].delta;       /*  Σ(w_(h+1) * δ_(h+1)) */ 
                }                                                                                                       /*                       */ 
                                     
                double y = network.layers[i].neuron[j].output;
                network.layers[i].neuron[j].delta = calculate_delta_derivative(y)*(weight_delta_sum);
                printf("Delta_%d_%d: %lf\n", i+1, j+1, network.layers[i].neuron[j].delta);
            }

            //Calculate error derivatives for the first hidden layer
            if(i == 0)
            {
                for (int l = 0; l < neuronsPerLayer[i]; l++)
                {
                    network.layers[i].neuron[l].error_derivative[0] = network.layers[i].neuron[l].delta;
                    network.layers[i].neuron[l].error_derivative[1] = network.layers[i].neuron[l].delta * x.x1;
                    network.layers[i].neuron[l].error_derivative[2] = network.layers[i].neuron[l].delta * x.x2;
                }
            }
            else
            {
                for (int l = 0; l < neuronsPerLayer[i]; l++)
                {
                    network.layers[i].neuron[l].error_derivative[0] = network.layers[i].neuron[l].delta;

                    for (int j = 0; j < neuronsPerLayer[i-1]; j++)
                    {
                        network.layers[i].neuron[l].error_derivative[j+1] = network.layers[i].neuron[l].delta * network.layers[i-1].neuron[j].output;
                    }
                }
            }
        }
    }
}


void gradient_descent()
{
        
    FILE *error_file = fopen("test_vectors.txt", "w+");
    if(error_file == NULL)
    {
        printf("error file could not be opened!!");
        exit(1);
    }

    for (int i = 0; i < 1; i++)
    {
        // Reset errors.
        int errors_i = 0;
        for (int j = 0; j < WEIGHTS_NUM; j++)
        {
            error_derivatives[j] = 0.0;
        }
        
        //Batch
        for (int j = 0; j < B; j++)
        {
            
            backpropagation(train_set[j]);

            for (int i_layer = 0; i_layer < H; i_layer++)
            {
                for (int i_neuron = 0; i_neuron < neuronsPerLayer[i_layer]; i_neuron++)
                {
                    int k = (i_layer == 0) ? 2 : neuronsPerLayer[i_layer - 1];
                    for (int l = 0; l <= k; l++)
                    {
                        error_derivatives[errors_i] += network.layers[i_layer].neuron[i_neuron].error_derivative[l];
                        errors_i++;
                    }
                }
            }
            errors_i = 0.0; //Reset the index of the error_derivatives array so tey can be updated in the next iterration.
        }

        //Update weights
        errors_i = 0.0; //Reset it se we can update the weights in the network.
        for (int i_layer = 0; i_layer < H; i_layer++)
        {
            for (int i_neuron = 0; i_neuron < neuronsPerLayer[i_layer]; i_neuron++)
            {
                int k = (i_layer == 0) ? 2 : neuronsPerLayer[i_layer - 1];
                for (int l = 0; l <= k; l++)
                {
                    network.layers[i_layer].neuron[i_neuron].w[l] -= n * error_derivatives[errors_i];
                    printf(" new weight[%d]: %lf\n", errors_i, network.layers[i_layer].neuron[i_neuron].w[l]);
                    errors_i++;
                }
            }
        }
        errors_i = 0.0;;
    }
}

void print_network()
{   
    int i,j,k;
    int layer;

    for (i = 0; i < H; i++)
    {   
        printf("[LAYER %d] with %d neurons\n",i ,neuronsPerLayer[i]);
        for(j = 0; j < neuronsPerLayer[i]; j++)
        {
            printf("\t[NEURON %d]\n",j);
            layer  = ( i == 0 ) ? 2 : neuronsPerLayer[i];

            for(k = 0; k <= layer; k++)
                printf("\t\t Weight[%d] = %lf\n",k,network.layers[i].neuron[j].w[k]);

            printf("\t\t ------- \n");

            for(k = 0; k <= layer; k++)
                printf("\t\t Error Derivative[%d] = %lf\n",k,network.layers[i].neuron[j].error_derivative[k]);

            printf("\t\t ------- \n");

            printf("\t\t Output = %lf\n", network.layers[i].neuron[j].output);

            printf("\t\t ------- \n");

            printf("\t\t Delta = %.20lf\n", network.layers[i].neuron[j].delta);
        }
    }

    printf(" -------------------------------------------------------------\n");
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
        network.layers[i].neuron = (Neuron_t *) malloc(sizeof(Neuron_t) * neuronsPerLayer[i]);

        if(network.layers[i].neuron == NULL)
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

            network.layers[i].neuron[j].w = (double *) malloc(sizeof(double) * lastLayerNeurons);
            network.layers[i].neuron[j].error_derivative = (double *) malloc(sizeof(double) * lastLayerNeurons); 
            if(network.layers[i].neuron[j].w == NULL)
            {
                printf("Memory allocation for the weights failed!");
                exit(0);
            }
            if(network.layers[i].neuron[j].error_derivative == NULL)
            {
                printf("Memory allocation for the new weights failed!");
                exit(0);
            }

            for (int k = 0; k < lastLayerNeurons; k++)
            {
                network.layers[i].neuron[j].w[k] = RANDOM_DOUBLE(-1.0, 1.0);
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
    
    gradient_descent();
        
    print_network();

    return 0;
}

