#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define d   2       /* Number of inputs */
#define p   3       /* Number of outputs */
#define K   3       /* Number of categories */
#define H1  7       /* Number of neurons in the first layer */
#define H2  4       /* Number of neurons in the second layer*/
#define H3  5       /* Number of neurons in the third layer */
#define HL  3       /* Number of hidden layers */
#define H   4       /* Number of layers including the output layer */
#define f   0       /* Type of activation function to be used (0 for logistic, 1 for tanh, 2 for relu) */
#define n   0.05   /* Learning rate */
#define a   1       /* Gradient for the activation functions*/
#define B   4000       /* Batch for gradient descent B=1 -> Stochastic , B=4000 -> Batch, B=500 -> mini-Batch*/

#define EPOCHS 700  /* Number of epochs before termination */
#define N    4000 /* Number of inputs per set */
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
    double error_signal;
    double delta_i;
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
void reverse_pass(Input_t x);
void gradient_descent();
void back_propagation(Input_t x);
double calculate_squared_error();
void reset_partial_derivatives();


/* ------------GLOBALS------------- */
Network_t network;
Input_t train_set[N];
Input_t test_set[N];

int neuronsPerLayer[4] = {H1, H2, H3, p}; // Hidden + output
double square_errors[B];
double partial_derivatives[WEIGHTS_NUM];  // This array will be used to update the weights after each batch.
double global_error = 0.0;
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

double calculate_activation_derivative(double y)
{
    switch(f)
    {
        case 0: return (y * (1 - y));
        case 1: return (1 - pow(y, 2));
        case 2: return (y <= 0) ? 0 : 1;
    }
}

void forward_pass(Input_t x)
{
    double u_i = 0.0;

    for (int h = 0; h < H; h++) //for each layer
    {
        if (h == 0) // first hidden layer has input from x
        {
            for (int j = 0; j < neuronsPerLayer[h]; j++)
            {
                u_i += network.layers[0].neuron[j].w[0]; // y_0; Bias
                u_i += network.layers[0].neuron[j].w[1] * x.x1;
                u_i += network.layers[0].neuron[j].w[2] * x.x2;

                network.layers[0].neuron[j].output = activation_function(u_i);
                
                u_i = 0.0;
            }
        }
        else //Second layer until output layer
        {
            for (int i = 0; i < neuronsPerLayer[h]; i++)
            {
                for (int j = 1; j <= neuronsPerLayer[h-1]; j++)
                {
                    u_i += network.layers[h].neuron[i].w[0];
                    u_i += network.layers[h].neuron[i].w[j] * network.layers[h-1].neuron[j-1].output;

                    network.layers[h].neuron[i].output = activation_function(u_i);

                    u_i = 0.0;
                }
            }
        }
    }

    for (int i = 0; i < p; i++)
    {
        network.layers[HL].neuron[i].error_signal = network.layers[HL].neuron[i].output - x.category[i];
    }
}

void reverse_pass(Input_t x)
{
    // Calculate δi for every neuron in the network
    for (int h = HL; h >= 0; h--)
    {
        if (h == HL) //Output layer
        {
            for (int i = 0; i < p; i++)
            {
                double u_i = network.layers[h].neuron[i].output;
                network.layers[h].neuron[i].delta_i = network.layers[HL].neuron[i].error_signal * calculate_activation_derivative(u_i);
            }
        }
        else //Hidden layers
        {
            for (int i = 0; i < neuronsPerLayer[h]; i++)
            {
                double weight_delta_sum = 0.0;

                //Calculate the sum of the weights and deltas to the next layer
                for (int j = 0; j < neuronsPerLayer[h+1]; j++)
                {
                    for (int l = 0; l < neuronsPerLayer[h]; l++)
                        weight_delta_sum += network.layers[h+1].neuron[j].w[l] * network.layers[h+1].neuron[j].delta_i;
                }

                double u_i = network.layers[h].neuron[i].output;
                network.layers[h].neuron[i].delta_i = calculate_activation_derivative(u_i) * weight_delta_sum;
            }
        }
    }
}

void calculate_partial_derivatives(Input_t x)
{
    //Calculate the error derivatives for each neuron
    for (int h = HL; h >= 0; h--)
    {
        if (h == 0) // first hidden layer
        {
            for (int i = 0; i < neuronsPerLayer[h]; i++)
            {
                network.layers[h].neuron[i].error_derivative[0] = network.layers[h].neuron[i].delta_i;
                network.layers[h].neuron[i].error_derivative[1] = network.layers[h].neuron[i].delta_i * x.x1;
                network.layers[h].neuron[i].error_derivative[2] = network.layers[h].neuron[i].delta_i * x.x2;
            }
        }
        else // rest of the layers
        {
            for (int i = 0; i < neuronsPerLayer[h]; i++)
            {
                network.layers[h].neuron[i].error_derivative[0] = network.layers[h].neuron[i].delta_i;
                

                for (int j = 0; j < neuronsPerLayer[h-1]; j++)
                {
                    network.layers[h].neuron[i].error_derivative[j+1] = network.layers[h].neuron[i].delta_i * network.layers[h-1].neuron[j].output;
                }
            }
        }
    }
}

void back_propagation(Input_t x)
{
    forward_pass(x);
    reverse_pass(x);
    calculate_partial_derivatives(x);
}

void reset_partial_derivatives()
{
    for (int i = 0; i < WEIGHTS_NUM; i++)
    {
        partial_derivatives[i] = 0.0;
    }
}

double calculate_squared_error()
{
    double sum = 0.0;
    
    for (int i = 0; i < N; i++)
    {
        forward_pass(train_set[i]);
        sum = 0.0;

        for (int j = 0; j < p; j++)
        {
            sum += pow(train_set[i].category[j] - network.layers[HL].neuron[j].output, 2);
        }

        sum /= 2.0;
    }
    return sum;
}

void gradient_descent()
{
    int cond = 0;
    int epoch = 0;       // We use this to check whether 700 epochs have passed
    int input_count = 0; // We use this counter to check whether an epoch has passed
    int p_d_counter = 0;
    double new_error = 0.0;
    double old_error = 0.0;

    while(epoch < EPOCHS)
    {        
        for (int b = 0; b < B; b++)
        {
            back_propagation(train_set[input_count]); // perform forward and reverse pass and calculate partial derivatives;
            input_count++;
            

            reset_partial_derivatives();

            for (int h = 0; h < H; h++)
            {
                for (int i = 0; i < neuronsPerLayer[h]; i++)
                {
                    if (h == 0) // first hidden layer
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            partial_derivatives[p_d_counter] += network.layers[h].neuron[i].error_derivative[j];
                            p_d_counter++;
                        }
                    }
                    else // All other layers
                    {
                        for (int j = 0; j < neuronsPerLayer[h-1]; j++)
                        {
                            partial_derivatives[p_d_counter] += network.layers[h].neuron[i].error_derivative[j];
                            p_d_counter++;
                        }
                    }
                }
            }
            p_d_counter = 0;
        }

        
        // Update the weights
        for (int h = 0; h < H; h++)
        {
            for (int i = 0; i < neuronsPerLayer[h]; i++)
            {
                if (h == 0) // First hidden layer
                {
                    for (int j = 0; j < 3; j++)
                    {
                        network.layers[h].neuron[i].w[j] -= n * partial_derivatives[p_d_counter];
                        p_d_counter++;
                    }
                }
                else
                {
                    for (int j = 0; j <= neuronsPerLayer[h-1]; j++)
                    {
                        network.layers[h].neuron[i].w[j] -= n * partial_derivatives[p_d_counter];
                        p_d_counter++;
                    }
                }
            }
        }
        p_d_counter = 0;

        if(input_count == N)  // If an epoch is done calculate global error
        {            
            new_error = calculate_squared_error();
            old_error = new_error;
            new_error = 0.0;
            printf("Global error: %lf\n", old_error);
            epoch++;
            input_count = 0;
        }

        cond++;
    }
}

void print_network()
{   
    int i,j,k;
    int layer, weights;

    for (i = 0; i < H; i++)
    {   
        printf("[LAYER %d] with %d neurons\n",i ,neuronsPerLayer[i]);
        for(j = 0; j < neuronsPerLayer[i]; j++)
        {
            printf("\t[NEURON %d]\n",j);
            weights = (i == 0) ? 3 : neuronsPerLayer[i-1] + 1;

            for(k = 0; k < weights; k++)
                printf("\t\t Weight[%d] = %lf\n",k,network.layers[i].neuron[j].w[k]);

            printf("\t\t ------- \n");

            for(k = 0; k < weights; k++)
                printf("\t\t Error Derivative[%d] = %lf\n",k,network.layers[i].neuron[j].error_derivative[k]);

            printf("\t\t ------- \n");

            printf("\t\t Output = %lf\n", network.layers[i].neuron[j].output);

            printf("\t\t ------- \n");

            printf("\t\t Delta = %.20lf\n", network.layers[i].neuron[j].delta_i);
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

    //Allocate memory for the error derivatives and the weights and initialize the weights
    int lastLayerNeurons;
    for (int i = 0; i < H; i++) // for every layer including the output (when i = 3)
    {
        for (int j = 0; j < neuronsPerLayer[i]; j++) // for every neuron in every layer
        {
            lastLayerNeurons = (i == 0) ? 3 : neuronsPerLayer[i-1] + 1; // The +1 is for the Bias 

            network.layers[i].neuron[j].w = malloc(sizeof(double) * lastLayerNeurons);
            network.layers[i].neuron[j].error_derivative = malloc(sizeof(double) * lastLayerNeurons); 
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
                printf("Layer %d, neuron %d, weight[%d] = %lf\n", i, j, k, network.layers[i].neuron[j].w[k]);
            }
        }
    }
}

//This function sets up the train and test vectors 
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
    for (int i = 0; i < N; i++) 
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

