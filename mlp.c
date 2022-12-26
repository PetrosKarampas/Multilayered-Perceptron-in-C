#include <stdio.h>
#include <math.h>

#define LAYERS_NUM 3
#define INPUT_NUM 5

//Neuron struct for each neuron in the network
typedef struct neuron
{
    double polarization = 1.0;
    double input[2];
    double w[INPUT_NUM];
}neuron;

//  layer struct that contains the neurons for each layer;
typedef struct layer
{
    neuron* neurons;
}layer;

char buffer[2000] = {0};

int main(int argc, char* argv[]) {

    layer layers[LAYERS_NUM];

    int c;
    FILE *file;
    file = fopen("train_vectors.txt", "r");

    if(!file)
    {
        printf("file cold not be opened");
        return 1;
    }
    int result = fscanf(file, "%s", buffer);
    while(result != EOF)
    {
        printf("%s\n", buffer);
        result = fscanf(file, "%s", buffer);
    }

    fclose(file);


    //read line from a file?


}

