#include <iostream>
#include <iomanip>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <cmath>
#include <cstdio>

enum{
    OUT_L = 1,
    HID_L = 0
};

double error(double *output, double *solution, int n){
    double error = 0, delta;
    for(int i = 0; i < n; ++i){
        delta = output[i] - solution[i];
        error += delta * delta;
    }
    return error / 2;
}

double sigmoid(double input){
    return 1 / (1 + exp(-input));
}

double summ(double *signal, double *weight, int n){
    double sum = 0;
    for(int i = 0; i < n; ++i){
        sum += signal[i] * weight[i];
    }
    return sum;
}


void backpropagation(double ***weight, double **output, double *solution, int *neurons, int layers, double moment, double speed)
{
    //current layer index
    int cli = layers - 1;
    double sum;
    //allocting deltas array
    double **delta = new double* [layers];
    for (int i = 0; i < layers; ++i){
        delta[i] = new double [neurons[i]];
    }
    //computing deltal on output layer
    for (int i = 0; i < neurons[cli]; ++i){
        delta[cli][i] = -2 * moment * output[cli][i] * (1 - output[cli][i]) * (solution[i] - output[cli][i]);
    }
    //computing deltas on hidden layers
    while (--cli){
        for (int i = 0; i < neurons[cli]; ++i){
            sum = 0;
            for (int j = 0; j < neurons[cli + 1]; ++j){
                sum += delta[cli + 1][j] * weight[cli][j][i];
            }
            delta[cli][i] = 2 * moment * output[cli][i] * (1 - output[cli][i]) * sum;
        }
    }
    //recomputing weights
    while(++cli < layers){
        for (int i = 0; i < neurons[cli]; ++i){
            for (int j = 0; j < neurons[cli - 1]; ++j){
                weight[cli][i][j] -= speed * delta[cli][i] * output[cli - 1][j];
            }
        }
    }
    // deleting memory of delta array
    for (int i = 0; i < layers; ++i){
        delete[] delta[i];
    }
    delete[] delta;
}

void backpropagation2(double ***weight, double ***dweight, double **output, double *solution, int *neurons, int layers, double moment, double speed)
{
    //current layer index
    int cli = layers - 1;
    double sum;
    //allocting deltas array
    double **delta = new double* [layers];
    for (int i = 0; i < layers; ++i){
        delta[i] = new double [neurons[i]];
    }
    //computing deltal on output layer
    for (int i = 0; i < neurons[cli]; ++i){
        delta[cli][i] = -1 * output[cli][i] * (1 - output[cli][i]) * (solution[i] - output[cli][i]);
    }
    //computing deltas on hidden layers
    while (--cli){
        for (int i = 0; i < neurons[cli]; ++i){
            sum = 0;
            for (int j = 0; j < neurons[cli + 1]; ++j){
                sum += delta[cli + 1][j] * weight[cli][j][i];
            }
            delta[cli][i] = output[cli][i] * (1 - output[cli][i]) * sum;
        }
    }
    //recomputing weights
    while(++cli < layers){
        for (int i = 0; i < neurons[cli]; ++i){
            for (int j = 0; j < neurons[cli - 1]; ++j){
                weight[cli][i][j] += moment * dweight[cli][i][j] + (1 - moment) * speed * delta[cli][i] * output[cli - 1][j];
            }
        }
    }
    // deleting memory of delta array
    for (int i = 0; i < layers; ++i){
        delete[] delta[i];
    }
    delete[] delta;
}



int main()
{

    return 0;
}
