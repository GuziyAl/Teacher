#include "matrix_operations.h"


void showMatrix(double **matrix,int strings, int columns)
{
    for (int i = 0; i < strings; ++i) {
        for (int j = 0; j < columns; ++j) {
            cout << setw(4) << matrix[i][j];
        }
        cout << endl;
    }
}

void showVector (double *vector, int neurons) {
    for (int i = 0;i < neurons; ++i) {
        cout << setw(2) << vector [i];
    }
}

double **transponate (double **matrix,int neurons, int weights) {

    double **transponated = crtLayer(weights, neurons);
    for (int i = 0; i < neurons; ++i) {
        for (int j = 0; j < weights; ++j) {
            transponated[j][i] = matrix [i][j];
        }
    }
    return transponated;
}

double *matrixAndVectorMultiplication(double **left,
                                      int neurons,
                                      int weights,
                                      double *right) {
    double *ans = new double [neurons];
    for (int i = 0; i < neurons; ++i) {
        double ammount = 0.0;
        for (int j = 0;j < weights; ++j) {
            ammount += (left[i][j] * right[j]);
        }
        ans[i] = ammount;
    }
    return ans;
}

void normalization (double *input_vector) {
    double first_cord = input_vector [0];
    double second_cord = input_vector [1];
    double third_cord = input_vector [2];
    double fourth_cord = input_vector [3];

    double lenght = sqrt((first_cord * first_cord) + (second_cord * second_cord) + (third_cord * third_cord) + (fourth_cord * fourth_cord));
    input_vector [0] = first_cord / lenght;
    input_vector [1] = second_cord / lenght;
    input_vector [2] = third_cord / lenght;
    input_vector [3] = fourth_cord / lenght;

}

double **crtLayer (int neurons, int weights) {
    double **layer;
    layer = new double* [neurons];
    for (int var = 0; var < neurons; ++var) {
        layer[var] = new double [weights];
    }
    return layer;
}

void deleteLayer(double **layer, int neurons) {
    for (int i = 0; i < neurons; i++) {
        delete [] layer[i];
    }
    delete [] layer;
}

