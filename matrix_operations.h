#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

void showVector (double *vector, int neurons);
void showMatrix(double **matrix,int strings, int columns);

double *matrixAndVectorMultiplication(double **left,
                                      int neurons,
                                      int weights,
                                      double *right);
double **transponate (double **matrix,int neurons, int weights);

void normalization (double *input_vector);

double **crtLayer (int neurons, int weights);
void deleteLayer(double **layer, int neurons);

#endif // MATRIX_OPERATIONS_H
