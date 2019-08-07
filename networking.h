#ifndef NETWORKING_H
#define NETWORKING_H

#include <cmath>
#define E 2.7182818284590452354

double activationFunction(double net);

double outerLayerDelta (double target, double outer_out);
double hiddenLayerDelta (double hidden_out, double sub_delta);

void weightsCorrection (double **layer,
                        int neuron_num,
                        int weights,
                        double *deltas);

double exampleError (double *targets_list,
                     int outer_num,
                     double *outer_outs);

void evaluateErrorsVector(double *errors_vector, double delta, double *prev_layer_out, double learning_speed);

#endif // NETWORKING_H
