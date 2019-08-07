#include "networking.h"

double activationFunction(double net)
{
    double tmp_1 = pow(E,(-1 * net));
    double tmp_2 = 1.0 + tmp_1;
    double tmp = 1.0 / tmp_2;
    return tmp;
}

double outerLayerDelta (double target, double outer_out) {
    return (outer_out * (1.0 - outer_out)) * (target - outer_out);
}

double hiddenLayerDelta (double hidden_out, double sub_delta) {
    return (hidden_out * (1.0 - hidden_out)) * sub_delta;
}

void weightsCorrection (double **layer,
                        int neuron_num,
                        int weights,
                        double *deltas) {
    for (int i = 0; i < weights; ++i) {
        layer[neuron_num][i] += deltas[i];
    }
}

double exampleError (double *targets_list,
                     int outer_num,
                     double *outer_outs) {
    double tmp_ammount = 0.0;
    for (int i = 0; i < outer_num; ++i) {
        tmp_ammount += pow((targets_list[i] - outer_outs[i]),2.0);
    }
    return (1.0 / outer_num) * (tmp_ammount);
}

void evaluateErrorsVector(double *errors_vector, double delta, double *prev_layer_out, double learning_speed) {
    for (int i = 0; i < 4; ++i) {
        errors_vector [i] = learning_speed * delta * prev_layer_out[i];
    }
}
















