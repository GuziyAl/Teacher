#ifndef IOFLUX_H
#define IOFLUX_H

#include <string>
#include <fstream>
#include <cmath>
#include "matrix_operations.h"

using namespace std;

double crtNumber(char *num, int num_length);

void readSignalsAndOutFromFile (double *isignals, ifstream *fin);

void makeJournalNote (int epoch_num,
                      double **hidden_layer,
                      int hid_neurons,
                      int hid_weights,
                      double **outer_layer,
                      int out_neurons,
                      int out_weights,
                      double network_error);

#endif // IOFLUX_H
