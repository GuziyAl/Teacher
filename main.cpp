#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include "matrix_operations.h"
#include "networking.h"
#include "ioflux.h"

// Settings
#define EPOCHS 500000
#define LEARNING_RATE 0.2
#define ERROR_MIN_VALUE 0.00001     // 0.2 (Tested value)
#define NUMBER_OF_EXAMPLES 15.0

using namespace std;

int main()
{

    double **hidden = crtLayer(4, 4); // 4 hidden layer neurons/ input_signals_number(or number of neurons on prev layer)

    hidden [0][0] = -0.5;
    hidden [0][1] = -1.0;
    hidden [0][2] = -0.5;
    hidden [0][3] = -0.5;

    hidden [1][0] = -0.5;
    hidden [1][1] = -0.5;
    hidden [1][2] = -0.5;
    hidden [1][3] = 0.5;

    hidden [2][0] = 1.0;
    hidden [2][1] = -1.0;
    hidden [2][2] = 0.5;
    hidden [2][3] = -1.0;

    hidden [3][0] = -0.5;
    hidden [3][1] = 0.5;
    hidden [3][2] = 1.0;
    hidden [3][3] = 0.5;

    cout << "Hidden Random: " << endl;
    showMatrix(hidden, 4, 4);

    cout << endl;

    double **outer = crtLayer(3 , 4); // 3 outer layer neurons/ number of neurons on last hidden layer layer

    outer [0][0] = -0.5;
    outer [0][1] = -0.5;
    outer [0][2] = -1.0;
    outer [0][3] = -0.5;

    outer [1][0] = 1.0;
    outer [1][1] = 0.5;
    outer [1][2] = -1.0;
    outer [1][3] = -0.5;

    outer [2][0] = 0.5;
    outer [2][1] = -1.0;
    outer [2][2] = -1.0;
    outer [2][3] = 1;

    cout << "Outer Random: " << endl;
    showMatrix(outer, 3, 4);

    double input_signals [7]; // input buffer
    double targets_list [3];

    double hidden_out [4]; // Hidden layer values that differs from weighted ammounts due activ. function
    double hidden_deltas [4]; // Hidden error
    double hidden_first_errors [4];
    double hidden_second_errors [4];
    double hidden_third_errors [4];
    double hidden_fourth_errors [4];

    double outer_out[3];
    double outer_deltas [3];
    double outer_first_errors [4];
    double outer_second_errors [4];
    double outer_third_errors [4];

    double network_error = 1000; // 1000 -- kinda error code, if n_e == 1000 there are some problems with errorFunc evaluation
    for (int i = 0; i < EPOCHS; ++i) {
        ifstream fin ("C:\\Learning\\list_to_learn.txt");

        cout << endl << ">>>> Epoch #" << i + 1 << " <<<<" << endl << endl;

        double ex_errors_ammount = 0.0;

        for (int j = 0; j < NUMBER_OF_EXAMPLES; ++j) {

            readSignalsAndOutFromFile (input_signals, &fin);
            normalization(input_signals);

            targets_list[0] = input_signals[4];
            targets_list[1] = input_signals[5];
            targets_list[2] = input_signals[6];

            double input_vector [4];
            for (int i = 0; i < 4; ++i) {
                input_vector[i] = input_signals[i];
            }

            double *hidden_net = matrixAndVectorMultiplication(hidden, 4, 4, input_vector);
            for (int i = 0; i < 4; ++i) {
                hidden_out[i] = activationFunction(hidden_net[i]);
            }
            delete [] hidden_net;

            double *outer_net = matrixAndVectorMultiplication(outer, 3, 4, hidden_out);
            for (int i = 0; i < 3; ++i) {
                outer_out[i] = activationFunction(outer_net[i]);
            }
            delete [] outer_net;

            for (int i = 0; i < 3; ++i) {
                outer_deltas[i] = outerLayerDelta(targets_list[i], outer_out[i]);
            }

            evaluateErrorsVector(outer_first_errors, outer_deltas[0], hidden_out, LEARNING_RATE);
            evaluateErrorsVector(outer_second_errors, outer_deltas[1], hidden_out, LEARNING_RATE);
            evaluateErrorsVector(outer_third_errors, outer_deltas[2], hidden_out, LEARNING_RATE);

            double **transponed_outer = transponate(outer, 3, 4);
            double *sub_hidden_deltas = matrixAndVectorMultiplication(transponed_outer, 4, 3, outer_deltas);
            deleteLayer(transponed_outer, 4);

            for (int i = 0; i < 4; ++i) {
                hidden_deltas [i] = hiddenLayerDelta(hidden_out [i], sub_hidden_deltas[i]);
            }
            delete [] sub_hidden_deltas;

            evaluateErrorsVector(hidden_first_errors, hidden_deltas[0], input_vector, LEARNING_RATE);
            evaluateErrorsVector(hidden_second_errors, hidden_deltas[1], input_vector, LEARNING_RATE);
            evaluateErrorsVector(hidden_third_errors, hidden_deltas[2], input_vector, LEARNING_RATE);
            evaluateErrorsVector(hidden_fourth_errors, hidden_deltas[3], input_vector, LEARNING_RATE);

            weightsCorrection(outer, 0, 4, outer_first_errors);
            weightsCorrection(outer, 1, 4, outer_second_errors);
            weightsCorrection(outer, 2, 4, outer_third_errors);

            weightsCorrection(hidden, 0, 4, hidden_first_errors);
            weightsCorrection(hidden, 1, 4, hidden_second_errors);
            weightsCorrection(hidden, 2, 4, hidden_third_errors);
            weightsCorrection(hidden, 3, 4, hidden_fourth_errors);

            double error = exampleError(targets_list, 3, outer_out);
            cout << "Example error. Example #" << j + 1 << ":  " << error << endl;
            ex_errors_ammount += error;

        }
        fin.close();

        ofstream ferr ("C:\\Learning\\errors.txt",ios_base :: app);
        network_error = (ex_errors_ammount / NUMBER_OF_EXAMPLES);
        ferr << network_error << endl;
        cout << "Network error. Epoch #" << i + 1 << ":  " << network_error << endl;

        if (network_error < ERROR_MIN_VALUE) {
            makeJournalNote(i+1, hidden, 4, 4, outer, 3, 4, network_error);
            break;
        }
        else if (i == (100000-1) || i == (250000-1) || i == (750000 - 1) || i == (EPOCHS - 1)) { // Watch points
            makeJournalNote(i+1, hidden, 4, 4, outer, 3, 4, network_error);
        }

    }
    makeJournalNote(EPOCHS, hidden, 4, 4, outer, 3, 4, network_error);

    deleteLayer(hidden, 4);
    deleteLayer(outer, 3);

    return 0;
}

//double **arr = crtLayer(3,4);
//for (int i = 0; i < 3; ++i) {
//    for (int j = 0; j < 4; ++j) {
//        printf("arr[%d][%d] = ", i+1, j+1);
//        cin >> arr[i][j];
//    }
//}
//showMatrix(arr, 3,4);
