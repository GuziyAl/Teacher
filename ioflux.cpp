#include "ioflux.h"

double crtNumber(char *num, int num_length)
{
        double number = 0;
        int point_pos;
        for (point_pos = 0;point_pos < num_length; ++point_pos) {
            if (num[point_pos] == '.') {
                break;
            }
        }

        // До запятой
        for (int i = 0;i < point_pos; ++i) {
            number += ((int)num[i] - '0') * pow(10,point_pos - i - 1);
        }

        // После запятой
        for (int i = point_pos + 1;i < num_length; ++i) {
            number += ((int)num[i] - '0') * pow(10 ,-1 * (i - 1));
        }

        return number;
}

void readSignalsAndOutFromFile (double *isignals, ifstream *fin)
{
    string buffer;

    for (int i = 0; i < 7; ++i) {
        *fin >> buffer;
        char tmp [4];
        for (int j = 0; j < 4; ++j) {
            tmp [j] = buffer [j];
        }

        isignals[i] = crtNumber(tmp, 4);

    }
}

void makeJournalNote (int epoch_num,
                      double **hidden_layer,
                      int hid_neurons,
                      int hid_weights,
                      double **outer_layer,
                      int out_neurons,
                      int out_weights,
                      double network_error) {

    string new_path = "note_of_epoch_#" + to_string(epoch_num);
    string path = "C:\\Learning\\" + new_path + ".txt";
    ofstream fout (path);

    fout << "Epoch #" << epoch_num << endl;
    fout << "Error: " << network_error << endl;

    for (int i = 0; i < hid_neurons; ++i) {
        fout << "Hidden neuron №" << i + 1 << endl;
        for (int j = 0; j < hid_weights; ++j) {
            fout << "weight ["<< i+1 << "] [" << j+1 << "] = " << hidden_layer[i][j] << endl;
        }
    }

    for (int i = 0; i < out_neurons; ++i) {
        fout << "Outer neuron №" << i + 1 << endl;
        for (int j = 0; j < out_weights; ++j) {
            fout << "weight ["<< i+1 << "] [" << j+1 << "] = " << outer_layer[i][j] << endl;
        }
    }

}








