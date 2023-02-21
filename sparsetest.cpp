#include <fstream>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <Eigen/Dense>
using namespace std;
using namespace std::chrono;
using namespace Eigen;

void writeToFile(long time,uint16_t number){
    ofstream myfile;
    string anfang = "results/results";
    string n = to_string(number);
    string ende = ".txt";

    myfile.open (anfang + n + ende, std::ios_base::app);
    myfile << time << "\n";
    myfile.close();
}

Eigen::VectorXi readWeight(int col){
    Eigen::VectorXi inp(50000);

    string S = to_string(col);
    string beg = "numberfiles/weights/W";
    string ende = ".txt";

    string inputPath = beg + S + ende;

    std::ifstream infile(inputPath);
    int a;
    uint16_t k = 0;
    while (infile >> a)
    {
        inp(k) = a;
        k++;
    }
    return inp;
}

int main(){
    Eigen::MatrixXi weights(50000,256);
    weights.setZero();
    
    for (int w = 0;w<256;w++){
        weights(all,w) = readWeight(w);
    }

    Eigen::MatrixXi inp (1,50000);
    inp.setZero();
    std::ifstream infile("numberfiles/N60.txt");
    int a;
    uint16_t k = 0;
    while (infile >> a)
    {
        inp(0,k) = a;
        k++;
    }

    for (uint16_t o = 0;o<10000;o++){
        auto start = high_resolution_clock::now();
        
        Eigen::VectorXi out(256);
        out.setZero();
        for (uint16_t i = 0; i<50000;i++){
            if(inp.coeff(0,i) != 0){
                Eigen::VectorXi tw;
                tw = weights(i,all);
                out += tw;
            }
        } 

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(stop - start);
        writeToFile(duration.count(),60);
    }

    
    return 0;
}