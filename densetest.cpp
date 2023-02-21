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
    Eigen::MatrixXi inp (1,50000);
    Eigen::MatrixXi weights(50000,256);
    inp.setZero();
    weights.setZero();
    
    std::ifstream infile("numberfiles/N60.txt");
    int a;
    uint16_t k = 0;
    while (infile >> a)
    {
        inp(0,k) = a;
        k++;
    }

    for (int w = 0;w<256;w++){
        weights(all,w) = readWeight(w);
    }

    Eigen::MatrixXi out(1,256);
    out.setZero();
    
    out = inp * weights;
    
    for (uint16_t o = 0;o<1000;o++){
        auto start = high_resolution_clock::now();
        
        Eigen::MatrixXi out(1,256);
        out.setZero();
        
        out = inp * weights;

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(stop - start);
        writeToFile(duration.count(),62);
    }

    return 0;
}