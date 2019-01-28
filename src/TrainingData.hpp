#include<iostream>
#include<vector>
#include<cassert>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<cmath>

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

    // void setRange(int minimum, int maximum);
    // int convert(double number);

private:
    int min;
    int max;
    ifstream m_trainingDataFile;
};

// void TrainingData::setRange(int minimum, int maximum) // Deixa eu bagunçar o meu códigho kkkkk
// {
//     this->min = minimum;
//     this->max = maximum;
// }

// int TrainingData::convert(double number)
// {
//     double aux =   ((number+1)*100) / 2;
//     return (int)(max-min)*aux/100;
// }


void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) 
    {
        abort();
    }

    while (!ss.eof()) 
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
	cout<<"Abrindo arquivo |"<<filename<<"|"<<endl;
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) 
    {
        double oneValue;
        while (ss >> oneValue) 
        {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) 
    {
        double oneValue;
        while (ss >> oneValue) 
        {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}