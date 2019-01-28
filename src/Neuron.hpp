#include<vector>
#include<cmath>
#include<cstdlib>
#include "GeneralLibrary.h"

using namespace std;

class Neuron;

#ifndef NeuralNet_NEURON
	#define NeuralNet_NEURON
	typedef vector<Neuron> Layer;
#endif


class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;   // taxa de aprendizado da rede, de 0.0 a 1.0 (recomendado 0.15) (0.0 == não aprende nada) (1.0 = aprende demais)
    static double alpha; // multiplicador do peso anterior (ver geração do eta... essa variável adiciona uma fração do peso anterior ao novo peso)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return (rand() / double(RAND_MAX)); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};


void Neuron::updateInputWeights(Layer &prevLayer)
{
    // Os pesos que serão atualizados estão na struct "Connection", nos neurônios da camada anterior "prevLayer"

    for (unsigned n = 0; n < prevLayer.size(); ++n) 
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =                 // Entrada individual, calculada com relação à taxa de aprendizado
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Também adiciona um "momentum" = Uma pração do peso anterior;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    // Soma as "contibuições" dos erros aos nós (backprop)

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) 
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    //Tangente Hiperbólica, com taxa de saída : [-1.0 ; 1.0] (ela inclui o -1 e 1 nesse caso... mesmo a teoria não dizendo isso...)

    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // derivada de tangente hiperbólica
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Soma as saídas da camada anterior
    // Essa função inclui o viés "bias"

    for (unsigned n = 0; n < prevLayer.size(); ++n) 
    {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) 
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

