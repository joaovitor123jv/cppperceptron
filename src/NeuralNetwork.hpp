#include<iostream>
#include<cassert>
#include<vector>
#include<cstdio>
#include "Neuron.hpp"

using std::vector;
using std::cout;
using std::endl;

#ifndef NeuralNet_NEURON
	typedef vector<Neuron> Layer;
	#define NeuralNet_NEURON
#endif //NeuralNet_NEURON


class NeuralNetwork
{
public:
    NeuralNetwork (const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};




void NeuralNetwork::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) 
    {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void NeuralNetwork::backProp(const vector<double> &targetVals)
{
    // Calcula a média geral dos erros da rede (RMS das saídas dos neurônios errados)

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) 
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // obtem um negócio da saída
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // Calcula os gradientes da camada de saída

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) 
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calcula gradientes das camadas "ocultas" (hidden layer)

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) 
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) 
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    //Atualiza os pesos das conexões para todas as camadas que têm saída para a primeira "camada oculta"

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) 
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) 
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void NeuralNetwork::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Define os valores de entrada nos neuronios de entrada
    for (unsigned i = 0; i < inputVals.size(); ++i) 
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // propagação "forward"
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) 
    {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) 
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

NeuralNetwork::NeuralNetwork(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) 
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Adiciona neurônios à camada criada e um bias em cada camada.

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) 
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Fez um neurônio!!" << endl;
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0);
    }
}
