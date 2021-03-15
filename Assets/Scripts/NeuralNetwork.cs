using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class NeuralNetwork : MonoBehaviour
{
    public class Layer
    {
        public readonly float[] inputs; // "owned" by previous layer
        public readonly float[] outputs; // neuron outputs
        public readonly float[,] weights; // matrix [outputs, inputs] of synaptic weights (one row of weights for each output neuron)
        public readonly float[] biases; // offset added to weighted sum of inputs
        public readonly float[] feedback; // learning via back propagation
        public Neuron.ActivationType activationType = Neuron.ActivationType.ReLU;

        public Layer(float[] inputs, int numOutputs)
        {
            int numInputs = inputs.Length;
            this.inputs = inputs;
            outputs = new float[numOutputs];
            weights = new float[numOutputs, numInputs];
            biases = new float[numOutputs];
            feedback = new float[numOutputs];

            // Initialize the each neuron's bias with random noise
            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                biases[outIndex] = UnityEngine.Random.Range(-0.5f, 0.5f);
            }

            // Initialize the matrix of synaptic weights with random noise
            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                for (int inIndex = 0; inIndex < numInputs; inIndex++)
                {
                    weights[outIndex, inIndex] = UnityEngine.Random.Range(-0.5f, 0.5f);
                }
            }
        }

        public void Activate()
        {
            int numInputs = inputs.Length;
            int numOutputs = outputs.Length;
            Assert.IsTrue(weights.GetLength(0) == numOutputs);
            Assert.IsTrue(weights.GetLength(1) == numInputs);
            var activationFunc = Neuron.ActivationFunctions[(int)activationType];

            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                float weightedSum = 0f;
                for (int inIndex = 0; inIndex < numInputs; inIndex++)
                {
                    weightedSum += inputs[inIndex] * weights[outIndex, inIndex];
                }
                outputs[outIndex] = activationFunc(weightedSum + biases[outIndex]);
            }
        }

        public void BackPropagate(float[] errors, float[,] nextLayerWeights)
        {
            int numInputs = inputs.Length;
            int numOutputs = outputs.Length;
            int numErrors = errors.Length;
            Assert.IsTrue(nextLayerWeights == null || nextLayerWeights.GetLength(0) == numErrors);
            Assert.IsTrue(nextLayerWeights == null || nextLayerWeights.GetLength(1) == numOutputs);
            Assert.IsTrue(weights.GetLength(0) == numOutputs);
            Assert.IsTrue(weights.GetLength(1) == numInputs);
            Assert.IsTrue(biases.Length == numOutputs);
            Assert.IsTrue(feedback.Length == numOutputs);
            var dActivationFunc = Neuron.ActivationDerivatives[(int)activationType];

            // Calculate feedback signals
            if (nextLayerWeights == null)
            {
                Assert.IsTrue(numErrors == numOutputs);
                for (int outIter = 0; outIter < numOutputs; outIter++)
                {
                    float slope = dActivationFunc(outputs[outIter]);
                    feedback[outIter] = slope * errors[outIter];
                }
            }
            else
            {
                for (int outIter = 0; outIter < numOutputs; outIter++)
                {
                    float slope = dActivationFunc(outputs[outIter]);
                    float weightedError = 0f;
                    for (int nextIter = 0; nextIter < numErrors; nextIter++)
                    {
                        weightedError += errors[nextIter] * nextLayerWeights[nextIter, outIter];
                    }
                    feedback[outIter] = slope * weightedError;
                }
            }
        }

        public void UpdateWeightsAndBiases(float learningRate)
        {
            int numInputs = inputs.Length;
            int numOutputs = outputs.Length;
            Assert.IsTrue(weights.GetLength(0) == numOutputs);
            Assert.IsTrue(weights.GetLength(1) == numInputs);
            Assert.IsTrue(biases.Length == numOutputs);
            Assert.IsTrue(feedback.Length == numOutputs);

            for (int outIter = 0; outIter < numOutputs; outIter++)
            {
                float change = learningRate * feedback[outIter];
                biases[outIter] += change;
                for (int inIter = 0; inIter < numInputs; inIter++)
                {
                    weights[outIter, inIter] += change * inputs[inIter];
                }
            }
        }
    }

    [Serializable]
    public struct LayerInfo
    {
        public int neuronCount;
        public Neuron.ActivationType activationType;
    }

    public List<LayerInfo> layersInfo = new List<LayerInfo>();
    public List<Layer> layers = new List<Layer>();
    public Layer InputLayer => layers[0];
    public Layer OutputLayer => layers[layers.Count - 1];
    public float[] SensoryInputs { get; private set; }
    public float[] Targets { get; private set; }
    public float[] Results { get; private set; }
    public float Loss { get; private set; }
    [Range(0.001f, 0.5f)] public float learningRate = 0.01f;
    public int numTrainingIterations = 1000;

    public void Initialize(int numInputs)
    {
        SensoryInputs = new float[numInputs];

        // Create the layers and connect them to each other
        layers = new List<Layer>(layersInfo.Count);
        float[] inputs = SensoryInputs;
        for (int i = 0; i < layersInfo.Count; i++)
        {
            var l = layersInfo[i];
            var layer = new Layer(inputs, l.neuronCount);
            layers.Add(layer);
            // For the next layer
            inputs = layer.outputs;
        }

        Results = OutputLayer.outputs;
        Targets = new float[Results.Length];
    }

    private static float CalculateLoss(float[] targets, float[] outputs, float[] errors)
    {
        Assert.IsTrue(targets.Length == outputs.Length);
        Assert.IsTrue(errors.Length == outputs.Length);
        float loss = 0f;
        for (int i = 0; i < outputs.Length; i++)
        {
            float error = targets[i] - outputs[i];
            errors[i] = error;
            loss += error * error * 0.5f;
        }
        return loss;
    }

    public void Think()
    {
        foreach (var l in layers)
        {
            l.Activate();
        }
    }

    public void Learn(float[] targets, float learningRate)
    {
        // Populate the network with feed-forward data
        Think();

        float[] errors = new float[OutputLayer.feedback.Length];
        Loss = CalculateLoss(targets, OutputLayer.outputs, errors);

        // Propagate errors backward through the network,
        // and update each layer's weights and biases with
        // gradient descent in order to reduce error in
        // future predictions.
        float[,] nextLayerWeights = null;
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            var layer = layers[i];
            layer.BackPropagate(errors, nextLayerWeights);
            layer.UpdateWeightsAndBiases(learningRate);
            errors = layer.feedback;
            nextLayerWeights = layer.weights;
        }
    }
}
