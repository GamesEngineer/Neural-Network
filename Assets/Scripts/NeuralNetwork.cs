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
        public readonly float[,] weights; // [inputs, outputs]
        public readonly float[] biases;
        public readonly float[] feedback; // learning via back propagation
        public Neuron.ActivationType activationType = Neuron.ActivationType.ReLU;

        public Layer(float[] inputs, int numOutputs)
        {
            int numInputs = inputs.Length;
            this.inputs = inputs;
            outputs = new float[numOutputs];
            weights = new float[numInputs, numOutputs];
            biases = new float[numOutputs];
            feedback = new float[numOutputs];

            // Initialize the each neuron's bias with random noise
            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                biases[outIndex] = UnityEngine.Random.Range(-0.5f, 0.5f);
            }

            // Initialize the matrix of synaptic weights with random noise
            for (int inIndex = 0; inIndex < numInputs; inIndex++)
            {
                for (int outIndex = 0; outIndex < numOutputs; outIndex++)
                {
                    weights[inIndex, outIndex] = UnityEngine.Random.Range(-0.5f, 0.5f);
                }
            }
        }

        public void Activate()
        {
            int numInputs = inputs.Length;
            int numOutputs = outputs.Length;
            Assert.IsTrue(weights.GetLength(0) == numInputs);
            Assert.IsTrue(weights.GetLength(1) == numOutputs);
            var activationFunc = Neuron.ActivationFunctions[(int)activationType];

            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                float weightedSum = 0f;
                for (int inIndex = 0; inIndex < numInputs; inIndex++)
                {
                    weightedSum += inputs[inIndex] * weights[inIndex, outIndex];
                }
                outputs[outIndex] = activationFunc(weightedSum + biases[outIndex]);
            }
        }

        public void BackPropagate(float[] errors, float learningRate)
        {
            int numOutputs = outputs.Length;
            Assert.IsTrue(feedback.Length == numOutputs);
            int numInputs = inputs.Length;
            Assert.IsTrue(weights.GetLength(0) == numInputs);
            Assert.IsTrue(weights.GetLength(1) == numOutputs);
            Assert.IsTrue(biases.Length == numOutputs);
            var dActivationFunc = Neuron.ActivationDerivatives[(int)activationType];

            // Calculate feedback signals
            for (int outIter = 0; outIter < numOutputs; outIter++)
            {
                float weightedError = 0f;
                for (int inIter = 0; inIter < numInputs; inIter++)
                {
                    weightedError += errors[outIter] * weights[inIter, outIter];
                }
                feedback[outIter] = dActivationFunc(outputs[outIter]) * weightedError;
            }

            // Update weights and biases
            for (int outIter = 0; outIter < numOutputs; outIter++)
            {
                float change = feedback[outIter] * learningRate; // TODO - should this be negated?
                biases[outIter] += change;
                for (int inIter = 0; inIter < numInputs; inIter++)
                {
                    weights[inIter, outIter] += inputs[inIter] * change;
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
    public float[] Results { get; private set; }
    public float[] Targets { get; private set; }
    [Range(0.001f, 0.5f)] public float learningRate = 0.01f;
    public bool train;
    public int numTrainingIterations = 1000;

    void Start()
    {
        Initialize();
    }

    void Update()
    {
        if (train)
        {
            for (int iter = 1; iter <= numTrainingIterations; iter++)
            {
                Train(Targets, learningRate);
            }
        }
        else
        {
            Think();
        }
    }

    private void Initialize()
    {
        SensoryInputs = new float[layersInfo[0].neuronCount];

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

    private void Think()
    {
        foreach (var l in layers)
        {
            l.Activate();
        }
    }

    private void Train(float[] targets, float learningRate)
    {
        Think();

        float cost = CalculateLoss(targets, OutputLayer.outputs, OutputLayer.feedback);
        Debug.Log($"Cost: {cost}");
        float[] errors = OutputLayer.feedback;

        // Propagate errors backward through the network,
        // and update each layer's weights and biases with
        // gradient descent in order to reduce error in
        // future predictions.
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            layers[i].BackPropagate(errors, learningRate);
            errors = layers[i].feedback;
        }
    }
}
