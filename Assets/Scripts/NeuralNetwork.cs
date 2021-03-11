using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class NeuralNetwork : MonoBehaviour
{
    public class Layer
    {
        public readonly float[] inputs;
        public readonly float[] outputs;
        public readonly float[,] weights; // [inputs, outputs]
        public readonly float[] biases;
        public readonly float[] feedback; // via back propagation

        public Layer(float[] inputs, int numOutputs)
        {
            int numInputs = inputs.Length;
            this.inputs = inputs;
            outputs = new float[numOutputs];
            weights = new float[numInputs, numOutputs];
            biases = new float[numOutputs];
            feedback = new float[numOutputs];

            // Initialize the neuron biases with random noise
            for (int i = 0; i < numOutputs; i++)
            {
                biases[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
            }

            // Initialize the weights matrix with random noise
            for (int j = 0; j < numInputs; j++)
            {
                for (int i = 0; i < numOutputs; i++)
                {
                    weights[j, i] = UnityEngine.Random.Range(-0.5f, 0.5f);
                }
            }
        }

        public void Activate(float[] inputs, Func<float, float> activationFunc)
        {
            int numInputs = inputs.Length;
            int numNeurons = outputs.Length;
            Assert.IsTrue(weights.GetLength(0) == numInputs);
            Assert.IsTrue(weights.GetLength(1) == numNeurons);

            for (int i = 0; i < numNeurons; i++)
            {
                float weightedSum = 0f;
                for (int j = 0; j < numInputs; j++)
                {
                    weightedSum += inputs[j] * weights[j, i];
                }
                outputs[i] = activationFunc(weightedSum + biases[i]);
            }
        }

        public void BackPropagate(float[] errors, Func<float, float> dActivationFunc, float learningRate)
        {
            int numNeurons = outputs.Length;
            Assert.IsTrue(errors.Length == numNeurons);
            int numInputs = weights.GetLength(0);
            Assert.IsTrue(weights.GetLength(1) == numNeurons);
            Assert.IsTrue(biases.Length == numNeurons);

            for (int i = 0; i < numNeurons; i++)
            {
                biases[i] -= feedback[i] * learningRate;
                for (int j = 0; j < numInputs; j++)
                {
                    weights[i, j] -= feedback[i] * inputs[j] * learningRate;
                }
            }
        }
    }

    public enum ActivationType
    {
        Tanh,
        Sigmoid,
        ReLU,
    }

    [Serializable]
    public struct LayerInfo
    {
        public int neuronCount;
        public ActivationType activationType;
    }

    public List<LayerInfo> layersInfo = new List<LayerInfo>();
    public List<Layer> layers = new List<Layer>();
    public float[] sensoryInputs;
    [Range(0.001f, 0.5f)] public float learningRate = 0.01f;
    public bool step;

    void Start()
    {
        Initialize();
    }

    void Update()
    {
        if (step)
        {
            Think(sensoryInputs, Tanh);
            step = false;
        }
    }

    private void Initialize()
    {
        layers = new List<Layer>(layersInfo.Count);

        if (sensoryInputs == null)
        {
            sensoryInputs = new float[layersInfo[0].neuronCount];
        }
        float[] inputs = sensoryInputs;

        for (int i = 0; i < layersInfo.Count; i++)
        {
            var l = layersInfo[i];
            var layer = new Layer(inputs, l.neuronCount);
            layers.Add(layer);

            // For the next layer
            inputs = layer.outputs;
        }
    }

    // Range: [-1..+1]
    private static float Tanh(float x)
    {
        float e2x = Mathf.Exp(2f * x);
        return (e2x - 1f) / (e2x + 1f);
    }

    private static float dTanh(float x)
    {
        float t = Tanh(x);
        return 1f - t * t;
    }

    // Range: [0..1]
    private static float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    private static float dSigmoid(float x)
    {
        float s = Sigmoid(x);
        return s * (1f - s);
    }

    private static float ReLU(float x)
    {
        return Mathf.Max(0f, x);
    }

    private static float dReLU(float x)
    {
        return x > 0f ? 1f : 0f;
    }

    private static float ErrorSqrdMag(float target, float actual)
    {
        float delta = target - actual;
        return delta * delta * 0.5f;
    }

    private static float TotalError(float[] target, float[] actual, out float[] errors)
    {
        Assert.IsTrue(target.Length == actual.Length);
        errors = new float[target.Length];
        float totalError = 0f;
        for (int i = 0; i < target.Length; i++)
        {
            float e = ErrorSqrdMag(target[i], actual[i]);
            errors[i] = e;
            totalError += e;
        }
        return totalError;
    }

    private void Think(float[] inputs, Func<float, float> activationFunc)
    {
        foreach (var l in layers)
        {
            l.Activate(inputs, activationFunc);
            inputs = l.outputs;
        }
    }

    private void Train(float[] inputs, float[] target,
        Func<float, float> activationFunc,
        Func<float, float> dActivationFunc,
        float learningRate)
    {
        Think(inputs, activationFunc);

        float[] outputs = layers[layers.Count - 1].outputs;
        float cost = TotalError(target, outputs, out float[] errors);
        Debug.Log($"Cost: {cost}");

        // Propagate errors backward through the network,
        // and update each layer's weights and biases with
        // gradient descent in order to reduce error in
        // future predictions.
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            var layer = layers[i];
            layer.BackPropagate(errors, dActivationFunc, learningRate);
        }
    }
}
