using System;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork : MonoBehaviour
{
    public class Layer
    {
        public readonly float[] neurons;
        public readonly float[,] weights; // [inputs, neurons]
        public readonly float[] biases;
        public Layer(int numNeurons, int numInputs)
        {
            neurons = new float[numNeurons];
            weights = new float[numInputs, numNeurons];
            // Initialize the neuron biases with random noise
            for (int i = 0; i < numNeurons; i++)
            {
                biases[i] = UnityEngine.Random.Range(-0.5f, 0.5f);
            }
            // Initialize the weights matrix with random noise
            for (int j = 0; j < numInputs; j++)
            {
                for (int i = 0; i < numNeurons; i++)
                {
                    weights[j,i] = UnityEngine.Random.Range(-0.5f, 0.5f);
                }
            }
        }

        public void Activate(float[] inputs, Func<float, float> activationFunc)
        {
            int numInputs = weights.GetLength(0);
            int numNeurons = neurons.Length;

            for (int i = 0; i < numNeurons; i++)
            {
                float weightedSum = 0f;
                for (int j = 0; j < numInputs; j++)
                {
                    weightedSum += inputs[j] * weights[j, i];
                }
                neurons[i] = activationFunc(weightedSum + biases[i]);
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
        int numInputs = sensoryInputs.Length;

        for (int i = 0; i < layersInfo.Count; i++)
        {
            var l = layersInfo[i];
            var layer = new Layer(l.neuronCount, numInputs);
            layers.Add(layer);

            // For the next layer
            numInputs = l.neuronCount;
        }
    }

    // Range: [-1..+1]
    private float Tanh(float x)
    {
        float e2x = Mathf.Exp(2f * x);
        return (e2x - 1f) / (e2x + 1f);
    }

    // Range: [0..1]
    private float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    private void Think(float[] inputs, Func<float, float> activationFunc)
    {
        foreach (var l in layers)
        {
            l.Activate(inputs, activationFunc);
            inputs = l.neurons;
        }
    }

    private void Train(float[] inputs, Func<float, float> activationFunc)
    {
        // TODO
    }
}
