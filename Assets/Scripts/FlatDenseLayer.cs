// Copyright 2021 Game-U Enterprises LLC
using System;
using UnityEngine;
using UnityEngine.Assertions;

public class FlatDenseLayer : INeuralLayer
{
    public INeuralLayer InLayer => input;
    public INeuralLayer OutLayer { get; set; }
    public int Width => 1;
    public int Height => 1;
    public int Depth => depth;
    public float[,,] Outputs => activations;
    public float[,,] Feedback => feedback;
    public Neuron.ActivationType Activation => config.activationType;
    public float[] ChannelMin { get; private set; }
    public float[] ChannelMax { get; private set; }

    public FlatDenseLayer(INeuralLayer input, NeuralLayerConfig config)
    {
        Assert.IsTrue(config.kernelSize == 0, "Flat layer's kernel size must be zero.");
        this.config = config;
        this.input = input;
        input.OutLayer = this;
        depth = config.channelCount;
        signals = new float[depth];
        activations = new float[depth, 1, 1];
        weights = new float[depth, input.Depth, input.Height, input.Width];
        biases = new float[depth];
        feedback = new float[depth, 1, 1];
        ChannelMin = new float[depth];
        ChannelMax = new float[depth];
        activationFunc = Neuron.ActivationFunctions[(int)config.activationType];
        dActivationFunc = Neuron.ActivationDerivatives[(int)config.activationType];

        // Use He-style initialization of synaptic weights.
        // Initialize each channel's weights with a pseudo-normal distribution,
        // and set its bias to shift the result of activation toward zero.
        float range = 1f / Mathf.Sqrt(input.Width * input.Height * input.Depth / 2f);
        for (int outZ = 0; outZ < depth; outZ++)
        {
            float bias = 0f;
            for (int inZ = 0; inZ < input.Depth; inZ++)
            {
                for (int inY = 0; inY < input.Height; inY++)
                {
                    for (int inX = 0; inX < input.Width; inX++)
                    {
                        float r = UnityEngine.Random.Range(-range, +range)
                                + UnityEngine.Random.Range(-range, +range);
                        bias += r;
                        weights[outZ, inZ, inY, inX] = r;
                    }
                }
            }
            biases[outZ] = -bias;
        }
    }

    public void Activate()
    {
        Tensor.Fill(ChannelMin, float.PositiveInfinity);
        Tensor.Fill(ChannelMax, float.NegativeInfinity);

        for (int outZ = 0; outZ < depth; outZ++)
        {
            float weightedSum = biases[outZ];
            for (int inZ = 0; inZ < input.Depth; inZ++)
            {
                for (int inY = 0; inY < input.Height; inY++)
                {
                    for (int inX = 0; inX < input.Width; inX++)
                    {
                        weightedSum += InLayer.Outputs[inZ, inY, inX] * weights[outZ, inZ, inY, inX];
                    }
                }
            }
            signals[outZ] = weightedSum;
            float o = activationFunc(weightedSum);
            activations[outZ, 0, 0] = o;
            if (o < ChannelMin[outZ]) ChannelMin[outZ] = o;
            if (o > ChannelMax[outZ]) ChannelMax[outZ] = o;
        }

        OutLayer.Activate();
    }

    public void BackPropagate()
    {
        Assert.IsTrue(OutLayer.Width == 1);
        Assert.IsTrue(OutLayer.Height == 1);
        for (int z = 0; z < depth; z++)
        {
            float slope = dActivationFunc(signals[z]);
            if (slope == 0f)
            {
                feedback[z, 0, 0] = 0f;
                continue;
            }
            float weightedError = OutLayer.CalculateWeightedFeedback(z, 0, 0);
            feedback[z, 0, 0] = slope * weightedError;
        }

        InLayer.BackPropagate();
    }

    public void UpdateWeightsAndBiases(float learningRate)
    {
        for (int outZ = 0; outZ < depth; outZ++)
        {
            float change = learningRate * feedback[outZ, 0, 0];
            if (change == 0f) continue;

            biases[outZ] += change;
            for (int inZ = 0; inZ < input.Depth; inZ++)
            {
                for (int inY = 0; inY < input.Height; inY++)
                {
                    for (int inX = 0; inX < input.Width; inX++)
                    {
                        weights[outZ, inZ, inY, inX] += change * input.Outputs[inZ, inY, inX];
                    }
                }
            }
        }

        OutLayer.UpdateWeightsAndBiases(learningRate);
    }

    public float CalculateWeightedFeedback(int inZ, int inY, int inX)
    {
        float e = 0f;
        for (int outZ = 0; outZ < depth; outZ++)
        {
            e += feedback[outZ, 0, 0] * weights[outZ, inZ, inY, inX];
        }
        return e;
    }

    private readonly NeuralLayerConfig config;
    private readonly INeuralLayer input;
    private readonly int depth;
    private readonly float[] signals; // pre-activation value of neurons
    private readonly float[,,] activations; // activated neuron outputs
    private readonly float[,,,] weights; // weights [outZ, inZ, inY, inX]
    private readonly float[] biases; // entire channel (Z) uses the same bias value
    private readonly float[,,] feedback; // learning via back propagation
    private readonly Func<float, float> activationFunc; // neuron activation function
    private readonly Func<float, float> dActivationFunc; // derivative of activation fuction
}
