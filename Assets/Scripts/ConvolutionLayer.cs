// Copyright 2021 Game-U Enterprises LLC
using System;
using UnityEngine;
using UnityEngine.Assertions;

public class ConvolutionLayer : INeuralLayer
{
    public readonly NeuralLayerConfig config;
    public INeuralLayer InLayer => input;
    public INeuralLayer OutLayer { get; set; }
    public int Width => width;
    public int Height => height;
    public int Depth => depth;
    public float[,,] Outputs => activations;
    public float[,,] Feedback => feedback;
    public Neuron.ActivationType Activation => config.activationType;
    public float[] ChannelMin { get; private set; }
    public float[] ChannelMax { get; private set; }

    public ConvolutionLayer(INeuralLayer input, NeuralLayerConfig config)
    {
        Assert.IsTrue(config.kernelSize % 2 == 1, "Convolution layer's kernel size must be an odd number.");
        this.config = config;
        this.input = input;
        input.OutLayer = this;
        depth = config.channelCount;
        height = config.CalculateOutputSize(input.Height);
        width = config.CalculateOutputSize(input.Width);
        signals = new float[depth, height, width];
        activations = new float[depth, height, width];
        kernels = new float[depth, config.kernelSize, config.kernelSize];
        biases = new float[depth];
        feedback = new float[depth, height, width];
        ChannelMin = new float[depth];
        ChannelMax = new float[depth];
        activationFunc = Neuron.ActivationFunctions[(int)config.activationType];
        dActivationFunc = Neuron.ActivationDerivatives[(int)config.activationType];

        // Use He-style initialization of synaptic weights.
        // Initialize each channel's weights with a pseudo-normal distribution,
        // and set its bias to shift the result of activation toward zero.
        float range = 1f / Mathf.Sqrt(config.kernelSize * config.kernelSize * input.Depth / 2f);
        for (int outZ = 0; outZ < depth; outZ++)
        {
            float bias = 0f;
            for (int kernelY = 0; kernelY < config.kernelSize; kernelY++)
            {
                for (int kernelX = 0; kernelX < config.kernelSize; kernelX++)
                {
                    // Create a pseudo-normal distribution
                    float r = UnityEngine.Random.Range(-range, +range)
                            + UnityEngine.Random.Range(-range, +range);
                    bias += r;
                    kernels[outZ, kernelY, kernelX] = r;
                }
            }
            biases[outZ] = -bias;
        }
    }

    public void Activate(bool withDropout)
    {
        Tensor.Fill(ChannelMin, float.PositiveInfinity);
        Tensor.Fill(ChannelMax, float.NegativeInfinity);

        for (int outZ = 0; outZ < depth; outZ++)
        {
            for (int outY = 0; outY < height; outY++)
            {
                for (int outX = 0; outX < width; outX++)
                {
                    float activation;
                    if (withDropout && UnityEngine.Random.value < config.dropout)
                    {
                        signals[outZ, outY, outX] = float.NegativeInfinity;
                        activation = 0f;
                    }
                    else
                    {
                        float weightedSum = biases[outZ];
                        weightedSum += Tensor.CrossCorrelation(outX, outY, outZ, kernels, input.Outputs, config.stride);
                        signals[outZ, outY, outX] = weightedSum;
                        activation = activationFunc(weightedSum);
                    }
                    activations[outZ, outY, outX] = activation;
                    if (activation < ChannelMin[outZ]) ChannelMin[outZ] = activation;
                    if (activation > ChannelMax[outZ]) ChannelMax[outZ] = activation;
                }
            }
        }

        OutLayer.Activate(withDropout);
    }

    public void BackPropagate()
    {
        for (int outZ = 0; outZ < depth; outZ++)
        {
            for (int outY = 0; outY < height; outY++)
            {
                for (int outX = 0; outX < width; outX++)
                {
                    float slope = dActivationFunc(signals[outZ, outY, outX]);
                    if (slope == 0f)
                    {
                        feedback[outZ, outY, outX] = 0f;
                        continue;
                    }
                    float weightedError = OutLayer.CalculateWeightedFeedback(outZ, outY, outX);
                    feedback[outZ, outY, outX] = slope * weightedError;
                }
            }
        }

        InLayer.BackPropagate();
    }

    public void UpdateWeightsAndBiases(float learningRate)
    {
        for (int outZ = 0; outZ < depth; outZ++)
        {
            for (int outY = 0; outY < height; outY++)
            {
                for (int outX = 0; outX < width; outX++)
                {
                    float change = learningRate * feedback[outZ, outY, outX];
                    if (change == 0f) continue;

                    biases[outZ] += change;

                    for (int inZ = 0; inZ < input.Depth; inZ++)
                    {
                        for (int kernelY = 0; kernelY < config.kernelSize; kernelY++)
                        {
                            int inY = config.GetInputIndex(outY, kernelY);
                            if (inY < 0 || inY >= input.Height) continue;

                            for (int kernelX = 0; kernelX < config.kernelSize; kernelX++)
                            {
                                int inX = config.GetInputIndex(outX, kernelX);
                                if (inX < 0 || inX >= input.Width) continue;

                                kernels[outZ, kernelY, kernelX] += change * input.Outputs[inZ, inY, inX];
                            }
                        }
                    }
                }
            }
        }

        OutLayer.UpdateWeightsAndBiases(learningRate);
    }

    public float CalculateWeightedFeedback(int z, int y, int x) => Tensor.Convolution(x, y, z, kernels, feedback, config.stride);

    public float GetKernelValue(int channelIndex, int kernelX, int kernelY) => kernels[channelIndex, kernelY, kernelX];

    public float GetBias(int channelIndex) => biases[channelIndex];

    private readonly INeuralLayer input;
    private readonly int width;
    private readonly int height;
    private readonly int depth;
    private readonly float[,,] signals; // pre-activation value of neurons
    private readonly float[,,] activations; // activated neuron outputs
    private readonly float[,,] kernels; // convolution kernels [depth, kernelY, kernelX]
    private readonly float[] biases; // entire channel (Z) uses the same bias value
    private readonly float[,,] feedback; // learning via back propagation
    private readonly Func<float, float> activationFunc; // neuron activation function
    private readonly Func<float, float> dActivationFunc; // derivative of activation fuction
}
