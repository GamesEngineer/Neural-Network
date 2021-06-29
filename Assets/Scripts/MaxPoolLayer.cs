// Copyright 2021 Game-U Enterprises LLC
using System;
using UnityEngine;
using UnityEngine.Assertions;

public class MaxPoolLayer : INeuralLayer
{
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

    public MaxPoolLayer(INeuralLayer input, NeuralLayerConfig config)
    {
        Assert.IsTrue(config.activationType == Neuron.ActivationType.MaxPool);
        Assert.IsTrue(config.channelCount == input.Depth);
        Assert.IsTrue(config.kernelSize == config.stride);
        this.config = config;
        this.input = input;
        input.OutLayer = this;
        depth = input.Depth;
        height = config.CalculateOutputSize(input.Height);
        width = config.CalculateOutputSize(input.Width);
        activations = new float[depth, height, width];
        feedback = new float[depth, height, width];
        maxInputCoords = new Vector2Int[depth, height, width];
        ChannelMin = new float[depth];
        ChannelMax = new float[depth];
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
                    float o = GetMaxInputInKernelWindow(outZ, outY, outX);
                    activations[outZ, outY, outX] = o;
                    if (o < ChannelMin[outZ]) ChannelMin[outZ] = o;
                    if (o > ChannelMax[outZ]) ChannelMax[outZ] = o;
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
                    feedback[outZ, outY, outX] = OutLayer.CalculateWeightedFeedback(outZ, outY, outX);
                }
            }
        }

        InLayer.BackPropagate();
    }

    public void UpdateWeightsAndBiases(float learningRate)
    {
        // nothing to update, so just go the the next layer
        OutLayer.UpdateWeightsAndBiases(learningRate);
    }

    public float CalculateWeightedFeedback(int inZ, int inY, int inX)
    {
        // Only propagate feedback to the maximum input in the pooling window.
        // Other inputs in the pooling window will receive no feedback because
        // they did not contribute to the error.
        int outZ = inZ;
        int outY = inY / config.stride;
        int outX = inX / config.stride;
        if (outX >= width || outY >= height) return 0f;
        Vector2Int maxIn = maxInputCoords[outZ, outY, outX];
        return maxIn.x == inX && maxIn.y == inY ? feedback[outZ, outY, outX] : 0f;
    }

    private float GetMaxInputInKernelWindow(int outZ, int outY, int outX)
    {
        maxInputCoords[outZ, outY, outX] = Vector2Int.zero;
        float maxInput = float.NegativeInfinity;

        for (int kernelY = 0; kernelY < config.kernelSize; kernelY++)
        {
            int inY = outY * config.stride + kernelY;
            if (inY < 0 || inY >= input.Height) continue;

            for (int kernelX = 0; kernelX < config.kernelSize; kernelX++)
            {
                int inX = outX * config.stride + kernelX;
                if (inX < 0 || inX >= input.Width) continue;

                float activation = input.Outputs[outZ, inY, inX];
                if (activation > maxInput)
                {
                    maxInput = activation;
                    maxInputCoords[outZ, outY, outX] = new Vector2Int(inX, inY);
                }
            }
        }

        return maxInput;
    }

    private readonly NeuralLayerConfig config;
    private readonly INeuralLayer input;
    private readonly int depth; // shared by input and output
    private readonly int width;
    private readonly int height;
    private readonly float[,,] activations; // neuron outputs
    private readonly float[,,] feedback; // learning via back propagation
    private readonly Vector2Int[,,] maxInputCoords;
}
