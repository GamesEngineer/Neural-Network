// Copyright 2021 Game-U Enterprises LLC
using System;
using UnityEngine;
using UnityEngine.Assertions;

public class OutputLayer : INeuralLayer
{
    public INeuralLayer InLayer => inLayer;
    public INeuralLayer OutLayer { get => null; set => throw new InvalidOperationException(); }
    public int Width => inLayer.Width;
    public int Height => inLayer.Height;
    public int Depth => inLayer.Depth;
    public float[,,] Outputs => outputs;
    public float[,,] Feedback => feedback;
    public Neuron.ActivationType Activation { get; }
    public float[] ChannelMin { get; private set; }
    public float[] ChannelMax { get; private set; }
    public float[,,] Targets => targets;
    public int TargetsOneHotIndex { get; private set; } = -1; // Only valid when using SoftMax activation
    public int OutputsWinnerIndex { get; private set; } = -1; // Only valid when using SoftMax activation

    public OutputLayer(INeuralLayer inLayer, Neuron.ActivationType activation)
    {
        this.inLayer = inLayer;
        inLayer.OutLayer = this;
        Activation = activation;
        feedback = new float[Depth, Width, Height];
        targets = new float[Depth, Width, Height];
        outputs = new float[Depth, Width, Height];
        ChannelMin = new float[Depth];
        ChannelMax = new float[Depth];
        shiftedExp = new double[Depth];
    }

    public void Activate(bool withDropout)
    {
        Tensor.Fill(ChannelMin, float.PositiveInfinity);
        Tensor.Fill(ChannelMax, float.NegativeInfinity);

        if (Activation == Neuron.ActivationType.SoftMax)
        {
            Assert.IsTrue(Height == 1);
            Assert.IsTrue(Width == 1);
            // Find largest value so we can shift everything towards zero (for numerical stability when using Exp)
            OutputsWinnerIndex = -1;
            float maximum = float.NegativeInfinity;
            for (int z = 0; z < Depth; z++)
            {
                if (inLayer.Outputs[z, 0, 0] > maximum)
                {
                    maximum = inLayer.Outputs[z, 0, 0];
                    OutputsWinnerIndex = z;
                }
            }
            double expSum = 0;
            for (int z = 0; z < Depth; z++)
            {
                double i = Math.Exp(inLayer.Outputs[z, 0, 0] - maximum);
                shiftedExp[z] = i;
                expSum += i;
            }
            Assert.IsTrue(expSum != 0.0);
            Assert.IsFalse(double.IsNaN(expSum));
            Assert.IsFalse(double.IsInfinity(expSum));
            double invExpSum = 1.0 / expSum;
            for (int z = 0; z < Depth; z++)
            {
                float o = (float)(shiftedExp[z] * invExpSum);
                outputs[z, 0, 0] = o;
                ChannelMin[z] = 0f;
                ChannelMax[z] = 1f;
            }
        }
        else
        {
            for (int z = 0; z < Depth; z++)
            {
                ChannelMin[z] = inLayer.ChannelMin[z];
                ChannelMax[z] = inLayer.ChannelMax[z];
                for (int y = 0; y < Height; y++)
                {
                    for (int x = 0; x < Width; x++)
                    {
                        outputs[z, y, x] = inLayer.Outputs[z, y, x];
                    }
                }
            }
        }

        /*stop*/
    }

    public void BackPropagate()
    {
        // feedback is already calculated when computing loss
        inLayer.BackPropagate();
    }

    public void UpdateWeightsAndBiases(float learningRate)
    {
        /*stop*/
    }

    public float CalculateWeightedFeedback(int inZ, int inY, int inX) => feedback[inZ, inY, inX];
    
    public float CalculateLoss()
    {
        float loss = 0f;
        if (Activation == Neuron.ActivationType.SoftMax)
        {
            Assert.IsTrue(Height == 1);
            Assert.IsTrue(Width == 1);
            for (int z = 0; z < Depth; z++)
            {
                float t = Targets[z, 0, 0];
                float o = Outputs[z, 0, 0];
                // Cross-entropy loss
                if (t == 1f)
                {
                    TargetsOneHotIndex = z;
                    loss -= Mathf.Log(Mathf.Max(o, 1.267e-14f));
                }
                // derivative of cross-entropy loss with respect to softmax inputs
                feedback[z, 0, 0] = (t - o) * Depth; // TODO - is this correct?
            }
        }
        else
        {
            for (int z = 0; z < Depth; z++)
            {
                for (int y = 0; y < Height; y++)
                {
                    for (int x = 0; x < Width; x++)
                    {
                        float error = Targets[z, y, x] - Outputs[z, y, x];
                        // Mean squared error loss
                        loss += error * error;
                        // derivative of squared error with respect to inputs
                        feedback[z, y, x] = 2f * error;
                    }
                }
            }
        }

        // Normalize
        loss /= (Width * Height * Depth);
        return (float)loss;
    }

    private readonly INeuralLayer inLayer;
    private readonly float[,,] outputs;
    private readonly float[,,] feedback;
    private readonly float[,,] targets;
    private readonly double[] shiftedExp;
}
