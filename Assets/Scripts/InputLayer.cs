// Copyright 2021 Game-U Enterprises LLC
using System;
using UnityEngine;
using UnityEngine.Assertions;

public class InputLayer : INeuralLayer
{
    public INeuralLayer InLayer => null;
    public INeuralLayer OutLayer { get; set; }
    public int Width => signals.GetLength(2);
    public int Height => signals.GetLength(1);
    public int Depth => signals.GetLength(0);
    public float[,,] Outputs => signals;
    public float[,,] Feedback => null;
    public Neuron.ActivationType Activation => Neuron.ActivationType.None;
    public float[] ChannelMin { get; private set; }
    public float[] ChannelMax { get; private set; }

    public InputLayer(int width, int height, int depth)
    {
        signals = new float[depth, width, height];
        ChannelMin = new float[depth];
        ChannelMax = new float[depth];
    }

    public void Activate(bool withDropout = false)
    {
        Tensor.Fill(ChannelMin, float.PositiveInfinity);
        Tensor.Fill(ChannelMax, float.NegativeInfinity);

        for (int z = 0; z < Depth; z++)
        {
            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    float o = signals[z, y, x];
                    if (o < ChannelMin[z]) ChannelMin[z] = o;
                    if (o > ChannelMax[z]) ChannelMax[z] = o;
                }
            }
        }

        OutLayer.Activate(withDropout);
    }

    public void BackPropagate() { /*stop*/ }

    public void UpdateWeightsAndBiases(float learningRate) => OutLayer.UpdateWeightsAndBiases(learningRate);

    public float CalculateWeightedFeedback(int inZ, int inY, int inX) => throw new NotImplementedException();

    public readonly float[,,] signals;
}
