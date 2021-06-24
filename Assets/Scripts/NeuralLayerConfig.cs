// Copyright 2021 Game-U Enterprises LLC
using System;
using UnityEngine;

[Serializable]
public struct NeuralLayerConfig
{
    public int channelCount;
    public Neuron.ActivationType activationType;
    [Range(0, 7)] public int kernelSize;
    [Range(0, 7)] public int stride;

    public int CalculateOutputSize(int numInputsAlongDimension)
    {
        int kernelExtents = Mathf.CeilToInt(kernelSize / 2f);
        return (numInputsAlongDimension - kernelSize + kernelExtents) / stride + 1;
    }

    public int GetInputIndex(int outIndex, int kernelValueIndex)
    {
        int kernelExtents = Mathf.CeilToInt(kernelSize / 2f);
        return outIndex * stride + kernelValueIndex - kernelExtents;
    }

    public INeuralLayer CreateLayer(INeuralLayer inLayer)
    {
        INeuralLayer layer;
        if (activationType == Neuron.ActivationType.MaxPool)
        {
            layer = new MaxPoolLayer(inLayer, this);
        }
        else if (kernelSize > 0)
        {
            layer = new ConvolutionLayer(inLayer, this);
        }
        else
        {
            layer = new FlatDenseLayer(inLayer, this);
        }
        return layer;
    }
}
