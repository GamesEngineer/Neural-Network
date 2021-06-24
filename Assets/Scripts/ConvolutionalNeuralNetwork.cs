// Copyright 2021 Game-U Enterprises LLC
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class ConvolutionalNeuralNetwork : MonoBehaviour
{
    #region Network Configuration & Tuning Parameters

    [SerializeField]
    protected List<NeuralLayerConfig> configuration = new List<NeuralLayerConfig>();

    [SerializeField, Range(0.0001f, 0.1f)]
    protected float learningRate = 0.001f;

    #endregion

    public InputLayer InLayer { get; private set; }
    public OutputLayer OutLayer { get; private set; }
    public int LayerCount { get; private set; }
    public float Loss { get; private set; }

    public void Initialize(int width, int height, int depth, Neuron.ActivationType outputActivation)
    {
        InLayer = new InputLayer(width, height, depth);
        LayerCount++;
        INeuralLayer previousLayer = InLayer;
        foreach (var layerInfo in configuration)
        {
            previousLayer = layerInfo.CreateLayer(previousLayer);
            LayerCount++;
        }
        OutLayer = new OutputLayer(previousLayer, outputActivation);
        LayerCount++;

#if DEBUG
        for (INeuralLayer l = InLayer; l != null; l = l.OutLayer)
        {
            Debug.Log($"{l.GetType().Name}: {l.Width}x{l.Height}x{l.Depth}; {l.Activation}");
        }
#endif
    }

    public void ChangeConfiguration(List<NeuralLayerConfig> newConfiguration)
    {
        Assert.IsNull(InLayer, "Cannot change configuration after initialization.");
        configuration = newConfiguration;
    }

    public INeuralLayer GetLayer(int index)
    {
        INeuralLayer layer = InLayer;
        while (index-- > 0 && layer != null)
        {
            layer = layer.OutLayer;
        }
        return layer;
    }

    public void Think()
    {
        if (InLayer == null) return;
        InLayer.Activate();
    }

    public void Learn(float learningRateMultiplier = 1f)
    {
        if (InLayer == null || OutLayer == null) return;
        InLayer.Activate();
        Loss = OutLayer.CalculateLoss();
        OutLayer.BackPropagate();
        InLayer.UpdateWeightsAndBiases(learningRate * learningRateMultiplier);
    }


}
