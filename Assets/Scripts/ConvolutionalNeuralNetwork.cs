// Copyright 2021 Game-U Enterprises LLC
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.Assertions;
using System.IO;

public class ConvolutionalNeuralNetwork : MonoBehaviour, ISerializationCallbackReceiver
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
        InLayer.Activate(withDropout: true);
        Loss = OutLayer.CalculateLoss();
        OutLayer.BackPropagate();
        InLayer.UpdateWeightsAndBiases(learningRate * learningRateMultiplier);
    }

    #region Work In Progress

    const string checkpointFilename = "cnn-checkpoint.txt";

    [MenuItem("Tools/Save CNN Checkpoint")]
    public static void SaveCheckpoint()
    {
        var cnn = FindObjectOfType<ConvolutionalNeuralNetwork>();
        if (cnn == null)
        {
            Debug.LogWarning($"Save Failed! Cannot find a {nameof(ConvolutionalNeuralNetwork)} in the scene");
            return;
        }

        string fullPath = Application.persistentDataPath + checkpointFilename;
        print($"Saving checkpoint to {fullPath}...");
        using (StreamWriter writer = new StreamWriter(fullPath))
        {
            string json = JsonUtility.ToJson(cnn, prettyPrint: true);
            writer.WriteLine(json);
        }
        print("Checkpoint saved");
    }

    [MenuItem("Tools/Load CNN Checkpoint")]
    public static void LoadCheckpoint()
    {
        var cnn = FindObjectOfType<ConvolutionalNeuralNetwork>();
        if (cnn == null)
        {
            Debug.LogWarning($"Save Failed! Cannot find a {nameof(ConvolutionalNeuralNetwork)} in the scene");
            return;
        }
        
        string fullPath = Application.persistentDataPath + checkpointFilename;
        print($"Loading checkpoint from {fullPath}...");

        using (StreamReader reader = new StreamReader(fullPath))
        {
            string json = reader.ReadToEnd();
            JsonUtility.FromJsonOverwrite(json, cnn);
        }

        print("Checkpoint loaded");
    }

    [Serializable]
    public class LayerData
    {
        public string name;
        public int channelCount;
        public int zSize;
        public int ySize;
        public int xSize;
        public float[] weights; // flattened [channel, inZ, inY, inX]
        public float[] biases; // [channel]
        public LayerData(string name, int channelCount, int zSize, int ySize, int xSize)
        {
            this.name = name;
            this.channelCount = channelCount;
            this.zSize = zSize;
            this.ySize = ySize;
            this.xSize = xSize;
            weights = new float[channelCount * zSize * ySize * xSize];
            biases = new float[channelCount];
        }

        public int GetWeightIndex(int c, int z, int y, int x)
        {
            return x
                + y * xSize
                + z * xSize * ySize
                + c * xSize * ySize * zSize;
        }
    }

    [HideInInspector]
    public List<LayerData> layers = new List<LayerData>();

    public void OnBeforeSerialize()
    {
        layers.Clear();
        if (InLayer == null || OutLayer == null) return;

        for (INeuralLayer layer = InLayer; layer != null; layer = layer.OutLayer)
        {
            var convLayer = layer as ConvolutionLayer;
            if (convLayer != null)
            {
                int inDepth = layer.InLayer.Depth;
                var layerData = new LayerData(layer.GetType().Name, layer.Depth, inDepth, convLayer.KernelSize, convLayer.KernelSize);
                layers.Add(layerData);

                for (int c = 0; c < layer.Depth; c++)
                {
                    layerData.biases[c] = layer.GetBias(c);

                    for (int z = 0; z < inDepth; z++)
                    {
                        for (int y = 0; y < convLayer.KernelSize; y++)
                        {
                            for (int x = 0; x < convLayer.KernelSize; x++)
                            {
                                float w = convLayer.GetKernelValue(c, x, y, z);
                                int n = layerData.GetWeightIndex(c, z, y, x);
                                layerData.weights[n] = w;
                            }
                        }
                    }
                }
            }
            else if (layer is FlatDenseLayer)
            {
                INeuralLayer i = layer.InLayer;
                var layerData = new LayerData(layer.GetType().Name, layer.Depth, i.Depth, i.Height, i.Width);
                layers.Add(layerData);

                for (int c = 0; c < layer.Depth; c++)
                {
                    layerData.biases[c] = layer.GetBias(c);

                    for (int z = 0; z < i.Depth; z++)
                    {
                        for (int y = 0; y < i.Height; y++)
                        {
                            for (int x = 0; x < i.Width; x++)
                            {
                                int n = layerData.GetWeightIndex(c, z, y, x);
                                float w = layer.GetWeight(c, z, y, x);
                                layerData.weights[n] = w;
                            }
                        }
                    }
                }
            }
            else
            {
                var layerData = new LayerData(layer.GetType().Name, layer.Depth, 0, 0, 0);
                layers.Add(layerData);
            }
        }
    }

    public void OnAfterDeserialize()
    {
        // TODO
    }

    #endregion

}
