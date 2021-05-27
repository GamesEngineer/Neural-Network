// Copyright 2021 Game-U Enterprises LLC
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class ConvolutionalNeuralNetwork : MonoBehaviour
{
    public interface ILayer
    {
        ILayer InLayer { get; }
        ILayer OutLayer { get; set; }
        int Width { get; }
        int Height { get; }
        int Depth { get; }
        float[/*depth*/,/*height*/,/*width*/] Outputs { get; }
        float[/*depth*/,/*height*/,/*width*/] Feedback { get; }
        Neuron.ActivationType Activation { get; }
        void Activate();
        void BackPropagate();
        void UpdateWeightsAndBiases(float learningRate);
        float CalculateWeightedFeedback(int inZ, int inY, int inX);
    }

    #region Network Configuration & Tuning Parameters

    [Serializable]
    public struct LayerInfo
    {
        public int channelCount;
        public Neuron.ActivationType activationType;
        public int kernelSize;
        public int stride;
        //public bool padding;

        public int CalculateOutputSize(int numInputsAlongDimension)
        {
            //if (padding)
            {
                numInputsAlongDimension += 2;
            }
            return ((numInputsAlongDimension - kernelSize) / stride) + 1;
        }

        public ILayer CreateLayer(ILayer inLayer)
        {
            // TODO - Thursday Pro.Code
            ILayer layer;
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
                layer = new FlatLayer(inLayer, this);
            }
            return layer;
        }
    }

    [SerializeField]
    protected List<LayerInfo> configuration = new List<LayerInfo>();

    [SerializeField, Range(0.0001f, 0.1f)]
    protected float learningRate = 0.01f;

    #endregion

    public class FlatLayer : ILayer
    {
        public readonly LayerInfo config;

        private readonly ILayer input;
        private readonly int depth;
        private readonly float[,,] signals; // pre-activation value of neurons
        private readonly float[,,] activations; // activated neuron outputs
        private readonly float[,,,] weights; // weights [outZ, inZ, inY, inX]
        private readonly float[] biases; // entire channel (Z) uses the same bias value
        private readonly float[,,] feedback; // learning via back propagation
        private readonly Func<float, float> activationFunc; // neuron activation function
        private readonly Func<float, float> dActivationFunc; // derivative of activation fuction

        public ILayer InLayer => input;

        public ILayer OutLayer { get; set; }

        public int Width => 1;

        public int Height => 1;

        public int Depth => depth;

        public float[,,] Outputs => activations;

        public float[,,] Feedback => feedback;

        public Neuron.ActivationType Activation => config.activationType;

        public FlatLayer(ILayer input, LayerInfo config)
        {
            Assert.IsTrue(config.kernelSize == 0, "Flat layer's kernel size must be zero.");

            this.config = config;
            this.input = input;
            input.OutLayer = this;

            depth = config.channelCount;

            signals = new float[depth, 1, 1];
            activations = new float[depth, 1, 1];
            weights = new float[depth, input.Depth, input.Height, input.Width];
            biases = new float[depth];
            feedback = new float[depth, 1, 1];

            activationFunc = Neuron.ActivationFunctions[(int)config.activationType];
            dActivationFunc = Neuron.ActivationDerivatives[(int)config.activationType];

            // Initialize the synaptic weights with random noise
            for (int outZ = 0; outZ < depth; outZ++)
            {
                for (int inZ = 0; inZ < input.Depth; inZ++)
                {
                    for (int inY = 0; inY < InLayer.Height; inY++)
                    {
                        for (int inX = 0; inX < InLayer.Width; inX++)
                        {
                            weights[outZ, inZ, inY, inX] = UnityEngine.Random.Range(-0.5f, 0.5f);
                        }
                    }
                }
                biases[outZ] = UnityEngine.Random.Range(-0.1f, 0.1f);
            }
        }

        public void Activate()
        {
            for (int outZ = 0; outZ < depth; outZ++)
            {
                float weightedSum = 0f;
                for (int inZ = 0; inZ < InLayer.Depth; inZ++)
                {
                    for (int inY = 0; inY < InLayer.Height; inY++)
                    {
                        for (int inX = 0; inX < InLayer.Width; inX++)
                        {
                            weightedSum += InLayer.Outputs[inZ, inY, inX] * weights[outZ, inZ, inY, inX];
                        }
                    }
                }
                float preOutput = weightedSum + biases[outZ];
                signals[outZ, 0, 0] = preOutput;
                activations[outZ, 0, 0] = activationFunc(preOutput);
            }

            OutLayer.Activate();
        }

        public void BackPropagate()
        {
            for (int z = 0; z < depth; z++)
            {
                float slope = dActivationFunc(signals[z, 0, 0]);
                float weightedError = OutLayer.CalculateWeightedFeedback(z, 0, 0);
                feedback[z, 0, 0] += slope * weightedError;
            }

            InLayer.BackPropagate();
        }

        public void UpdateWeightsAndBiases(float learningRate)
        {
            for (int outZ = 0; outZ < depth; outZ++)
            {
                float change = learningRate * feedback[outZ, 0, 0];
                biases[outZ] += change;
                for (int inZ = 0; inZ < InLayer.Depth; inZ++)
                {
                    for (int inY = 0; inY < InLayer.Height; inY++)
                    {
                        for (int inX = 0; inX < InLayer.Width; inX++)
                        {
                            weights[outZ, inZ, inY, inX] += change * input.Outputs[inZ, inY, inX];
                        }
                    }
                }
                // Clear the feedback so we don't use it again (useful for batch learning)
                feedback[outZ, 0, 0] = 0f;
            }
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
    }

    public class ConvolutionLayer : ILayer
    {
        public readonly LayerInfo config;

        private readonly ILayer input;
        private readonly float[,,] signals; // pre-activation value of neurons
        private readonly float[,,] activations; // activated neuron outputs
        private readonly float[,,,] kernels; // convolution kernels [outZ, inZ, kernelY, kernelX]
        private readonly float[] biases; // entire channel (Z) uses the same bias value
        private readonly float[,,] feedback; // learning via back propagation
        private readonly Func<float, float> activationFunc; // neuron activation function
        private readonly Func<float, float> dActivationFunc; // derivative of activation fuction
        private readonly int width;
        private readonly int height;
        private readonly int depth;

        public ILayer InLayer => input;
        public ILayer OutLayer { get; set; }
        public int Width => width;
        public int Height => height;
        public int Depth => depth;
        public float[,,] Outputs => activations;
        public float[,,] Feedback => feedback;
        public float GetKernelValue(int inZ, int kernelIndex, int kernelX, int kernelY) => kernels[kernelIndex, inZ, kernelY, kernelX];

        // TODO - Start here for May, Thursday 27
        public Neuron.ActivationType Activation => config.activationType;

        public ConvolutionLayer(ILayer input, LayerInfo config)
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
            kernels = new float[depth, input.Depth, config.kernelSize, config.kernelSize];
            biases = new float[depth];
            feedback = new float[depth, height, width];

            activationFunc = Neuron.ActivationFunctions[(int)config.activationType];
            dActivationFunc = Neuron.ActivationDerivatives[(int)config.activationType];

            float invNumKernelValues = 1f / (config.kernelSize * config.kernelSize);

            // Initialize the kernel weights with random noise
            for (int outZ = 0; outZ < depth; outZ++)
            {
                for (int inZ = 0; inZ < input.Depth; inZ++)
                {
                    for (int kernelY = 0; kernelY < config.kernelSize; kernelY++)
                    {
                        for (int kernelX = 0; kernelX < config.kernelSize; kernelX++)
                        {
                            kernels[outZ, inZ, kernelY, kernelX] = UnityEngine.Random.Range(-0.5f, 0.5f) * invNumKernelValues;
                        }
                    }
                }
                biases[outZ] = UnityEngine.Random.Range(-0.01f, 0.01f);
            }
        }

        public void Activate()
        {
            for (int outZ = 0; outZ < depth; outZ++)
            {
                for (int outY = 0; outY < height; outY++)
                {
                    for (int outX = 0; outX < width; outX++)
                    {
                        float weightedSum = CrossCorrelation(outX, outY, outZ, kernels, input.Outputs, config.stride);
                        float neuronSignal = weightedSum + biases[outZ];
                        signals[outZ, outY, outX] = neuronSignal;
                        activations[outZ, outY, outX] = activationFunc(neuronSignal);
                    }
                }
            }

            OutLayer.Activate();
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
                        float weightedError = OutLayer.CalculateWeightedFeedback(outZ, outY, outX);
                        feedback[outZ, outY, outX] += slope * weightedError;
                    }
                }
            }

            InLayer.BackPropagate();
        }

        public void UpdateWeightsAndBiases(float learningRate)
        {
            float kernelExtents = config.kernelSize / 2f;

            for (int outZ = 0; outZ < depth; outZ++)
            {
                for (int outY = 0; outY < height; outY++)
                {
                    for (int outX = 0; outX < width; outX++)
                    {
                        float change = learningRate * feedback[outZ, outY, outX];
                        change /= config.kernelSize * config.kernelSize;
                        biases[outZ] += change;
                        for (int inZ = 0; inZ < input.Depth; inZ++)
                        {
                            for (int kernelY = 0; kernelY < config.kernelSize; kernelY++)
                            {
                                int inY = Mathf.CeilToInt(outY * config.stride + kernelY - kernelExtents);
                                if (inY < 0 || inY >= input.Height) continue;

                                for (int kernelX = 0; kernelX < config.kernelSize; kernelX++)
                                {
                                    int inX = Mathf.CeilToInt(outX * config.stride + kernelX - kernelExtents);
                                    if (inX < 0 || inX >= input.Width) continue;

                                    kernels[outZ, inZ, kernelY, kernelX] += change * input.Outputs[inZ, inY, inX];
                                }
                            }
                        }
                        // Clear the feedback so we don't use it again (useful for batch learning)
                        feedback[outZ, outY, outX] = 0f;
                    }
                }
            }
        }
        
        public float CalculateWeightedFeedback(int inZ, int inY, int inX) => Convolution(inX, inY, inZ, kernels, feedback, config.stride);
    }

    public class MaxPoolLayer : ILayer
    {
        public readonly LayerInfo config;
        private readonly ILayer input;
        private readonly float[,,] activations; // neuron outputs
        private readonly float[,,] feedback; // learning via back propagation
        private readonly Vector2Int[,,] maxInputCoords;

        private readonly int depth; // shared by input and output
        private readonly int width;
        private readonly int height;
        private readonly float kernelExtents;

        public ILayer InLayer => input;
        public ILayer OutLayer { get; set; }
        public int Width => width;
        public int Height => height;
        public int Depth => depth;
        public float[,,] Outputs => activations;
        public float[,,] Feedback => feedback;
        public Neuron.ActivationType Activation => config.activationType;

        public MaxPoolLayer(ILayer input, LayerInfo config)
        {
            Assert.IsTrue(config.activationType == Neuron.ActivationType.MaxPool);
            Assert.IsTrue(config.channelCount == input.Depth);
            Assert.IsTrue(config.kernelSize == config.stride);

            this.config = config;
            this.input = input;
            input.OutLayer = this;

            depth = input.Depth;
            height = input.Height / config.stride;// config.CalculateOutputSize(input.Height);
            width = input.Width / config.stride;// config.CalculateOutputSize(input.Width);
            kernelExtents = config.kernelSize / 2f;

            activations = new float[depth, height, width];
            feedback = new float[depth, height, width];
            maxInputCoords = new Vector2Int[depth, height, width];
        }

        public void Activate()
        {
            for (int outZ = 0; outZ < depth; outZ++)
            {
                for (int outY = 0; outY < height; outY++)
                {
                    for (int outX = 0; outX < width; outX++)
                    {
                        activations[outZ, outY, outX] = GetMaxInputInKernelWindow(kernelExtents, outZ, outY, outX);
                    }
                }
            }

            OutLayer.Activate();
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
            // nothing to update
        }

        private float GetMaxInputInKernelWindow(float kernelExtents, int z, int outY, int outX)
        {
            // Initialize with invalid values
            maxInputCoords[z, outY, outX] = new Vector2Int(-1, -1);
            // Ensure that at least one input will overwrite these invalid values
            float maxInput = float.NegativeInfinity;

            for (int kernelY = 0; kernelY < config.kernelSize; kernelY++)
            {
                //int inY = Mathf.CeilToInt(outY * config.stride + kernelY - kernelExtents);
                int inY = outY * config.stride + kernelY;
                if (inY < 0 || inY >= input.Height) continue;

                for (int kernelX = 0; kernelX < config.kernelSize; kernelX++)
                {
                    //int inX = Mathf.CeilToInt(outX * config.stride + kernelX - kernelExtents);
                    int inX = outX * config.stride + kernelX;
                    if (inX < 0 || inX >= input.Width) continue;

                    float activation = input.Outputs[z, inY, inX];
                    if (activation > maxInput)
                    {
                        maxInput = activation;
                        maxInputCoords[z, outY, outX] = new Vector2Int(inX, inY);
                    }
                }
            }

            return maxInput;
        }

        public float CalculateWeightedFeedback(int inZ, int inY, int inX)
        {
            // Only propagate feedback to the maximum input in the pooling window.
            // Other inputs in the pooling window will receive no feedback because
            // they did not contribute to the error.
            int outZ = inZ;
            int outY = inY / config.stride;
            int outX = inX / config.stride;
            if (outX >= width || outY >= height) return 0f; // HACK - FIXME!
            Vector2Int maxIn = maxInputCoords[outZ, outY, outX];
            return maxIn.x == inX && maxIn.y == inY ? feedback[outZ, outY, outX] : 0f;
        }
    }

    public class InputLayer : ILayer
    {
        public ILayer InLayer => null;
        public ILayer OutLayer { get; set; }
        public int Width => signals.GetLength(2);
        public int Height => signals.GetLength(1);
        public int Depth => signals.GetLength(0);
        public float[,,] Outputs => signals;
        public float[,,] Feedback => null;
        public Neuron.ActivationType Activation => Neuron.ActivationType.None;
        public void Activate() => OutLayer.Activate();
        public void BackPropagate() { }
        public void UpdateWeightsAndBiases(float learningRate) => OutLayer.UpdateWeightsAndBiases(learningRate);
        public float CalculateWeightedFeedback(int inZ, int inY, int inX) => throw new NotImplementedException();
        public InputLayer(int width, int height, int depth)
        {
            signals = new float[depth, width, height];
        }
        public readonly float[,,] signals;
    }

    public class OutputLayer : ILayer
    {
        public ILayer InLayer => inLayer;
        public ILayer OutLayer { get => null; set => throw new InvalidOperationException(); }
        public int Width => inLayer.Width;
        public int Height => inLayer.Height;
        public int Depth => inLayer.Depth;
        public float[,,] Outputs => outputs;
        public float[,,] Feedback => errors;
        public float[,,] Targets => targets;
        public Neuron.ActivationType Activation { get; }
        public void Activate()
        {
            float expSum;
            if (Activation == Neuron.ActivationType.SoftMax)
            {
                expSum = 0f;
                for (int z = 0; z < Depth; z++)
                {
                    for (int y = 0; y < Height; y++)
                    {
                        for (int x = 0; x < Width; x++)
                        {
                            expSum += Mathf.Exp(inLayer.Outputs[z, y, x]);
                        }
                    }
                }
                for (int z = 0; z < Depth; z++)
                {
                    for (int y = 0; y < Height; y++)
                    {
                        for (int x = 0; x < Width; x++)
                        {
                            outputs[z, y, x] = Mathf.Exp(inLayer.Outputs[z, y, x]) / expSum;
                        }
                    }
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
                            outputs[z, y, x] = inLayer.Outputs[z, y, x];
                        }
                    }
                }
            }
        }

        public void BackPropagate() => inLayer.BackPropagate();
        
        public void UpdateWeightsAndBiases(float learningRate) { }

        public OutputLayer(ILayer inLayer, Neuron.ActivationType activation)
        {
            this.inLayer = inLayer;
            this.Activation = activation;
            inLayer.OutLayer = this;
            this.errors = new float[Depth, Width, Height];
            this.targets = new float[Depth, Width, Height];
            this.outputs = new float[Depth, Width, Height];
        }

        public float CalculateLoss()
        {
            float loss = 0f;
            if (Activation == Neuron.ActivationType.SoftMax)
            {
                for (int z = 0; z < Depth; z++)
                {
                    for (int y = 0; y < Height; y++)
                    {
                        for (int x = 0; x < Width; x++)
                        {
                            // Cross-entropy loss
                            loss -= Targets[z, y, x] * Mathf.Log(Outputs[z, y, x]);
                            errors[z, y, x] = -Targets[z, y, x] / Outputs[z, y, x];
                        }
                    }
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
                            // loss is the mean squared error
                            loss += error * error;
                            errors[z, y, x] = 2f * error; // derivative of (error)^2 is 2*error
                        }
                    }
                }

                loss /= (Width * Height * Depth);
            }

            return loss;
        }

        public float CalculateWeightedFeedback(int inZ, int inY, int inX) => errors[inZ, inY, inX];

        private readonly ILayer inLayer;
        private readonly float[,,] outputs;
        private readonly float[,,] errors;
        private readonly float[,,] targets;
    }

    public InputLayer InLayer { get; private set; }

    public OutputLayer OutLayer { get; private set; }

    public int LayerCount { get; private set; }
    public ILayer GetLayer(int index) 
    {
        ILayer layer = InLayer;
        while (index-- > 0 && layer != null) 
        {
            layer = layer.OutLayer;
        }
        return layer;
    }

    public float Loss { get; private set; }

    public void Initialize(int width, int height, int depth, Neuron.ActivationType outputActivation)
    {
        InLayer = new InputLayer(width, height, depth);
        LayerCount++;
        ILayer previousLayer = InLayer;
        foreach (var layerInfo in configuration)
        {
            previousLayer = layerInfo.CreateLayer(previousLayer);
            LayerCount++;
        }
        OutLayer = new OutputLayer(previousLayer, outputActivation);
        LayerCount++;

#if DEBUG
        for (ILayer l = InLayer; l != null; l = l.OutLayer)
        {
            Debug.Log($"{l.GetType().Name}: {l.Depth}x{l.Height}x{l.Width}; {l.Activation}");
        }
#endif
    }

    public void Think()
    {
        if (InLayer == null) return;
        InLayer.Activate();
    }

    public void Learn(float learningRateMultiplier = 1f)
    {
        if (InLayer == null || OutLayer == null) return;
        Think();
        Loss = OutLayer.CalculateLoss();
        OutLayer.BackPropagate();
        InLayer.UpdateWeightsAndBiases(learningRate * learningRateMultiplier);
    }

    // The following implementations of cross-correlation and convolution operators are informed and inspired by:
    //     1) https://en.wikipedia.org/wiki/Cross-correlation
    //     2) https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
    //     3) https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

    /// <summary>
    /// Computes the cross-correlation (sliding inner-product) of kernels[kernelIndex, ...] * tensor[...].
    /// </summary>
    /// <param name="column">Index of column in the OUTPUT tensor</param>
    /// <param name="row">Index of row in the OUTPUT tensor</param>
    /// <param name="kernelIndex">Index of the kernel filter to use</param>
    /// <param name="kernels">Array of kernel filters [kernelIndex, tensor depth, kernel size, kernel size]</param>
    /// <param name="tensor">Input tensor [depth, height, width] </param>
    /// <returns>The cross-correlation of tensor[...] * kernels[kernelIndex, ...]</returns>
    public static float CrossCorrelation(int column, int row, int kernelIndex, float[,,,] kernels, float[,,] tensor, int stride = 1)
    {
        Assert.IsTrue(kernels.GetLength(1) == tensor.GetLength(0));
        Assert.IsTrue(kernels.GetLength(2) == kernels.GetLength(3));

        int depth = tensor.GetLength(0);
        int height = tensor.GetLength(1);
        int width = tensor.GetLength(2);

        int kernelSize = kernels.GetLength(2);
        float kernelExtents = kernelSize / 2f;

        float weightedSum = 0f;

        for (int z = 0; z < depth; z++)
        {
            for (int kernelY = 0; kernelY < kernelSize; kernelY++)
            {
                int y = Mathf.CeilToInt(row * stride + kernelY - kernelExtents);
                if (y < 0 || y >= height) continue;

                for (int kernelX = 0; kernelX < kernelSize; kernelX++)
                {
                    int x = Mathf.CeilToInt(column * stride + kernelX - kernelExtents);
                    if (x < 0 || x >= width) continue;

                    float i = tensor[z, y, x];
                    float w = kernels[kernelIndex, z, kernelY, kernelX];
                    weightedSum += i * w;
                }
            }
        }

        return weightedSum;
    }

    /// <summary>
    /// Computes the convolution (sliding inner-product) of kernels[kernelIndex,...] * tensor[...].
    /// </summary>
    /// <param name="column">Index of column in the OUTPUT tensor</param>
    /// <param name="row">Index of row in the OUTPUT tensor</param>
    /// <param name="kernelIndex">Index of the kernel filter to use</param>
    /// <param name="kernels">Array of kernel filters [kernelIndex, tensor depth, kernel size, kernel size]</param>
    /// <param name="tensor">Input tensor [depth, height, width] </param>
    /// <returns>The convolution of tensor[...] * kernels[kernelIndex,...]</returns>
    public static float Convolution(int column, int row, int kernelIndex, float[,,,] kernels, float[,,] tensor, int stride = 1)
    {
        Assert.IsTrue(kernels.GetLength(0) == tensor.GetLength(0));
        Assert.IsTrue(kernels.GetLength(2) == kernels.GetLength(3));

        int depth = tensor.GetLength(0);
        int height = tensor.GetLength(1);
        int width = tensor.GetLength(2);

        int kernelSize = kernels.GetLength(2);
        float kernelExtents = kernelSize / 2f;

        float weightedSum = 0f;

        for (int z = 0; z < depth; z++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                int y = Mathf.CeilToInt(row * stride + n - kernelExtents);
                if (y < 0 || y >= height) continue;

                // Convolution is the same as a cross-correlation with a kernel that is rotated 180°
                int kernelY = kernelSize - n - 1;

                for (int m = 0; m < kernelSize; m++)
                {
                    int x = Mathf.CeilToInt(column * stride + m - kernelExtents);
                    if (x < 0 || x >= width) continue;
                    
                    int kernelX = kernelSize - m - 1;

                    float o = tensor[z, y, x];
                    float w = kernels[z, kernelIndex, kernelY, kernelX];
                    weightedSum += o * w;
                }
            }
        }

        return weightedSum;
    }

    public void ChangeConfiguration(List<LayerInfo> newConfig)
    {
        Assert.IsNull(InLayer, "Cannot change configuration after initialization");
        configuration = newConfig;
    }
}
