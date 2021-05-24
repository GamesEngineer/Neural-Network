// Copyright 2021 Gameu Enterprises LLC
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
        void Activate();
        void BackPropagate();
        void UpdateWeightsAndBiases(float learningRate);
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
            else
            {
                layer = new ConvolutionLayer(inLayer, this);
            }
            return layer;
        }
    }

    [SerializeField]
    protected List<LayerInfo> configuration = new List<LayerInfo>();

    [SerializeField, Range(0.0001f, 0.1f)]
    protected float learningRate = 0.01f;

    #endregion

    public class ConvolutionLayer : ILayer
    {
        public readonly LayerInfo config;

        private readonly ILayer input;
        private readonly float[,,] signals; // pre-activation value of neurons
        private readonly float[,,] activations; // activated neuron outputs
        private readonly float[,,,] kernels; // convolution kernels [outZ, inZ, kernelY, kernelX] (x,y) are constrained to kernel size
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

        // TODO - Start here for May, Thursday 27
        public float FeedbackConvolution(int x, int y, int z) => Convolution(x, y, z, kernels, feedback, config.stride);

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
            }
        }

        // TODO - Thursday 5/20 Pro.Code - start here
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
#if false // FIXME
                        float weightedError = Convolution(outX, outY, outZ, kernels, feedback, config.stride);
                        feedback[outZ, outY, outX] += slope * weightedError;
#endif
                    }
                }
            }
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
    }

    public class MaxPoolLayer : ILayer
    {
        public readonly LayerInfo config;
        private readonly ILayer input;
        private readonly float[,,] activations; // neuron outputs
        private readonly float[,,] feedback; // learning via back propagation

        public readonly int depth; // shared by input and output
        public readonly int outputWidth;
        public readonly int outputHeight;
        public ILayer InLayer => input;
        public ILayer OutLayer { get; set; }
        public int Width => outputWidth;
        public int Height => outputHeight;
        public int Depth => depth;
        public float[,,] Outputs => activations;
        public float[,,] Feedback => feedback;

        public MaxPoolLayer(ILayer input, LayerInfo config)
        {
            Assert.IsTrue(config.activationType == Neuron.ActivationType.MaxPool);

            this.config = config;
            this.input = input;
            input.OutLayer = this;

            depth = input.Outputs.GetLength(0);

            int inputHeight = input.Outputs.GetLength(1);
            int inputWidth = input.Outputs.GetLength(2);

            outputHeight = config.CalculateOutputSize(inputHeight);
            outputWidth = config.CalculateOutputSize(inputWidth);

            activations = new float[depth, outputHeight, outputWidth];
            feedback = new float[depth, outputHeight, outputWidth];
        }

        public void Activate()
        {
            float kernelExtents = config.kernelSize / 2f;

            for (int outZ = 0; outZ < depth; outZ++)
            {
                for (int outY = 0; outY < outputHeight; outY++)
                {
                    for (int outX = 0; outX < outputWidth; outX++)
                    {
                        float maxValue = GetMaxInputInKernelWindow(kernelExtents, outZ, outY, outX, out int _, out int _);
                        activations[outZ, outY, outX] = maxValue;
                    }
                }
            }
        }

        public void BackPropagate()
        {
            // Only propagate feedback to the maximum input in the pooling window.
            // Other inputs in the pooling window will receive no feedback because
            // they did not contribute to the error.

            throw new NotImplementedException();
        }

        public void UpdateWeightsAndBiases(float learningRate)
        {
            // nothing to update
        }

        private float GetMaxInputInKernelWindow(float kernelExtents, int z, int outY, int outX, out int maxInY, out int maxInX)
        {
            // Initialize with invalid values
            maxInX = -1;
            maxInY = -1;
            // Ensure that at least one input will overwrite these invalid values
            float maxInput = float.NegativeInfinity;

            for (int kernelY = 0; kernelY < config.kernelSize; kernelY++)
            {
                int inY = Mathf.CeilToInt(outY * config.stride + kernelY - kernelExtents);
                if (inY < 0 || inY >= input.Height) continue;

                for (int kernelX = 0; kernelX < config.kernelSize; kernelX++)
                {
                    int inX = Mathf.CeilToInt(outX * config.stride + kernelX - kernelExtents);
                    if (inX < 0 || inX >= input.Width) continue;

                    float activation = input.Outputs[z, inY, inX];
                    if (activation > maxInput)
                    {
                        maxInX = inX;
                        maxInY = inY;
                        maxInput = activation;
                    }
                }
            }

            return maxInput;
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
        public void Activate() => OutLayer.Activate();
        public void BackPropagate() { }
        public void UpdateWeightsAndBiases(float learningRate) => OutLayer.UpdateWeightsAndBiases(learningRate);

        public InputLayer(int width, int height, int depth)
        {
            signals = new float[depth, width, height];
        }
        private readonly float[,,] signals;
    }

    public class OutputLayer : ILayer
    {
        public ILayer InLayer => inLayer;
        public ILayer OutLayer { get => null; set => throw new InvalidOperationException(); }
        public int Width => inLayer.Width;
        public int Height => inLayer.Height;
        public int Depth => inLayer.Depth;
        public float[,,] Outputs => inLayer.Outputs;
        public float[,,] Feedback => errors;
        public float[,,] Targets => targets;
        public void Activate() { }
        public void BackPropagate() => inLayer.BackPropagate();
        public void UpdateWeightsAndBiases(float learningRate) { }

        public OutputLayer(ILayer inLayer)
        {
            this.inLayer = inLayer;
            inLayer.OutLayer = this;
            this.errors = new float[Depth, Width, Height];
            this.targets = new float[Depth, Width, Height];
        }

        public float CalculateLoss()
        {
            float loss = 0f;

            for (int z = 0; z < Depth; z++)
            {
                for (int y = 0; y < Height; y++)
                {
                    for (int x = 0; x < Width; x++)
                    {
                        float error = Targets[z, y, x] - Outputs[z, y, x];
                        // loss is the mean squared error
                        loss += error * error;
                        Feedback[z, y, x] = 2f * error; // derivative of (error)^2 is 2*error
                    }
                }
            }

            loss = loss / (Width * Height * Depth);

            return loss;
        }

        private readonly ILayer inLayer;
        private readonly float[,,] errors;
        private readonly float[,,] targets;
    }

    public InputLayer InLayer { get; private set; }

    public OutputLayer OutLayer { get; private set; }

    public float Loss { get; private set; }

    public void Initialize(int width, int height, int depth)
    {
        InLayer = new InputLayer(width, height, depth);
        ILayer previousLayer = InLayer;
        foreach (var layerInfo in configuration)
        {
            previousLayer = layerInfo.CreateLayer(previousLayer);
        }
        OutLayer = new OutputLayer(previousLayer);
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
                    float w = kernels[kernelIndex, z, kernelY, kernelX];
                    weightedSum += o * w;
                }
            }
        }

        return weightedSum;
    }
}
