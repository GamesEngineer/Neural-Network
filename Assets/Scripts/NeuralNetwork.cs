using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class NeuralNetwork : MonoBehaviour
{
    public class Layer
    {
        public readonly float[] inputs; // "owned" by previous layer
        public readonly float[] preOutputs; // pre-activation value of neurons
        public readonly float[] outputs; // neuron outputs
        public readonly float[,] weights; // matrix [outputs, inputs] of synaptic weights (one row of weights for each output neuron)
        public readonly float[] biases; // offset added to weighted sum of inputs
        public readonly float[] feedback; // learning via back propagation
        public readonly Func<float, float> activationFunc; // neuron activation function
        public readonly Func<float, float> dActivationFunc; // derivative of activation fuction

        public Layer(float[] inputs, int numOutputs, Neuron.ActivationType activationType)
        {
            int numInputs = inputs.Length;
            this.inputs = inputs;
            preOutputs = new float[numOutputs];
            outputs = new float[numOutputs];
            weights = new float[numOutputs, numInputs];
            biases = new float[numOutputs];
            feedback = new float[numOutputs];
            activationFunc = Neuron.ActivationFunctions[(int)activationType];
            dActivationFunc = Neuron.ActivationDerivatives[(int)activationType];

            // Initialize the each neuron's bias with random noise
            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                biases[outIndex] = UnityEngine.Random.Range(-0.1f, 0.1f);
            }

            // Initialize the matrix of synaptic weights with random noise
            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                for (int inIndex = 0; inIndex < numInputs; inIndex++)
                {
                    weights[outIndex, inIndex] = UnityEngine.Random.Range(-0.5f, 0.5f);
                }
            }
        }

        /// <summary>
        /// Activate this layer's ouputs based on its current inputs.
        /// This method is used by the neural network during its Think phase.
        /// </summary>
        public void Activate()
        {
            int numInputs = inputs.Length;
            int numOutputs = outputs.Length;
            Assert.IsTrue(weights.GetLength(0) == numOutputs);
            Assert.IsTrue(weights.GetLength(1) == numInputs);

            /// For each neuron in the layer, the inputs are fed forward through the layer.
            /// This involves multiplying each input by the neuron's synaptic weights, adding
            /// the neuron's bias value, and then activating the neuron using the layer's
            /// activation function.
            for (int outIndex = 0; outIndex < numOutputs; outIndex++)
            {
                float weightedSum = 0f;
                for (int inIndex = 0; inIndex < numInputs; inIndex++)
                {
                    weightedSum += inputs[inIndex] * weights[outIndex, inIndex];
                }
                float preOutput = weightedSum + biases[outIndex];
                preOutputs[outIndex] = preOutput;
                outputs[outIndex] = activationFunc(preOutput);
            }
        }

        /// <summary>
        /// Updates this layer's feedback by propagating errors backward through the layer.
        /// This method is used by the neural network during its Learn phase.
        /// </summary>
        /// <param name="errors">Errors/feedback to be propagated backwards through this layer.
        /// Mathematically, these are partial derivatives of the neural network's loss function with respect to this layer's preOutputs.</param>
        /// <param name="nextLayerWeights">Weights of the next layer when the errors were computed. Should be null for the output layer.</param>
        public void BackPropagate(float[] errors, float[,] nextLayerWeights)
        {
            int numInputs = inputs.Length;
            int numOutputs = outputs.Length;
            int numErrors = errors.Length;
            Assert.IsTrue(weights.GetLength(0) == numOutputs);
            Assert.IsTrue(weights.GetLength(1) == numInputs);
            Assert.IsTrue(feedback.Length == numOutputs);

            // When computing this layer's feedback signals, we must take into account its
            // connection with the next layer in order to properly calculate the
            // partial derivatives of the loss with respect to this layer's pre-ouputs.
            // However, if this layer is an "ouput layer," then it is not connected to a
            // next layer, and so the calculation of its feedback signals is simpler.
            bool isOutputLayer = (nextLayerWeights == null);

            if (isOutputLayer)
            {
                Assert.IsTrue(numErrors == numOutputs);
                for (int outIter = 0; outIter < numOutputs; outIter++)
                {
                    float slope = dActivationFunc(preOutputs[outIter]);
                    feedback[outIter] += slope * errors[outIter];
                }
            }
            else
            {
                Assert.IsTrue(nextLayerWeights.GetLength(0) == numErrors);
                Assert.IsTrue(nextLayerWeights.GetLength(1) == numOutputs);
                for (int outIter = 0; outIter < numOutputs; outIter++)
                {
                    float slope = dActivationFunc(preOutputs[outIter]);
                    float weightedError = 0f;
                    for (int nextIter = 0; nextIter < numErrors; nextIter++)
                    {
                        weightedError += errors[nextIter] * nextLayerWeights[nextIter, outIter];
                    }
                    feedback[outIter] += slope * weightedError;
                }
            }
        }

        /// <summary>
        /// Apply this layer's feeback to the weights and biases.
        /// This method is used by the neural network during its Learn phase.
        /// </summary>
        /// <param name="learningRate"></param>
        public void UpdateWeightsAndBiases(float learningRate)
        {
            int numInputs = inputs.Length;
            int numOutputs = outputs.Length;
            Assert.IsTrue(weights.GetLength(0) == numOutputs);
            Assert.IsTrue(weights.GetLength(1) == numInputs);
            Assert.IsTrue(biases.Length == numOutputs);
            Assert.IsTrue(feedback.Length == numOutputs);

            for (int outIter = 0; outIter < numOutputs; outIter++)
            {
                float change = learningRate * feedback[outIter];
                biases[outIter] += change;
                for (int inIter = 0; inIter < numInputs; inIter++)
                {
                    weights[outIter, inIter] += change * inputs[inIter];
                }
                // Clear the feedback so we don't use it again (useful for batch learning)
                feedback[outIter] = 0f;
            }
        }
    }

    #region Network Configuration & Tuning Parameters
    
    [Serializable]
    public struct LayerInfo
    {
        public int neuronCount;
        public Neuron.ActivationType activationType;
    }

    [SerializeField]
    protected List<LayerInfo> configuration = new List<LayerInfo>();

    [SerializeField, Range(0.0001f, 0.1f)]
    protected float learningRate = 0.01f;

    #endregion

    #region Properties

    public Layer InputLayer => layers[0];
    public Layer OutputLayer => layers[layers.Count - 1];
    public float[] SensoryInputs { get; private set; }
    public float[] Targets { get; private set; }
    public float[] Results { get; private set; }
    public float[] Errors { get; private set; }
    public float Loss { get; private set; }

    #endregion

    private List<Layer> layers;

    /// <summary>
    /// Creates the neural network from the serialized configuration and the
    /// specified number of inputs. The network's weights and biases are
    /// initialized with random values.
    /// </summary>
    /// <param name="numInputs">Count of SensoryInputs values</param>
    public void Initialize(int numInputs)
    {
        SensoryInputs = new float[numInputs];

        // Create the layers and connect them to each other
        layers = new List<Layer>(configuration.Count);
        float[] inputs = SensoryInputs;
        foreach (var layerInfo in configuration)
        {
            var layer = new Layer(inputs, layerInfo.neuronCount, layerInfo.activationType);
            layers.Add(layer);
            // Connect this layer's outputs to the next layer's inputs
            inputs = layer.outputs;
        }

        Results = OutputLayer.outputs;
        Targets = new float[Results.Length];
        Errors = new float[Results.Length];
    }

    /// <summary>
    /// Feeds the SensoryInputs forward through the network.
    /// This generates a prediction that is stored in the Results.
    /// </summary>
    public void Think()
    {
        foreach (var layer in layers)
        {
            layer.Activate();
        }
    }

    /// <summary>
    /// Attempts to learn the association between the current SensoryInputs
    /// with the current Targets. This will update the Loss value as a metric
    /// that can be used to know how well the network predicted the Targets.
    /// </summary>
    /// <param name="learningRateMultiplier">Used to adjust the learning rate each epoch.</param>
    public void Learn(float learningRateMultiplier = 1f)
    {
        Think();
        Loss = CalculateLoss_MSE(Targets, Results, Errors);

        // Propagate errors backward through the network
        float[] feedback = Errors;
        float[,] nextLayerWeights = null;
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            Layer layer = layers[i];
            layer.BackPropagate(feedback, nextLayerWeights);
            feedback = layer.feedback;
            nextLayerWeights = layer.weights;
        }

        // Update each layer's weights and biases with the error feedback
        foreach (var layer in layers)
        {
            layer.UpdateWeightsAndBiases(learningRate * learningRateMultiplier);
        }
    }

    /// <summary>
    /// Calculates "loss" as the mean squared error between targets and predictions.
    /// </summary>
    /// <param name="targets">The expected values</param>
    /// <param name="predictions">The predicted values</param>
    /// <param name="errors">twice the difference between targets and predictions</param>
    /// <returns>the total loss value (sum of squared errors)</returns>
    public static float CalculateLoss_MSE(float[] targets, float[] predictions, float[] errors)
    {
        Assert.IsTrue(targets.Length == predictions.Length);
        Assert.IsTrue(errors.Length == predictions.Length);
        float loss = 0f;
        for (int i = 0; i < predictions.Length; i++)
        {
            // squared error
            float error = targets[i] - predictions[i];
            loss += error * error;
            errors[i] = 2f * error; // derivative of (error)^2 is 2*error
        }
        return loss / predictions.Length;
    }
}
