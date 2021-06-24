using System;
using UnityEngine;

public static class Neuron
{
    public enum ActivationType
    {
        None,
        Tanh,
        Sigmoid,
        ReLU,
        ELU,
        LeReLU,

        MaxPool,
        SoftMax,
    }

    public static Func<float, float>[] ActivationFunctions = { Identity, Tanh, Sigmoid, ReLU, ELU, LeReLU, null, null };
    public static Func<float, float>[] ActivationDerivatives = { Identity, dTanh, dSigmoid, dReLU, dELU, dLeReLU, null, null };

    private static float Identity(float x) => x;

    // Range: [-1..+1]
    private static float Tanh(float x)
    {
        x = Mathf.Clamp(x, -38f, 38f);
        float e2x = Mathf.Exp(2f * x);
        return (e2x - 1f) / (e2x + 1f);
    }

    private static float dTanh(float x)
    {
        float t = Tanh(x);
        return 1f - t * t;
    }

    // Range: [0..1]
    private static float Sigmoid(float x)
    {
        x = Mathf.Clamp(x, -76f, 76f);
        return 1f / (1f + Mathf.Exp(-x));
    }

    private static float dSigmoid(float x)
    {
        float s = Sigmoid(x);
        return s * (1f - s);
    }

    // Range: [0..infinity]
    private static float ReLU(float x)
    {
        //return Mathf.Max(0f, x);
        return x > 0f ? x : 0f;
    }

    private static float dReLU(float x)
    {
        return x > 0f ? 1f : 0f;
    }

    private static float ELU(float x)
    {
        return x >= 0f ? x : 0.1f * Mathf.Exp(x - 1f);
    }

    private static float dELU(float x)
    {
        return x >= 0f ? 1f : 0.1f * Mathf.Exp(x);
    }

    private static float LeReLU(float x)
    {
        return x > 0f ? x : 0.1f * x;
    }

    private static float dLeReLU(float x)
    {
        return x > 0f ? 1f : 0.1f;
    }
}
