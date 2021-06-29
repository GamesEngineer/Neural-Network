// Copyright 2021 Game-U Enterprises LLC
using UnityEngine;
using UnityEngine.Assertions;

public static class Tensor
{
    // The following implementations of cross-correlation and convolution operators are informed and inspired by:
    //     1) https://en.wikipedia.org/wiki/Cross-correlation
    //     2) https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
    //     3) https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

    /// <summary>
    /// Computes the cross-correlation (sliding inner-product) of tensor[...] * kernels[channelIndex, ...].
    /// </summary>
    /// <param name="column">Index of column in the OUTPUT tensor</param>
    /// <param name="row">Index of row in the OUTPUT tensor</param>
    /// <param name="channelIndex">Index of the kernel filter to use</param>
    /// <param name="kernels">Array of kernel filters [channels, tensor depth, kernel size, kernel size]</param>
    /// <param name="tensor">Input tensor [depth, height, width] </param>
    /// <returns>The cross-correlation of tensor[...] * kernels[channelIndex, ...]</returns>
    public static float CrossCorrelation(int column, int row, int channelIndex, float[,,] kernels, float[,,] tensor, int stride)
    {
        Assert.IsTrue(kernels.GetLength(1) == kernels.GetLength(2));
        int depth = tensor.GetLength(0);
        int height = tensor.GetLength(1);
        int width = tensor.GetLength(2);
        int kernelSize = kernels.GetLength(1);
        int kernelExtents = kernelSize / 2;
        float weightedSum = 0f;

        for (int kernelY = 0; kernelY < kernelSize; kernelY++)
        {
            int y = row * stride + kernelY - kernelExtents;
            y = Mathf.Clamp(y, 0, height - 1);

            for (int kernelX = 0; kernelX < kernelSize; kernelX++)
            {
                int x = column * stride + kernelX - kernelExtents;
                x = Mathf.Clamp(x, 0, width - 1);

                float w = kernels[channelIndex, kernelY, kernelX];
                for (int z = 0; z < depth; z++)
                {
                    float i = tensor[z, y, x];
                    weightedSum += i * w;
                }
            }
        }

        return weightedSum;
    }

    /// <summary>
    /// Computes the convolution (sliding inner-product) of tensor[...] * kernels[channelIndex,...].
    /// </summary>
    /// <param name="column">Index of column in the OUTPUT tensor</param>
    /// <param name="row">Index of row in the OUTPUT tensor</param>
    /// <param name="channelIndex">Index of the kernel filter to use</param>
    /// <param name="kernels">Array of kernel filters [channels, tensor depth, kernel size, kernel size]</param>
    /// <param name="tensor">Input tensor [depth, height, width] </param>
    /// <returns>The convolution of tensor[...] * kernels[channelIndex,...]</returns>
    public static float Convolution(int column, int row, int channelIndex, float[,,] kernels, float[,,] tensor, int stride)
    {
        Assert.IsTrue(kernels.GetLength(1) == kernels.GetLength(2));
        int depth = tensor.GetLength(0);
        int height = tensor.GetLength(1);
        int width = tensor.GetLength(2);
        int kernelSize = kernels.GetLength(1);
        int kernelExtents = kernelSize / 2;
        float weightedSum = 0f;

        for (int n = 0; n < kernelSize; n++)
        {
            int y = row * stride + n - kernelExtents;
            y = Mathf.Clamp(y, 0, height - 1);

            // Convolution is the same as a cross-correlation with a kernel that is rotated 180°
            int kernelY = kernelSize - 1 - n;

            for (int m = 0; m < kernelSize; m++)
            {
                int x = column * stride + m - kernelExtents;
                x = Mathf.Clamp(x, 0, width - 1);

                int kernelX = kernelSize - 1 - m;
                float w = kernels[channelIndex, kernelY, kernelX];
                for (int z = 0; z < depth; z++)
                {
                    float o = tensor[z, y, x];
                    weightedSum += o * w;
                }
            }
        }

        return weightedSum;
    }

    public static void Fill<T>(T[] array, T value)
    {
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = value;
        }
    }

    public static void Shuffle<T>(T[] a)
    {
        int l = a.Length;
        for (int n = 0; n < l; n++)
        {
            int m = UnityEngine.Random.Range(n, l);
            var tmp = a[n];
            a[n] = a[m];
            a[m] = tmp;
        }
    }
}
