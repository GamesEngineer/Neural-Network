// Copyright 2021 Game-U Enterprises LLC
//#define DEBUG_CONV_LAYER
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.UI;
using TMPro;

[RequireComponent(typeof(ConvolutionalNeuralNetwork))]
public class ImageClassifierNetworkTest : MonoBehaviour
{
    public RawImage subjectImage;
    public RawImage lossGraphImage;
    public RawImage layerImage;
    public TextMeshProUGUI layerName;
    public RawImage kernelImage;
    public TextMeshProUGUI labelText;
    public TextMeshProUGUI resultText;
    public Image progressBar;
    public bool stepManually;

    [Range(1, 1000)]
    public int numEpochs = 100;

    [Range(10, 60000)]
    public int maxTrainingItems = 1000;

    private ConvolutionalNeuralNetwork brain;
    private Texture2D subjectTexture;
    private Texture2D graphTexture;
    private Texture2D layerTexture;
    private Texture2D kernelTexture;
    private float progressBarMaxWidth;
    private bool isPredictionStale;
    private readonly List<Texture2D> trainingImages = new List<Texture2D>();
    private readonly List<byte> trainingLabels = new List<byte>();
    private int debugLayerIndex = 1;
    private int debugChannelIndex;
    private int debugInChannelIndex;

    // MNIST database of handwritten digits
    const string TRAINING_IMAGES_FILENAME = @"Assets/train-images-idx3-ubyte/train-images.idx3-ubyte";
    const string TRAINING_LABELS_FILENAME = @"Assets/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
    const int IMAGE_SIZE = 28;

    private void Awake()
    {
        brain = GetComponent<ConvolutionalNeuralNetwork>();
#if DEBUG_CONV_LAYER
        brain.ChangeConfiguration(
            new List<ConvolutionalNeuralNetwork.LayerInfo>
            {
                new ConvolutionalNeuralNetwork.LayerInfo
                {
                    channelCount = 100,
                    activationType = Neuron.ActivationType.LeReLU,
                    kernelSize = 3,
                    stride = 1,
                },
            });
#endif

        subjectTexture = new Texture2D(IMAGE_SIZE, IMAGE_SIZE, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        subjectImage.texture = subjectTexture;

        graphTexture = new Texture2D(128, 64, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        lossGraphImage.texture = graphTexture;

        layerTexture = new Texture2D(IMAGE_SIZE, IMAGE_SIZE, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        layerImage.texture = layerTexture;

        kernelTexture = new Texture2D(15, 15, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        kernelImage.texture = kernelTexture;

        progressBarMaxWidth = progressBar.rectTransform.sizeDelta.x;
        progressBar.enabled = false;
    }

    private void Start()
    {
        int seed = (int)DateTime.Now.Ticks;
        UnityEngine.Random.InitState(seed);
        StartCoroutine(LoadTrainingData());
    }

    private void Update()
    {
        if (progressBar.enabled) return;

        if (Input.GetKeyDown(KeyCode.Return))
        {
            StartCoroutine(LearnTrainingData());
            return;
        }
        
        if (Input.GetKeyDown(KeyCode.R))
        {
            SelectRandomItem();
            return;
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            // Clear the subject texture to black
            subjectTexture.SetPixels32(new Color32[subjectTexture.width * subjectTexture.height]);
            subjectTexture.Apply();
            labelText.text = "";
            isPredictionStale = true;
            return;
        }

        if (Input.GetMouseButton(0) || Input.GetMouseButton(1) || Input.GetMouseButton(2))
        {
            labelText.text = "";
            resultText.text = "...";
            isPredictionStale = true;
            return; // Keep the framerate high while drawing by not thinking at the same time
        }

        if (isPredictionStale)
        {
            SetInputSignals(subjectTexture);
            UpdatePrediction();
        }

        DrawConvLayerAndKernel();
    }

    private void DrawLossGraph(int epoch, float meanLoss, float maxLoss)
    {
        if (epoch > numEpochs) return;
        int x;
        if (numEpochs <= graphTexture.width)
        {
            x = epoch - 1;
        }
        else
        {
            x = Mathf.FloorToInt(graphTexture.width * (float)(epoch - 1) / (float)(numEpochs - 1));
        }
        int y = Mathf.FloorToInt(Mathf.Sqrt(Mathf.Clamp01(maxLoss)) * (graphTexture.height - 1));
        graphTexture.SetPixel(x, y, new Color(1f, 0.5f, 0f));
        y = Mathf.FloorToInt(Mathf.Sqrt(Mathf.Clamp01(meanLoss)) * (graphTexture.height - 1));
        graphTexture.SetPixel(x, y, Color.white);
        graphTexture.Apply();
    }

    private void DrawConvLayerAndKernel()
    {
        if (brain.InLayer == null) return;

        if (Input.GetKeyDown(KeyCode.UpArrow)) { debugLayerIndex += 1; }
        if (Input.GetKeyDown(KeyCode.DownArrow)) { debugLayerIndex += brain.LayerCount - 1; }
        debugLayerIndex %= brain.LayerCount;

        INeuralLayer layer = brain.GetLayer(debugLayerIndex);

        if (Input.GetKeyDown(KeyCode.RightArrow)) { debugChannelIndex += 1; }
        if (Input.GetKeyDown(KeyCode.LeftArrow)) { debugChannelIndex += layer.Depth - 1; }
        debugChannelIndex %= layer.Depth;

        if (layer.InLayer != null)
        {
            if (Input.GetKeyDown(KeyCode.PageUp)) { debugInChannelIndex += 1; }
            if (Input.GetKeyDown(KeyCode.PageDown)) { debugInChannelIndex += layer.InLayer.Depth - 1; }
            debugInChannelIndex %= layer.InLayer.Depth;
        }
        else
        {
            debugInChannelIndex = 0;
        }

        layerName.text = $"[{debugLayerIndex}] {layer.GetType().Name}\nShape: {layer.Width}x{layer.Height}x{layer.Depth}\nChannel: {debugInChannelIndex}->{debugChannelIndex}\nActivation: {layer.Activation}";

        layerTexture.SetPixels32(new Color32[layerTexture.width * layerTexture.height]);
        float min = layer.ChannelMin[debugChannelIndex];
        float layerNormalizer = 1f / (layer.ChannelMax[debugChannelIndex] - min);
        if (layer.Width == 1 && layer.Height == 1)
        {
            min = 0f;
            layerNormalizer = 1f / Mathf.Sqrt(layer.InLayer.Width * layer.InLayer.Height * layer.InLayer.Depth / 2f);
        }
        for (int ty = 0; ty < layerTexture.height; ty++)
        {
            int y = ty * layer.Height / layerTexture.height;
            for (int tx = 0; tx < layerTexture.width; tx++)
            {
                int x = tx * layer.Width / layerTexture.width;
                float r = layer.Outputs[debugChannelIndex, y, x] - min;
                r *= layerNormalizer;
                Color c = float.IsNaN(r) || float.IsInfinity(r) ? Color.magenta : new Color(r, r, r, 1f);
                layerTexture.SetPixel(tx, layerTexture.height - 1 - ty, c);
            }
        }
        layerTexture.Apply();

        layerNormalizer = layer.InLayer != null ? layer.InLayer.Depth : 1f;
        var convLayer = layer as ConvolutionLayer;
        if (convLayer != null)
        {
            kernelTexture.SetPixels32(new Color32[kernelTexture.width * kernelTexture.height]);
            for (int ty = 0; ty < kernelTexture.height; ty++)
            {
                int n = ty * convLayer.config.kernelSize / kernelTexture.height;
                for (int tx = 0; tx < kernelTexture.width; tx++)
                {
                    int m = tx * convLayer.config.kernelSize / kernelTexture.width;
                    float r = convLayer.GetKernelValue(debugChannelIndex, m, n, debugInChannelIndex);
                    r *= layerNormalizer;
                    float b = convLayer.GetBias(debugChannelIndex) * 0.1f;
                    Color c = float.IsNaN(r) || float.IsInfinity(r) ? Color.magenta : new Color(r, b * b, -r, 1f);
                    kernelTexture.SetPixel(tx, kernelTexture.height - 1 - ty, c);
                }
            }
            kernelTexture.Apply();
            //kernelImage.gameObject.SetActive(true);
        }
        else
        {
            kernelTexture.SetPixels32(new Color32[kernelTexture.width * kernelTexture.height]);
            for (int ty = 0; ty < kernelTexture.height; ty++)
            {
                int y = ty * layer.Height / kernelTexture.height;
                for (int tx = 0; tx < kernelTexture.width; tx++)
                {
                    int x = tx * layer.Width / kernelTexture.width;
                    float r = layer.GetWeight(debugChannelIndex, debugInChannelIndex, y, x);
                    r *= layerNormalizer;
                    float b = layer.GetBias(debugChannelIndex) * 0.1f;
                    Color c = float.IsNaN(r) || float.IsInfinity(r) ? Color.magenta : new Color(r, b * b, -r, 1f);
                    kernelTexture.SetPixel(tx, kernelTexture.height - 1 - ty, c);
                }
            }
            kernelTexture.Apply();
            //kernelImage.gameObject.SetActive(false);
        }

        layerImage.enabled = true;
        kernelImage.enabled = true;
    }

    public float GetDebugLayerOutput(int x, int y)
    {
        INeuralLayer layer = brain.GetLayer(debugLayerIndex);
        if (layer == null) return 0f;
        if (x < 0 || x >= layer.Width || y < 0 || y >= layer.Height) return 0f;
        return layer.Outputs[debugChannelIndex, y, x];
    }

    private IEnumerator LoadTrainingData()
    {
        labelText.text = "LOADING";
        resultText.text = 0.ToString("P0");
        progressBar.enabled = true;

        if (File.Exists(TRAINING_IMAGES_FILENAME))
        {
            using (FileStream stream = File.Open(TRAINING_IMAGES_FILENAME, FileMode.Open))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    int magic = ReadIntBigEndian(reader);
                    int numImages = ReadIntBigEndian(reader);
                    numImages = numImages > maxTrainingItems ? maxTrainingItems : numImages;
                    int numRows = ReadIntBigEndian(reader);
                    int numColumns = ReadIntBigEndian(reader);
                    Debug.Log($"{magic}: {numImages} x {numRows} x {numColumns}");
                    Assert.IsTrue(magic == 2051);
                    var t = DateTime.Now;
                    var s = new TimeSpan(0, 0, 0, 0, 33);
                    for (int i = 0; i < numImages; i++)
                    {
                        Texture2D texture = new Texture2D(numColumns, numRows, TextureFormat.RGB24, false);
                        trainingImages.Add(texture);
                        for (int row = numRows - 1; row >= 0; row--)
                        {
                            for (int col = 0; col < numColumns; col++)
                            {
                                byte p = reader.ReadByte();
                                Color32 c = new Color32(p, p, p, 255);
                                texture.SetPixel(col, row, c);
                            }
                        }
                        texture.Apply();

                        if (DateTime.Now - t > s)
                        {
                            t = DateTime.Now;
                            UpdateProgressBar(i, numImages - 1);
                            Graphics.CopyTexture(texture, subjectTexture);
                            subjectTexture.Apply();
                            yield return null;
                        }
                    }
                }
            }
        }

        if (File.Exists(TRAINING_LABELS_FILENAME))
        {
            using (FileStream stream = File.Open(TRAINING_LABELS_FILENAME, FileMode.Open))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    int magic = ReadIntBigEndian(reader);
                    int numLabels = ReadIntBigEndian(reader);
                    numLabels = numLabels > maxTrainingItems ? maxTrainingItems : numLabels;
                    Debug.Log($"{magic}: {numLabels}");
                    Assert.IsTrue(magic == 2049);
                    for (int i = 0; i < numLabels; i++)
                    {
                        trainingLabels.Add(reader.ReadByte());
                    }
                }
            }
        }

        SelectRandomItem();

        resultText.text = "Untrained";
        progressBar.enabled = false;
    }

    private int ReadIntBigEndian(BinaryReader reader)
    {
        byte[] data = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(data);
        }
        return BitConverter.ToInt32(data, 0);
    }

    private IEnumerator LearnTrainingData()
    {
        labelText.text = "TRAINING";
        progressBar.enabled = true;
#if DEBUG_CONV_LAYER
        brain.Initialize(IMAGE_SIZE, IMAGE_SIZE, depth: 1, Neuron.ActivationType.None);
#else
        brain.Initialize(IMAGE_SIZE, IMAGE_SIZE, depth: 1, Neuron.ActivationType.SoftMax);
#endif
        graphTexture.SetPixels32(new Color32[graphTexture.width * graphTexture.height]);
        graphTexture.Apply();

        int[] shuffledIndices = new int[trainingImages.Count];
        for (int i = 0; i < shuffledIndices.Length; i++)
        {
            shuffledIndices[i] = i;
        }

        float totalIterations = numEpochs * trainingImages.Count;
        float iteration = 1f;

        for (int epoch = 1; epoch <= numEpochs; epoch++)
        {
#if false
            float learningRateDecay = (float)(numEpochs - epoch + 1) / (float)(numEpochs);
#else
            float learningRateDecay = 1f / (1f + 0.01f * epoch);
#endif
            Tensor.Shuffle(shuffledIndices);

            float maxLoss = 0f;
            float meanLoss = 0f;
            for (int j = 0; j < shuffledIndices.Length; j++)
            {
                int shuffledIndex = shuffledIndices[j];
                Texture2D trainingImage = trainingImages[shuffledIndex];
                int trainingLabel = trainingLabels[shuffledIndex];
                UpdateProgressBar(iteration++, totalIterations);
                LearnImage(trainingImage, trainingLabel, learningRateDecay);
                meanLoss += brain.Loss;
                maxLoss = Mathf.Max(maxLoss, brain.Loss);
                Graphics.CopyTexture(trainingImage, subjectTexture);
                DrawConvLayerAndKernel();
                if (stepManually)
                {
                    yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));
                    yield return new WaitUntil(() => Input.GetKeyUp(KeyCode.Space));
                }
                else
                {
                    if (Input.GetKeyDown(KeyCode.Space))
                    {
                        yield return new WaitUntil(() => Input.GetKeyUp(KeyCode.Space));
                        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));
                        yield return new WaitUntil(() => Input.GetKeyUp(KeyCode.Space));
                    }
                    yield return null;
                }
            }
            meanLoss /= (float)shuffledIndices.Length;
            DrawLossGraph(epoch, meanLoss, maxLoss);
            Debug.Log($"[{epoch}] Loss: {meanLoss:F4} | {maxLoss:F4}");
        }

        progressBar.enabled = false;
        SelectRandomItem();
    }

    private void LearnImage(Texture2D image, int label, float learningRateMultiplier)
    {
        SetInputSignals(image);
        SetOutputTargets(label);
        brain.Learn(learningRateMultiplier);
    }

    private void SetInputSignals(Texture2D image)
    {
        if (brain.InLayer == null) return;

        for (int y = 0; y < image.height; y++)
        {
            for (int x = 0; x < image.width; x++)
            {
                brain.InLayer.signals[0, y, x] = image.GetPixel(x, image.height - 1 - y).r;
            }
        }
    }

    private void SetOutputTargets(int label)
    {
        if (brain.OutLayer == null) return;

#if DEBUG_CONV_LAYER
        Assert.IsTrue(brain.OutLayer.Height == brain.InLayer.Height);
        Assert.IsTrue(brain.OutLayer.Width == brain.InLayer.Width);
        float[,] kernel = new float[3, 3]
        {
#if true
            // Pseudo-emboss (using an asymetrical filter helps to find bugs)
            { -0.75f, -0.20f, -0.30f },
            { -0.75f, +0.50f, +0.125f },
            { +0.00f, +0.625f, +1.25f },
#else
            { 1.0f, 0.0f, 0.0f },
            { 0.0f, 0.0f, 0.0f },
            { 0.0f, 0.0f, 0.0f },
#endif
        };
        Func<float, float> activate = Neuron.ActivationFunctions[(int)brain.InLayer.OutLayer.Activation];
        for (int y = 0; y < brain.OutLayer.Height; y++)
        {
            for (int x = 0; x < brain.OutLayer.Width; x++)
            {
                float dst = 0f;
                for (int b = -1; b <= 1; b++)
                {
                    int n = y + b;
                    if (n < 0 || n >= brain.InLayer.Height) continue;

                    for (int a = -1; a <= 1; a++)
                    {
                        int m = x + a;
                        if (m < 0 || m >= brain.InLayer.Width) continue;

                        float src = brain.InLayer.signals[0, n, m];
                        dst += src * kernel[b + 1, a + 1];
                    }
                }
                float t = activate(dst);
                for (int z = 0; z < brain.OutLayer.Depth; z++)
                {
                    brain.OutLayer.Targets[z, y, x] = t;
                }
            }
        }
#else
        float depth = brain.OutLayer.Targets.GetLength(0);
        for (int n = 0; n < depth; n++)
        {
            brain.OutLayer.Targets[n, 0, 0] = label == n ? 1f : 0f;
        }
#endif
    }

    private void UpdatePrediction()
    {
        brain.Think();

        // Select the "winning" output neuron, and use its index as the predicted label
        int winner = -1;
        float winnerValue = -1f;
        System.Text.StringBuilder predictionsStr = new System.Text.StringBuilder("Predictions:");
        if (brain.OutLayer != null)
        {
            for (int i = 0; i < brain.OutLayer.Depth; i++)
            {
                float v = brain.OutLayer.Outputs[i, 0, 0];
                predictionsStr.Append($"  {i}, {v:P2}");
                if (v > winnerValue)
                {
                    winnerValue = v;
                    winner = i;
                }
            }
        }

        resultText.text = winner >= 0 ? $"Guess: {winner}" : "??";
        isPredictionStale = false;
    }

    private void SelectRandomItem()
    {
        // Clear the subject texture to black
        subjectTexture.SetPixels32(new Color32[subjectTexture.width * subjectTexture.height]);

        // Pick a random item (image, label) from the training set
        int r = UnityEngine.Random.Range(0, trainingImages.Count);
        labelText.text = $"Label: {trainingLabels[r]}";
        Graphics.CopyTexture(trainingImages[r], subjectTexture);
        subjectTexture.Apply();
        isPredictionStale = true;

        SetInputSignals(trainingImages[r]);
        SetOutputTargets(trainingLabels[r]);
    }

    private void UpdateProgressBar(float iteration, float totalIterations)
    {
        float progress = iteration++ / totalIterations;
        Vector2 sd = progressBar.rectTransform.sizeDelta;
        sd.x = progressBarMaxWidth * progress;
        progressBar.rectTransform.sizeDelta = sd;
        resultText.text = progress.ToString("P0");
    }
}
