#define SHOW_IMAGES_WHILE_LOADING
#define SHOW_IMAGES_WHILE_TRAINING

//#define DEBUG_CONV_LAYER
//#define DEBUG_POOL_LAYER
//#define DEBUG_MANUAL_TRAINING_STEPS

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
    public RawImage conv1_Image;
    public RawImage conv1Kernel_Image;
    
    public TextMeshProUGUI labelText;
    
    public TextMeshProUGUI resultText;

    public Image progressBar;

    [Range(1, 1000)]
    public int numEpochs = 10;

    [Range(100, 60000)]
    public int maxTrainingItems = 1000;

    private ConvolutionalNeuralNetwork brain;
    private Texture2D subjectTexture;
    private Texture2D conv1_Texture;
    private Texture2D conv1Kernel_Texture;
    private float progressBarMaxWidth;
    private bool isPredictionStale;
    private readonly List<Texture2D> trainingImages = new List<Texture2D>();
    private readonly List<byte> trainingLabels = new List<byte>();

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
                    channelCount = 1,
                    activationType = Neuron.ActivationType.ReLU,
                    kernelSize = 3,
                    stride = 1,
                }
            });
#elif DEBUG_POOL_LAYER
        brain.ChangeConfiguration(
            new List<ConvolutionalNeuralNetwork.LayerInfo>
            {
                        new ConvolutionalNeuralNetwork.LayerInfo
                        {
                            channelCount = 1,
                            activationType = Neuron.ActivationType.MaxPool,
                            kernelSize = 2,
                            stride = 2,
                        }
            });
#endif

        subjectTexture = new Texture2D(IMAGE_SIZE, IMAGE_SIZE, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        subjectImage.texture = subjectTexture;

        conv1_Texture = new Texture2D(IMAGE_SIZE, IMAGE_SIZE, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        conv1_Image.texture = conv1_Texture;

        conv1Kernel_Texture = new Texture2D(3, 3, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        conv1Kernel_Image.texture = conv1Kernel_Texture;

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
            UpdatePrediction();
        }

        DrawConvLayerAndKernel();
    }

    private int debugLayerIndex = 1;
    private int debugChannelIndex;
    private void DrawConvLayerAndKernel()
    {
        if (brain.InLayer == null) return;

        if (Input.GetKeyDown(KeyCode.UpArrow)) { debugLayerIndex += 1; }
        if (Input.GetKeyDown(KeyCode.DownArrow)) { debugLayerIndex += brain.LayerCount - 1; }
        debugLayerIndex %= brain.LayerCount;

        ConvolutionalNeuralNetwork.ILayer layer = brain.GetLayer(debugLayerIndex);

        if (Input.GetKeyDown(KeyCode.RightArrow)) { debugChannelIndex += 1; }
        if (Input.GetKeyDown(KeyCode.LeftArrow)) { debugChannelIndex += layer.Depth - 1; }
        debugChannelIndex %= layer.Depth;

        conv1_Texture.SetPixels32(new Color32[conv1_Texture.width * conv1_Texture.height]);
        for (int y = 0; y < layer.Height; y++)
        {
            for (int x = 0; x < layer.Width; x++)
            {
                float r = layer.Outputs[debugChannelIndex, y, x];
                Color c = new Color(r, 2f * r * r, -r, 1f);
                conv1_Texture.SetPixel(x, y, c);
            }
        }
        conv1_Texture.Apply();

        conv1Kernel_Texture.SetPixels32(new Color32[conv1Kernel_Texture.width * conv1Kernel_Texture.height]);
        var convLayer = layer as ConvolutionalNeuralNetwork.ConvolutionLayer;
        if (convLayer != null)
        {
            for (int n = 0; n < 3; n++)
            {
                for (int m = 0; m < 3; m++)
                {
                    float r = convLayer.GetKernelValue(0, debugChannelIndex, m, n);
                    Color c = new Color(r, 2f * r * r, -r, 1f);
                    conv1Kernel_Texture.SetPixel(m, n, c);
                }
            }
        }
        conv1Kernel_Texture.Apply();

        conv1_Image.enabled = true;
        conv1Kernel_Image.enabled = true;
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
#if SHOW_IMAGES_WHILE_LOADING
                            Graphics.CopyTexture(texture, subjectTexture);
                            subjectTexture.Apply();
#endif
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
#if DEBUG_CONV_LAYER || DEBUG_POOL_LAYER
        brain.Initialize(IMAGE_SIZE, IMAGE_SIZE, depth: 1, Neuron.ActivationType.None);
#else
        brain.Initialize(IMAGE_SIZE, IMAGE_SIZE, depth: 1, Neuron.ActivationType.SoftMax);
#endif

        int[] shuffledIndices = new int[trainingImages.Count];
        for (int i = 0; i < shuffledIndices.Length; i++)
        {
            shuffledIndices[i] = i;
        }

        float totalIterations = numEpochs * trainingImages.Count;
        float iteration = 1f;

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            float progress = (float)(numEpochs - 1 - epoch) / (float)(numEpochs - 1);
            Shuffle(shuffledIndices);

            for (int j = 0; j < shuffledIndices.Length; j++)
            {
                int shuffledIndex = shuffledIndices[j];
                Texture2D trainingImage = trainingImages[shuffledIndex];
                int trainingLabel = trainingLabels[shuffledIndex];
                UpdateProgressBar(iteration++, totalIterations);
                LearnImage(trainingImage, trainingLabel, progress);
#if SHOW_IMAGES_WHILE_TRAINING
                Graphics.CopyTexture(trainingImage, subjectTexture);
#endif
                DrawConvLayerAndKernel();
#if DEBUG_MANUAL_TRAINING_STEPS
                yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));
                yield return new WaitUntil(() => Input.GetKeyUp(KeyCode.Space));
#else
                yield return null;
#endif
            }
            Debug.Log($"Loss: {brain.Loss:F4}");
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

    private void SetOutputTargets(int label)
    {
        if (brain.OutLayer == null) return;

#if DEBUG_CONV_LAYER
        for (int y = 0; y < image.height; y++)
        {
            for (int x = 0; x < image.width; x++)
            {
                brain.OutLayer.Targets[0, y, x] = image.GetPixel(x, y).r;
            }
        }
#elif DEBUG_POOL_LAYER
        for (int y = 0; y < image.height / 2; y++)
        {
            for (int x = 0; x < image.width / 2; x++)
            {
                float m = 0f;
                int a = 2 * x;
                int b = 2 * y;
                m = Mathf.Max(m, image.GetPixel(a, b).r);
                m = Mathf.Max(m, image.GetPixel(a+1, b).r);
                m = Mathf.Max(m, image.GetPixel(a, b+1).r);
                m = Mathf.Max(m, image.GetPixel(a+1, b+1).r);
                brain.OutLayer.Targets[0, y, x] = m;
            }
        }
#else
        for (int n = 0; n < 10; n++)
        {
            brain.OutLayer.Targets[n, 0, 0] = label == n ? 1f : 0f;
        }
#endif
    }

    private void SetInputSignals(Texture2D image)
    {
        if (brain.InLayer == null) return;

        for (int y = 0; y < image.height; y++)
        {
            for (int x = 0; x < image.width; x++)
            {
                brain.InLayer.signals[0, y, x] = image.GetPixel(x, y).r;
            }
        }
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

        Debug.Log(predictionsStr.ToString());
        resultText.text = winner >= 0 ? winner.ToString() : "??";
        isPredictionStale = false;
    }

    private void SelectRandomItem()
    {
        // Clear the subject texture to black
        subjectTexture.SetPixels32(new Color32[subjectTexture.width * subjectTexture.height]);

        // Pick a random item (image, label) from the training set
        int r = UnityEngine.Random.Range(0, trainingImages.Count);
        labelText.text = $"LABEL: {trainingLabels[r]}";
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

    private static void Shuffle<T>(T[] a)
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
