#define SHOW_IMAGES_WHILE_LOADING
#define SHOW_IMAGES_WHILE_TRAINING
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

        if (brain.InLayer != null)
        {
            ConvolutionalNeuralNetwork.ILayer conv1_Layer = brain.InLayer.OutLayer;
            int z = (int)Time.time % conv1_Layer.Depth;
            for (int y = 0; y < IMAGE_SIZE; y++)
            {
                for (int x = 0; x < IMAGE_SIZE; x++)
                {
                    float r = conv1_Layer.Outputs[z, y, x];
                    Color c = new Color(r, r*r, -r, 1f);
                    conv1_Texture.SetPixel(x, y, c);
                }
            }
            conv1_Texture.Apply();
        }
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
        brain.Initialize(IMAGE_SIZE, IMAGE_SIZE, depth: 1);

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
                yield return null;
            }
        }

        progressBar.enabled = false;
        SelectRandomItem();
    }

    private void LearnImage(Texture2D image, int label, float learningRateMultiplier)
    {
        for (int y = 0; y < image.height; y++)
        {
            for (int x = 0; x < image.width; x++)
            {
                brain.InLayer.signals[0, y, x] = image.GetPixel(x, y).r;
            }
        }
        for (int n = 0; n < 10; n++)
        {
            brain.OutLayer.Targets[n, 0, 0] = label == n ? 1f : 0f;
        }
        brain.Learn(learningRateMultiplier);
    }

    private void UpdatePrediction()
    {
        brain.Think();

        // Select the "winning" output neuron, and use its index as the predicted label
        int winner = -1;
        float winnerValue = 0.5f;
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
