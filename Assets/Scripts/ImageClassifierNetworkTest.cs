using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.UI;
using TMPro;

[RequireComponent(typeof(ConvolutionalNeuralNetwork))]
public class ImageClassifierNetworkTest : MonoBehaviour
{
    public RawImage subjectImage;
    public TextMeshProUGUI labelText;
    public TextMeshProUGUI resultText;
    public int numEpochs = 1000;

    private ConvolutionalNeuralNetwork brain;
    private Texture2D subjectTexture;

    private readonly List<Texture2D> trainingImages = new List<Texture2D>();
    private readonly List<byte> trainingLabels = new List<byte>();

    private void Awake()
    {
        brain = GetComponent<ConvolutionalNeuralNetwork>();

        subjectTexture = new Texture2D(28, 28, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        subjectImage.texture = subjectTexture;
    }

    private void Start()
    {
        int seed = (int)DateTime.Now.Ticks;
        UnityEngine.Random.InitState(seed);

        LoadTrainingData();
        Reset();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            Reset();
        }

        // Keep the framerate high while drawing by not thinking at the same time
        if (Input.GetMouseButton(0) || Input.GetMouseButton(1) || Input.GetMouseButton(2))
        {
            resultText.enabled = false;
            return;
        }

        if (!resultText.enabled)
        {
            brain.Think();

            int winner = -1;
            float winnerValue = float.NegativeInfinity;
            for (int i = 0; i < brain.OutLayer.Outputs.GetLength(0); i++)
            {
                float v = brain.OutLayer.Outputs[i, 0, 0];
                if (v > winnerValue)
                {
                    winnerValue = v;
                    winner = i;
                }
                //Debug.Log($"{i}: {v}");
            }

            resultText.enabled = true;
            resultText.text = winner.ToString();
        }
    }

    private void LoadTrainingData()
    {
        const string trainingImagesFileName = @"Assets/train-images-idx3-ubyte/train-images.idx3-ubyte";
        const int MAX_TRAINING_IMAGES = 500;
        if (File.Exists(trainingImagesFileName))
        {
            using (FileStream stream = File.Open(trainingImagesFileName, FileMode.Open))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    int magic = ReadIntBigEndian(reader);
                    int numImages = ReadIntBigEndian(reader);
                    numImages = numImages > MAX_TRAINING_IMAGES ? MAX_TRAINING_IMAGES : numImages;
                    int numRows = ReadIntBigEndian(reader);
                    int numColumns = ReadIntBigEndian(reader);
                    Debug.Log($"{magic}: {numImages} x {numRows} x {numColumns}");
                    Assert.IsTrue(magic == 2051);
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
                    }
                }
            }
        }

        const string trainingLabelsFileName = @"Assets/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
        if (File.Exists(trainingLabelsFileName))
        {
            using (FileStream stream = File.Open(trainingLabelsFileName, FileMode.Open))
            {
                using (BinaryReader reader = new BinaryReader(stream))
                {
                    int magic = ReadIntBigEndian(reader);
                    int numLabels = ReadIntBigEndian(reader);
                    numLabels = numLabels > MAX_TRAINING_IMAGES ? MAX_TRAINING_IMAGES : numLabels;
                    Debug.Log($"{magic}: {numLabels}");
                    Assert.IsTrue(magic == 2049);
                    for (int i = 0; i < numLabels; i++)
                    {
                        trainingLabels.Add(reader.ReadByte());
                    }
                }
            }
        }
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

    private void LearnTrainingData()
    {
        int[] shuffledIndices = new int[trainingImages.Count];
        for (int i = 0; i < shuffledIndices.Length; i++)
        {
            shuffledIndices[i] = i;
        }

        for (int epoch = 1; epoch <= numEpochs; epoch++)
        {
            Shuffle(shuffledIndices);

            for (int j = 0; j < shuffledIndices.Length; j++)
            {
                int shuffledIndex = shuffledIndices[j];
                var t = trainingImages[shuffledIndex];
                //Graphics.CopyTexture(t, subjectImage.texture);
                float learningRateMultiplier = (float)(numEpochs - epoch) / (float)numEpochs;
                for (int y = 0; y < t.height; y++)
                {
                    for (int x = 0; x < t.width; x++)
                    {
                        brain.InLayer.Outputs[0, y, x] = t.GetPixel(x, y).r;
                    }
                }
                for (int label = 0; label < 10; label++)
                {
                    brain.OutLayer.Targets[label,0,0] = trainingLabels[shuffledIndex] == label ? 1f : 0f;
                }
                brain.Learn(learningRateMultiplier);
            }
        }
    }

    private void Reset()
    {
        subjectTexture.SetPixels32(new Color32[subjectTexture.width * subjectTexture.height]);

        // Pick a random image from the training set
        int r = UnityEngine.Random.Range(0, 100);
        Graphics.CopyTexture(trainingImages[r], subjectImage.texture);
        labelText.text = trainingLabels[r].ToString();

        subjectTexture.Apply();

        brain.Initialize(subjectTexture.width, subjectTexture.height, depth: 1);

        LearnTrainingData();
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
