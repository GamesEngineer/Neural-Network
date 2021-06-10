using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ClassifierNetworkTest : MonoBehaviour
{
    public enum TestFunction { Linear, Elipse, Hyperbola, SinePatches }
    public TestFunction function;

    public List<Vector2> points = new List<Vector2>();
    public RawImage domainImage;
    public RawImage lossGraph;
    public Toggle quantizePredictionsToggle;
    public int numTrainingEpochs = 5000;

    private int trainingEpoch;
    private Vector2[] shuffledPoints;
    private Texture2D domainTexture;
    private Texture2D graphTexture;
    private ConvolutionalNeuralNetwork brain;
    private float maxLoss;
    private float meanLoss;
    private Func<Vector2, float> testFunction;

    private static readonly Func<Vector2, float>[] testFunctions = { LinearFunc, ElipseFunc, HyperbolaFunc, SinePatchesFunc };

    private void Awake()
    {
        brain = GetComponent<ConvolutionalNeuralNetwork>();

        domainTexture = new Texture2D(128, 128, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        domainImage.texture = domainTexture;

        graphTexture = new Texture2D(128, 64, TextureFormat.RGB24, mipChain: false)
        {
            filterMode = FilterMode.Point,
            wrapMode = TextureWrapMode.Clamp
        };
        lossGraph.texture = graphTexture;

        testFunction = testFunctions[(int)function];
    }

    private void Start()
    {
        int seed = (int)DateTime.Now.Ticks;
        UnityEngine.Random.InitState(seed);
        Debug.Log($"Seed: {seed}");
        Reset();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            Reset();
        }

        UpdateDomainControls();

        if (trainingEpoch++ <= numTrainingEpochs)
        {
            ProcessEpoch();
        }

        DrawPredictions();
        DrawTrainingData();
        domainTexture.Apply();

        DrawLossGraph();
        graphTexture.Apply();
    }

    private void Reset()
    {
        shuffledPoints = points.ToArray();
        brain.Initialize(1, 1, 2, Neuron.ActivationType.SoftMax);
        trainingEpoch = 0;
        maxLoss = 0f;
        meanLoss = 0f;
        domainTexture.SetPixels32(new Color32[domainTexture.width * domainTexture.height]);
        graphTexture.SetPixels32(new Color32[graphTexture.width * graphTexture.height]);
    }

    #region Domain Control

    private float domainSize = 4f;
    private Vector2 domainOffset;

    private void ResetDomain()
    {
        domainSize = 4f;
        domainOffset = Vector2.zero;
    }
    
    private void UpdateDomainControls()
    {
        if (Input.GetKeyDown(KeyCode.Return))
        {
            ResetDomain();
        }

        if (Input.GetMouseButton(1))
        {
            domainOffset += (Input.GetAxis("Mouse X") * Vector2.right + Input.GetAxis("Mouse Y") * Vector2.up) * domainSize / domainTexture.width;
        }

        if (Input.GetMouseButtonDown(2))
        {
            quantizePredictionsToggle.isOn = !quantizePredictionsToggle.isOn;
        }

        domainSize += Input.mouseScrollDelta.y * -0.1f * domainSize;
    }

    #endregion

    private static float LinearFunc(Vector2 p)
    {
        return (0.333f - p.x) > 0.5f * p.y ? 1f : 0f;
    }

    private static float ElipseFunc(Vector2 p)
    {
        return 0.5f*p.x*p.x + 0.333f*p.y*p.y < 1f ? 1f : 0f;
    }

    private static float HyperbolaFunc(Vector2 p)
    {
        return (p.x * (p.x - 1f) - p.y * (0.333f * p.y + 0.2f)) > 0f ? 1f : 0f;
    }
    
    private static float SinePatchesFunc(Vector2 p)
    {
        return Mathf.Sin(3f * p.x) * Mathf.Sin(2f * p.y) > 0.2f ? 1f : 0f;
    }

    private void LearnPoint(Vector2 p)
    {
        float q = testFunction(p);
        brain.OutLayer.Targets[0, 0, 0] = q;
        brain.OutLayer.Targets[1, 0, 0] = 1 - q;
        brain.InLayer.Outputs[0, 0, 0] = p.x;
        brain.InLayer.Outputs[1, 0, 0] = p.y;
        float learningRateMultiplier = (float)(numTrainingEpochs - trainingEpoch) / (float)numTrainingEpochs;
        brain.Learn(learningRateMultiplier);
    }

    private void ProcessEpoch()
    {
        Shuffle(shuffledPoints);
        maxLoss = 0f;
        meanLoss = 0f;
        for (int i = 0; i < points.Count; i++)
        {
            LearnPoint(shuffledPoints[i]);
            meanLoss += brain.Loss;
            maxLoss = Mathf.Max(brain.Loss, maxLoss);
        }
        meanLoss /= points.Count;
        if (trainingEpoch < 100 || trainingEpoch % 100 == 0)
        {
            Debug.Log($"Loss is {meanLoss} | {maxLoss} after {trainingEpoch} iterations");
        }
    }

    private void DrawPredictions()
    {
        for (int y = 0; y < domainTexture.height; y++)
        {
            for (int x = 0; x < domainTexture.width; x++)
            {
                brain.InLayer.Outputs[0,0,0] = ((float)x / (float)domainTexture.width) * domainSize - domainSize / 2f - domainOffset.x;
                brain.InLayer.Outputs[1,0,0] = ((float)y / (float)domainTexture.height) * domainSize - domainSize / 2f - domainOffset.y;
                brain.Think();
                Color c;
                if (quantizePredictionsToggle.isOn)
                {
                    c = brain.OutLayer.WinnerIndex == 1 ? Color.yellow : Color.cyan;
                }
                else
                {
                    c = Color.Lerp(Color.cyan, Color.yellow, brain.OutLayer.Outputs[0, 0, 0]);
                }
                domainTexture.SetPixel(x, y, c);
            }
        }
    }

    private void DrawTrainingData()
    {
        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            int x = Mathf.FloorToInt((p.x + domainSize/2f + domainOffset.x) * (float)domainTexture.width / domainSize);
            int y = Mathf.FloorToInt((p.y + domainSize/2f + domainOffset.y) * (float)domainTexture.height / domainSize);
            if (x < 0 || x >= domainTexture.width || y < 0 || y >= domainTexture.height) continue;
            domainTexture.SetPixel(x, y, Color.Lerp(Color.blue, Color.red, testFunction(p)));
        }
    }

    private void DrawLossGraph()
    {
        if (trainingEpoch > numTrainingEpochs) return;
        int x = Mathf.FloorToInt(graphTexture.width * (float)(trainingEpoch-1) / (float)numTrainingEpochs);
        int y = Mathf.FloorToInt(Mathf.Sqrt(Mathf.Clamp01(maxLoss)) * (graphTexture.height - 1));
        graphTexture.SetPixel(x, y, new Color(1f, 0.5f, 0f));
        y = Mathf.FloorToInt(Mathf.Sqrt(Mathf.Clamp01(meanLoss)) * (graphTexture.height - 1));
        graphTexture.SetPixel(x, y, Color.white);
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
