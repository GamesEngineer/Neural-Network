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
    private NeuralNetwork brain;
    private float maxLoss;
    private float meanLoss;
    private Func<Vector2, float> testFunction;

    private static readonly Func<Vector2, float>[] testFunctions = { LinearFunc, ElipseFunc, HyperbolaFunc, SinePatchesFunc };

    private void Awake()
    {
        brain = GetComponent<NeuralNetwork>();

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
        brain.Initialize(numInputs: 2);
        trainingEpoch = 0;
        maxLoss = 0f;
        meanLoss = 0f;
        domainTexture.SetPixels32(new Color32[domainTexture.width * domainTexture.height]);
        graphTexture.SetPixels32(new Color32[graphTexture.width * graphTexture.height]);
    }

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
        brain.Targets[0] = testFunction(p);
        brain.SensoryInputs[0] = p.x;
        brain.SensoryInputs[1] = p.y;
        brain.Learn((float)(numTrainingEpochs - trainingEpoch) / (float)numTrainingEpochs);
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
        for (int h = 0; h < domainTexture.height; h++)
        {
            for (int w = 0; w < domainTexture.width; w++)
            {
                brain.SensoryInputs[0] = ((float)w / (float)domainTexture.width) * 4f - 2f;
                brain.SensoryInputs[1] = ((float)h / (float)domainTexture.height) * 4f - 2f;
                brain.Think();
                Color c;
                if (quantizePredictionsToggle.isOn)
                {
                    c = brain.Results[0] > 0.5f ? Color.yellow : Color.cyan;
                }
                else
                {
                    c = Color.Lerp(Color.cyan, Color.yellow, brain.Results[0]);
                }
                domainTexture.SetPixel(w, h, c);
            }
        }
    }

    private void DrawTrainingData()
    {
        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            int x = Mathf.FloorToInt((p.x + 2f) * (float)domainTexture.width / 4f);
            int y = Mathf.FloorToInt((p.y + 2f) * (float)domainTexture.height / 4f);
            if (x < 0 || x >= domainTexture.width || y < 0 || y >= domainTexture.height) continue;
            domainTexture.SetPixel(x, y, Color.Lerp(Color.blue, Color.red, testFunction(p)));
        }
    }

    private void DrawLossGraph()
    {
        if (trainingEpoch > numTrainingEpochs) return;
        int x = Mathf.FloorToInt(graphTexture.width * (float)(trainingEpoch-1) / (float)numTrainingEpochs);
        for (int h = 0; h < graphTexture.height; h++)
        {
            var c = graphTexture.GetPixel(x, h);
            c.a = 1f;
            graphTexture.SetPixel(x, h, c);
        }
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
