using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ClassifierNetworkTest : MonoBehaviour
{
    private static float TestFunc(float x, float y)
    {
#if false
        return (x * (x - 1f) - y * (0.333f * y + 0.2f)) > 0f ? 1f : 0f;
#else
        return Mathf.Sin(3f*x) * Mathf.Sin(2f*y) > 0.2f ? 1f : 0f;
#endif
    }

    public List<Vector2> points = new List<Vector2>();
    public Image outputImage;
    public Image maxLossBarImage;
    public Image meanLossBarImage;
    public RawImage lossGraph;
    public bool quantizePreditictions = true;
    public int numTrainingEpochs = 2000;
    private int trainingEpoch;
    private Vector2[] shuffledPoints;
    private Texture2D texture;
    private Texture2D graphTexture;
    private NeuralNetwork brain;
    private float maxLoss;
    private float meanLoss;

    private void Awake()
    {
        brain = GetComponent<NeuralNetwork>();
        texture = new Texture2D(128, 128);
        texture.filterMode = FilterMode.Point;
        texture.alphaIsTransparency = false;
        for (int h = 0; h < texture.height; h++)
        {
            for (int w = 0; w < texture.width; w++)
            {
                texture.SetPixel(w, h, Color.black);
            }
        }
        texture.Apply();
        outputImage.sprite = Sprite.Create(texture, new Rect(0,0, texture.width, texture.height), Vector2.one * 0.5f);

        graphTexture = new Texture2D(64, 32);
        lossGraph.texture = graphTexture;
    }

    void Start()
    {
        int seed = (int)System.DateTime.Now.Ticks;
        UnityEngine.Random.InitState(seed);
        Debug.Log($"Seed: {seed}");
        Reset();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Reset();
        }

        if (trainingEpoch++ <= numTrainingEpochs)
        {
            ProcessEpoch();
        }

        DrawPredictions();
        DrawTrainingData();
        texture.Apply();
        maxLossBarImage.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, 1000f * Mathf.Sqrt(maxLoss));
        meanLossBarImage.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, 1000f * Mathf.Sqrt(meanLoss));
        DrawLossGraph();
    }

    private void Reset()
    {
        shuffledPoints = points.ToArray();
        brain.Initialize(numInputs: 2);
        trainingEpoch = 0;
        maxLoss = 0f;
        meanLoss = 0f;
        graphTexture.SetPixels(new Color[graphTexture.width * graphTexture.height]);
    }

    private void LearnPoint(Vector2 p)
    {
        brain.Targets[0] = TestFunc(p.x, p.y);
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
        for (int h = 0; h < texture.height; h++)
        {
            for (int w = 0; w < texture.width; w++)
            {
                brain.SensoryInputs[0] = ((float)w / (float)texture.width) * 4f - 2f;
                brain.SensoryInputs[1] = ((float)h / (float)texture.height) * 4f - 2f;
                brain.Think();
                Color c;
                if (quantizePreditictions)
                {
                    c = brain.Results[0] > 0.5f ? Color.yellow : Color.cyan;
                }
                else
                {
                    c = Color.Lerp(Color.cyan, Color.yellow, brain.Results[0]);
                }
                texture.SetPixel(w, h, c);
            }
        }
    }

    private void DrawTrainingData()
    {
        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            int x = Mathf.FloorToInt((p.x + 2f) * (float)texture.width / 4f);
            int y = Mathf.FloorToInt((p.y + 2f) * (float)texture.height / 4f);
            if (x < 0 || x >= texture.width || y < 0 || y >= texture.height) continue;
            texture.SetPixel(x, y, Color.Lerp(Color.blue, Color.red, TestFunc(p.x, p.y)));
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
        graphTexture.SetPixel(x, y, Color.yellow);
        y = Mathf.FloorToInt(Mathf.Sqrt(Mathf.Clamp01(meanLoss)) * (graphTexture.height - 1));
        graphTexture.SetPixel(x, y, Color.white);

        graphTexture.Apply();
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
