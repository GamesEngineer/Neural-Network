using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ClassifierNetworkTest : MonoBehaviour
{
    NeuralNetwork brain;
    public List<Vector2> points = new List<Vector2>();
    public Image outputImage;
    private Texture2D texture;

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
    }

    private int trainingIteration;
    Vector2[] shuffledPoints;

    void Start()
    {
        int seed = (int)System.DateTime.Now.Ticks;
        UnityEngine.Random.InitState(seed);
        Debug.Log($"Seed: {seed}");

        shuffledPoints = points.ToArray();
        brain.Initialize(numInputs: 2);
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            shuffledPoints = points.ToArray();
            brain.Initialize(numInputs: 2);
            trainingIteration = 0;
        }

        trainingIteration++;
        float maxLoss = 0f;
        if (trainingIteration < brain.numTrainingIterations)
        {
            Shuffle(shuffledPoints);
            maxLoss = 0f;
            for (int i = 0; i < points.Count; i++)
            {
                LearnPoint(shuffledPoints[i]);
                maxLoss = Mathf.Max(brain.Loss, maxLoss);
            }
            if (trainingIteration < 100 || trainingIteration % 100 == 0)
            {
                Debug.Log($"Loss is {maxLoss} after {trainingIteration} iterations");
            }
        }

        DrawPredictions();
        DrawTrainingData();
        texture.Apply();
    }
    private void DrawPredictions()
    {
        // Draw map of learned predictions
        for (int h = 0; h < texture.height; h++)
        {
            for (int w = 0; w < texture.width; w++)
            {
                brain.SensoryInputs[0] = ((float)w / (float)texture.width) * 4f - 2f;
                brain.SensoryInputs[1] = ((float)h / (float)texture.height) * 4f - 2f;
                brain.Think();
                texture.SetPixel(w, h, Color.Lerp(Color.cyan, Color.yellow, brain.Results[0]));
            }
        }
    }

    private void DrawTrainingData()
    {
        // Draw points used for training
        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            int x = Mathf.FloorToInt((p.x + 2f) * (float)texture.width / 4f);
            int y = Mathf.FloorToInt((p.y + 2f) * (float)texture.height / 4f);
            if (x < 0 || x >= texture.width || y < 0 || y >= texture.height) continue;
            texture.SetPixel(x, y, TestFunc(p.x, p.y) > 0f ? Color.red : Color.blue);
        }
    }

#if false
    private float TestFunc(float x, float y)
    {
        return 0.5f-x > y*y ? 1.0f : 0f;
    }
#else
    private float TestFunc(float x, float y)
    {
        return (x * (x - 1f) - y * y) > 0f ? 1f : 0f;
    }
#endif

    private void LearnPoint(Vector2 p)
    {
        brain.Targets[0] = TestFunc(p.x, p.y);
        brain.SensoryInputs[0] = p.x;
        brain.SensoryInputs[1] = p.y;
        brain.Think();
        brain.Learn();
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
