using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ClassifierNetworkTest : MonoBehaviour
{
    NeuralNetwork brain;
    public List<Vector2> points = new List<Vector2>();
    private float[] targets;
    public Image outputImage;
    private Texture2D texture;

    private void Awake()
    {
        brain = GetComponent<NeuralNetwork>();
        targets = new float[1]; // classify as a single value
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

    void Start()
    {
        int seed = (int)System.DateTime.Now.Ticks;
        UnityEngine.Random.InitState(seed);
        Debug.Log($"Seed: {seed}");

        Vector2[] shuffledPoints = points.ToArray();
        brain.Initialize(numInputs: 2);
        int iter;
        float maxLoss = 0f;
        for (iter = 0; iter < brain.numTrainingIterations; iter++)
        {
            //Shuffle(shuffledPoints);
            maxLoss = 0f;
            for (int i = 0; i < points.Count; i++)
            {
                LearnPoint(shuffledPoints[i]);
                maxLoss = Mathf.Max(brain.Loss, maxLoss);
            }
            if (iter < 100 || iter % 100 == 0)
            {
                Debug.Log($"Loss is {maxLoss} after {iter} iterations");
            }
            if (maxLoss < 1e-20f)
            {
                break; // good enough
            }
        }
        Debug.Log($"Final loss is {maxLoss} after {iter} iterations");

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

        // Draw points used for training
        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            int x = Mathf.FloorToInt((p.x + 2f) * (float)texture.width / 4f);
            int y = Mathf.FloorToInt((p.y + 2f) * (float)texture.height / 4f);
            if (x < 0 || x >= texture.width || y < 0 || y >= texture.height) continue;
            texture.SetPixel(x, y, TestFunc(p.x, p.y) > 0f ? Color.blue : Color.red);
        }

        texture.Apply();
    }

    private float TestFunc(float x, float y)
    {
        return 0.5f-x > y*y ? 1.0f : 0f;
    }

    private void LearnPoint(Vector2 p)
    {
        targets[0] = TestFunc(p.x, p.y);
        brain.SensoryInputs[0] = p.x;
        brain.SensoryInputs[1] = p.y;
        brain.Learn(targets, brain.learningRate);
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
