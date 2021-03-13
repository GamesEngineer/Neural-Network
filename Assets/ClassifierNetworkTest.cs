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

    // Start is called before the first frame update
    void Start()
    {
        int seed = (int)System.DateTime.Now.Ticks;
        UnityEngine.Random.InitState(seed);
        Debug.Log($"Seed: {seed}");

        brain.Initialize(2);
        for (int i = 0; i < points.Count; i++)
        {
            LearnPoint(i);
        }
    }

    private void LearnPoint(int i)
    {
        var p = points[i];
        targets[0] = -p.x > p.y ? 1f : -1f;
        brain.SensoryInputs[0] = p.x;
        brain.SensoryInputs[1] = p.y;
        brain.Train(targets, brain.learningRate);
        brain.Think();
    }

    // Update is called once per frame
    void Update()
    {
        for (int h = 0; h < texture.height; h++)
        {
            for (int w = 0; w < texture.width; w++)
            {
                brain.SensoryInputs[0] = ((float)w / (float)texture.width) * 4f - 2f;
                brain.SensoryInputs[1] = ((float)h / (float)texture.height) * 4f - 2f;
                brain.Think();
                texture.SetPixel(w, h, brain.Results[0] > 0f ? Color.cyan : Color.yellow);
            }
        }

        for (int i = 0; i < points.Count; i++)
        {
            var p = points[i];
            int x = Mathf.FloorToInt((p.x + 2f) * (float)texture.width / 4f);
            int y = Mathf.FloorToInt((p.y + 2f) * (float)texture.height / 4f);
            if (x < 0 || x >= texture.width || y < 0 || y >= texture.height) continue;
            texture.SetPixel(x, y, brain.Results[0] > 0f ? Color.blue : Color.white);
        }

        texture.Apply();
    }
}
