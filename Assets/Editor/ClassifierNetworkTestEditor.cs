using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(ClassifierNetworkTest))]
public class ClassifierNetworkTestEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var cnTest = target as ClassifierNetworkTest;
        if (cnTest && GUILayout.Button("Randomize Points"))
        {
            for (int i = 0; i < cnTest.points.Count; i++)
            {
                float x = UnityEngine.Random.Range(-2f, 2f);
                float y = UnityEngine.Random.Range(-2f, 2f);
                cnTest.points[i] = new Vector2(x, y);
            }
        }
    }
}
