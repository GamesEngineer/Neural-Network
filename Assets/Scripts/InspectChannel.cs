using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using TMPro;

public class InspectChannel : MonoBehaviour, IPointerDownHandler
{
    public ImageClassifierNetworkTest icnTest;
    public TextMeshProUGUI valueText;
    private RawImage image;

    private void Awake()
    {
        image = GetComponent<RawImage>();
        valueText.enabled = false;
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        var texture = image.texture as Texture2D;

        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
            image.rectTransform,
            eventData.position,
            eventData.pressEventCamera,
            out Vector2 localPos))
        {
            valueText.enabled = false;
        }

        int x = Mathf.FloorToInt(texture.width * (localPos.x / image.rectTransform.rect.width + 0.5f));
        int y = Mathf.FloorToInt(texture.height * (localPos.y / image.rectTransform.rect.height + 0.5f));

        if (x < 0 || x >= texture.width || y < 0 || y >= texture.height)
        {
            valueText.enabled = false;
        }

        float value = icnTest.GetDebugLayerOutput(x, y);
        valueText.text = value.ToString();
        valueText.enabled = true;
    }
}
