#define ENABLE_SOFT_EDGES
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

[RequireComponent(typeof(RawImage))]
public class DrawIntoTextureWithMouse : MonoBehaviour, IPointerDownHandler, IDragHandler
{
    private RawImage image;
    private Texture2D texture;

    private void Awake()
    {
        image = GetComponent<RawImage>();
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        DrawAtPointerPosition(eventData);
    }
    
    public void OnDrag(PointerEventData eventData)
    {
        DrawAtPointerPosition(eventData);
    }

    private void DrawAtPointerPosition(PointerEventData eventData)
    {
        if (texture == null)
        {
            texture = image.texture as Texture2D;
        }

        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
            image.rectTransform,
            eventData.position,
            eventData.pressEventCamera,
            out Vector2 localPos)) return;

        int x = Mathf.RoundToInt(texture.width * (localPos.x / image.rectTransform.rect.width + 0.5f));
        int y = Mathf.RoundToInt(texture.height * (localPos.y / image.rectTransform.rect.height + 0.5f));

        if (x < 0 || x >= texture.width || y < 0 || y >= texture.height) return;

        Color c = eventData.button == PointerEventData.InputButton.Left ? Color.white : Color.black;
        texture.SetPixel(x, y, c);
#if ENABLE_SOFT_EDGES
        texture.SetPixel(x - 1, y, Color.Lerp(texture.GetPixel(x - 1, y), c, 0.25f));
        texture.SetPixel(x + 1, y, Color.Lerp(texture.GetPixel(x + 1, y), c, 0.25f));
        texture.SetPixel(x, y - 1, Color.Lerp(texture.GetPixel(x, y - 1), c, 0.25f));
        texture.SetPixel(x, y + 1, Color.Lerp(texture.GetPixel(x, y + 1), c, 0.25f));
        texture.SetPixel(x - 1, y - 1, Color.Lerp(texture.GetPixel(x - 1, y - 1), c, 0.0625f));
        texture.SetPixel(x + 1, y - 1, Color.Lerp(texture.GetPixel(x + 1, y - 1), c, 0.0625f));
        texture.SetPixel(x - 1, y + 1, Color.Lerp(texture.GetPixel(x - 1, y + 1), c, 0.0625f));
        texture.SetPixel(x + 1, y + 1, Color.Lerp(texture.GetPixel(x + 1, y + 1), c, 0.0625f));
#endif
        texture.Apply();
    }
}
