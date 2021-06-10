using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ImageConvolution : MonoBehaviour
{
    public RawImage sourceImage;
    public RawImage convolvedImage;

    private Texture2D sourceTexture;
    private Texture2D convolvedTexture;
    private readonly float[,] kernel = new float[3, 3];

    public TMP_Dropdown filterSelector;
    public Image kernelPanel;
    private readonly TextMeshProUGUI[,] kernelText = new TextMeshProUGUI[3,3];

    void Start()
    {
        sourceTexture = sourceImage.texture as Texture2D;
        convolvedTexture = new Texture2D(sourceTexture.width, sourceTexture.height, TextureFormat.RGB24, false);
        convolvedImage.texture = convolvedTexture;
        convolvedImage.SetNativeSize();

        #region Kernel Panel
        for (int b = 0; b < 3; b++)
        {
            for (int a = 0; a < 3; a++)
            {
                var child = kernelPanel.transform.GetChild(a + b * 3);
                kernelText[b, a] = child.GetComponent<TextMeshProUGUI>();
            }
        }
        #endregion

        UpdateKernel();
        Convolve();
    }

    private void UpdateKernel()
    {
        for (int b = 0; b < 3; b++)
        {
            for (int a = 0; a < 3; a++)
            {
                int filterIndex = filterSelector.value;
                kernel[b, a] = filters[filterIndex, b, a];
                kernelText[b, a].text = kernel[b, a].ToString();
            }
        }
    }

    /// <summary>
    /// This is a naive implementation of discrete convolution in 2D with a 3x3 kernel
    /// </summary>
    private void Convolve()
    {
        for (int y = 0; y < sourceTexture.height; y++)
        {
            for (int x = 0; x < sourceTexture.width; x++)
            {
                Color dst = Color.black;
                for (int b = -1; b <= 1; b++)
                {
                    for (int a = -1; a <= 1; a++)
                    {
                        Color src = sourceTexture.GetPixel(x + a, y - b); // NOTE: -b, because texture coords start at BOTTOM left
                        dst += src * kernel[b+1, a+1];
                    }
                }
                convolvedTexture.SetPixel(x, y, dst);
            }
        }
        convolvedTexture.Apply();
    }

    private float[,,] filters = new float[10, 3, 3]
    {
        { // Identity
            {  0f,  0f,  0f },
            {  0f,  1f,  0f },
            {  0f,  0f,  0f },
        },
        { // Outline
            { -1f, -1f, -1f },
            { -1f,  8f, -1f },
            { -1f, -1f, -1f },
        },
        { // Sharpen
            {  0f, -1f,  0f },
            { -1f,  5f, -1f },
            {  0f, -1f,  0f },
        },
        { // Blur
            { 0.0625f, 0.1250f, 0.0625f },
            { 0.1250f, 0.2500f, 0.1250f },
            { 0.0625f, 0.1250f, 0.0625f },
        },
        { // Emboss
            { -2f, -1f,  0f },
            { -1f,  1f,  1f },
            {  0f,  1f,  2f },
        },
        { // BottomSobel
            { -1f, -2f, -1f },
            {  0f,  0f,  0f },
            {  1f,  2f,  1f },
        },
        { // TopSobel
            {  1f,  2f,  1f },
            {  0f,  0f,  0f },
            { -1f, -2f, -1f },
        },
        { // LeftSobel
            {  1f,  0f, -1f },
            {  2f,  0f, -2f },
            {  1f,  0f, -1f },
        },
        { // RightSobel
            { -1f,  0f,  1f },
            { -2f,  0f,  2f },
            { -1f,  0f,  1f },
        },
        { // Outline2 (Laplacian)
            { 1f,   2f, 1f },
            { 2f, -12f, 2f },
            { 1f,   2f, 1f },
        },
    };

    public void OnFilterSelected()
    {
        UpdateKernel();
        Convolve();
    }
}
