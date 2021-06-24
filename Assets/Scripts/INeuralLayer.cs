// Copyright 2021 Game-U Enterprises LLC

public interface INeuralLayer
{
    INeuralLayer InLayer { get; }
    INeuralLayer OutLayer { get; set; }
    int Width { get; }
    int Height { get; }
    int Depth { get; }
    float[/*depth*/,/*height*/,/*width*/] Outputs { get; }
    float[/*depth*/,/*height*/,/*width*/] Feedback { get; }
    Neuron.ActivationType Activation { get; }
    void Activate(bool withDropout);
    void BackPropagate();
    void UpdateWeightsAndBiases(float learningRate);
    float CalculateWeightedFeedback(int inZ, int inY, int inX);
    float[] ChannelMin { get; }
    float[] ChannelMax { get; }
}
