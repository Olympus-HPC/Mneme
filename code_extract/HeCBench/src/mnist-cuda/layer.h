#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
  public:
    int M, N, O;

    float *output;
    float *preact;

    float *bias;
    float *weight;

    float *d_output;
    float *d_preact;
    float *d_weight;

    Layer(int M, int N, int O);

    ~Layer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
};
