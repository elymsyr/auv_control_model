#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <torch/script.h>
#include <vector>
#include <string>

class ModelInference {
public:
    ModelInference();
    ~ModelInference();

    bool loadModel(const std::string& modelPath);
    std::vector<float> runInference(const std::vector<float>& inputData);
    void releaseResources();
    torch::Tensor preprocess_input(const std::vector<float>& input_data);

private:
    torch::jit::script::Module model;
    bool loaded;
};

#endif // MODEL_INFERENCE_H