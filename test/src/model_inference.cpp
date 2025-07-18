#include "model_inference.h"
#include <iostream>
#include <torch/script.h>

ModelInference::ModelInference() : loaded(false) {}

ModelInference::~ModelInference() {
    releaseResources();
}

bool ModelInference::loadModel(const std::string& modelPath) {
    try {
        model = torch::jit::load(modelPath, torch::kCUDA);
        loaded = true;
        std::cout << "Model loaded: " << modelPath << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        loaded = false;
    }
    return loaded;
}

std::vector<float> ModelInference::runInference(const std::vector<float>& inputData) {
    if (!loaded) {
        std::cerr << "Model not loaded!" << std::endl;
        return {};
    }
    // Convert input to tensor and move to GPU
    torch::Tensor input = torch::from_blob((float*)inputData.data(), {(int)inputData.size()}, torch::kFloat32).unsqueeze(0).to(torch::kCUDA);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    torch::Tensor output = model.forward(inputs).toTensor().to(torch::kCPU);

    // Convert output tensor to std::vector<float>
    std::vector<float> result(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    return result;
}

void ModelInference::releaseResources() {
    loaded = false;
}