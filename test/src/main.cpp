#include <iostream>
#include <torch/torch.h>
#include "model_inference.h"
#include "utils.h"

int main() {
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    
    ModelInference model;
    // Load TorchScript model
    if (!model.loadModel("../fossen_net_scripted.pt")) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    // Load and normalize input data
    std::vector<float> inputData = loadInputData("input.txt"); // input.txt should contain comma/space separated values
    std::vector<float> normalizedInput = normalizeData(inputData, 0.0f, 1.0f);

    // Run inference
    std::vector<float> output = model.runInference(normalizedInput);

    // Print output
    std::cout << "Model output:" << std::endl;
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    model.releaseResources();
    return 0;
}