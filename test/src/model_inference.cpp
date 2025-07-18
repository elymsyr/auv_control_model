// model_inference.cpp
#include "model_inference.h"
#include <iostream>
#include <torch/torch.h>
#include <stdexcept>

ModelInference::ModelInference() : loaded(false) {}

ModelInference::~ModelInference() {
    releaseResources();
}

bool ModelInference::loadModel(const std::string& modelPath) {
    try {
        // Check CUDA availability first
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available");
        }
        
        model = torch::jit::load(modelPath, torch::kCUDA);
        model.to(torch::kCUDA);  // Ensure model is on GPU
        loaded = true;
        std::cout << "Model loaded: " << modelPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        loaded = false;
    }
    return loaded;
}

std::vector<float> ModelInference::runInference(const std::vector<float>& inputData) {
    if (!loaded) {
        throw std::runtime_error("Model not loaded!");
    }
    
    // Validate input size
    if (inputData.size() != 501) {
        std::ostringstream err;
        err << "Invalid input size. Expected 501 elements, got " << inputData.size();
        throw std::invalid_argument(err.str());
    }
    
    try {
        // Create tensor with explicit shape
        torch::Tensor input = torch::tensor(inputData, torch::kFloat32)
                              .view({1, 501})  // Batch size 1, 501 features
                              .to(torch::kCUDA);
        
        // Run inference
        auto output = model.forward({input}).toTensor()
                          .to(torch::kCPU)  // Move back to CPU
                          .contiguous();
        
        // Convert to vector
        float* output_ptr = output.data_ptr<float>();
        return std::vector<float>(output_ptr, output_ptr + output.numel());
        
    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        throw;
    }
}

void ModelInference::releaseResources() {
    loaded = false;
}