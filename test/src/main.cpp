// main.cpp
#include <iostream>
#include <torch/torch.h>
#include "model_inference.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    // Set default paths
    std::string modelPath = "../../pretrained/fossen_net_0/fossen_net_scripted.pt";
    std::string inputDataPath = "../input.txt";
    std::string scalerPath = "../../pretrained/fossen_net_0/scalers.json";

    if (torch::cuda::is_available()) {
        torch::globalContext().setBenchmarkCuDNN(true);
    }

    // Parse command-line arguments
    if (argc == 2) {
        // Only model path provided
        modelPath = argv[1];
    } else if (argc == 3) {
        // Both model and input paths provided
        modelPath = argv[1];
        inputDataPath = argv[2];
    } else if (argc > 3) {
        std::cerr << "Usage: " << argv[0] << " [model_path] [input_data_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../model.pt ../input.txt" << std::endl;
        return 1;
    }

    // Print configuration
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Model path: " << modelPath << std::endl;
    std::cout << "  Input data path: " << inputDataPath << std::endl;
    std::cout << "  CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "  CUDA device count: " << torch::cuda::device_count() << std::endl;

    try {
        ModelInference model;
        std::cout << "\nLoading model..." << std::endl;
        if (!model.loadScalers(scalerPath)) {
            std::cerr << "Failed to load scalers: " << scalerPath << std::endl;
            return 1;
        }
        if (!model.loadModel(modelPath)) {
            std::cerr << "Failed to load model: " << modelPath << std::endl;
            return 1;
        }
        

        // Load input data
        std::vector<float> inputData;
        try {
            std::cout << "Loading input data..." << std::endl;
            inputData = loadInputData(inputDataPath);
            std::cout << "  Loaded " << inputData.size() << " values" << std::endl;
            
            // Print first 10 values
            std::cout << "  First 10 values: ";
            for (int i = 0; i < 10 && i < inputData.size(); ++i) {
                std::cout << inputData[i] << " ";
            }
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Data loading error: " << e.what() << std::endl;
            return 1;
        }

        // Normalize data
        std::vector<float> normalizedInput;
        try {
            std::cout << "Normalizing data..." << std::endl;
            normalizedInput = normalizeData(inputData, 0.0f, 1.0f);
            std::cout << "  Normalized input size: " << normalizedInput.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Normalization error: " << e.what() << std::endl;
            return 1;
        }

        // Run inference
        try {
            // Warm-up runs
            if (torch::cuda::is_available()) {
                std::cout << "Running warm-up iterations..." << std::endl;
                for (int i = 0; i < 10; ++i) {
                    model.runInference(normalizedInput);
                }
            }

            std::cout << "Running inference..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<float> output = model.runInference(normalizedInput);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            
            std::cout << "  Inference time: " << duration.count() * 1000 << " ms" << std::endl;
            std::cout << "  Output size: " << output.size() << " values" << std::endl;
            
            std::cout << "\nModel output:" << std::endl;
            for (size_t i = 0; i < output.size(); ++i) {
                std::cout << "  Thruster " << i+1 << ": " << output[i] << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Inference failed: " << e.what() << std::endl;
            return 1;
        }

        model.releaseResources();
        std::cout << "\nInference completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}