// utils.cpp
#include "utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>

std::vector<float> normalizeData(const std::vector<float>& data, float min, float max) {
    if(data.empty()) {
        throw std::runtime_error("Cannot normalize empty data vector");
    }
    
    std::vector<float> normalized(data.size());
    float dataMin = *std::min_element(data.begin(), data.end());
    float dataMax = *std::max_element(data.begin(), data.end());
    
    // Handle constant data
    if(dataMax - dataMin < 1e-7) {
        std::fill(normalized.begin(), normalized.end(), (min + max)/2.0f);
        return normalized;
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        normalized[i] = (data[i] - dataMin) / (dataMax - dataMin) * (max - min) + min;
    }
    return normalized;
}

std::vector<float> loadInputData(const std::string& filePath) {
    std::vector<float> inputData;
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open input file: " + filePath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Remove commas and brackets
        line.erase(std::remove(line.begin(), line.end(), ','), line.end());
        line.erase(std::remove(line.begin(), line.end(), '['), line.end());
        line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
        
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            inputData.push_back(value);
        }
    }
    
    if (inputData.empty()) {
        throw std::runtime_error("No data loaded from input file");
    }
    
    return inputData;
}