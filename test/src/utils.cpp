#include "utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>

std::vector<float> normalizeData(const std::vector<float>& data, float min, float max) {
    std::vector<float> normalized(data.size());
    float dataMin = *std::min_element(data.begin(), data.end());
    float dataMax = *std::max_element(data.begin(), data.end());
    for (size_t i = 0; i < data.size(); ++i) {
        normalized[i] = (data[i] - dataMin) / (dataMax - dataMin) * (max - min) + min;
    }
    return normalized;
}

std::vector<float> loadInputData(const std::string& filePath) {
    std::vector<float> inputData;
    std::ifstream file(filePath);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            inputData.push_back(value);
            if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
        }
    }
    return inputData;
}