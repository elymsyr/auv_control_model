#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

// Utility functions for data preprocessing and normalization
std::vector<float> normalizeData(const std::vector<float>& data, float min, float max);
std::vector<float> loadInputData(const std::string& filePath);

#endif // UTILS_H