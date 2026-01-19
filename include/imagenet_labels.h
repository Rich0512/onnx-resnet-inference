#pragma once

/**
 * @file imagenet_labels.h
 * @brief ImageNet class labels for common categories
 */

#include <string>
#include <unordered_map>

namespace imagenet {

// Common ImageNet class labels
const std::unordered_map<int, std::string> LABELS = {
    // Cats
    {281, "tabby cat"},
    {282, "tiger cat"},
    {285, "Egyptian cat"},
    {287, "lynx"},
    {288, "leopard"},
    
    // Dogs
    {151, "Chihuahua"},
    {207, "golden retriever"},
    {208, "Labrador retriever"},
    {230, "Shetland sheepdog"},
    {235, "German shepherd"},
    {243, "bull mastiff"},
    {254, "husky"},
    {263, "Pembroke Welsh corgi"},
    
    // Animals
    {1, "goldfish"},
    {65, "sea snake"},
    {99, "goose"},
    {130, "flamingo"},
    {291, "lion"},
    {292, "tiger"},
    {340, "zebra"},
    {386, "African elephant"},
    {388, "giant panda"},
    {409, "gibbon"},
    
    // Objects
    {417, "balloon"},
    {508, "computer keyboard"},
    {531, "digital watch"},
    {609, "jeep"},
    {717, "pickup truck"},
    {779, "school bus"},
    {852, "tennis ball"},
    {920, "traffic light"},
    {950, "orange"},
    {954, "banana"}
};

/**
 * Get class name by class ID
 * @param class_id ImageNet class ID (0-999)
 * @return Class name or "unknown (class X)"
 */
inline std::string get_label(int class_id) {
    auto it = LABELS.find(class_id);
    if (it != LABELS.end()) {
        return it->second;
    }
    return "unknown (class " + std::to_string(class_id) + ")";
}

} // namespace imagenet
