#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

namespace fs = std::filesystem;

std::vector<int> labels;
std::vector<cv::Mat> faces;

// Function to load training data
void createTrain(const std::string& datasetPath) {
    for (const auto& idDir : fs::directory_iterator(datasetPath)) {
        if (!fs::is_directory(idDir)) continue;

        std::string id = idDir.path().filename().string();
        for (const auto& imgFile : fs::directory_iterator(idDir.path())) {
            try {
                // Read the image and convert to grayscale
                cv::Mat face = cv::imread(imgFile.path().string());
                cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);

                // Append the face and label
                faces.push_back(face);
                labels.push_back(std::stoi(id));
            } catch (...) {
                std::cerr << "Error processing: " << imgFile.path() << std::endl;
            }
        }
    }
}

int main() {
    // Create the LBPH face recognizer
    cv::Ptr<cv::face::LBPHFaceRecognizer> lbph = cv::face::LBPHFaceRecognizer::create();
    lbph->setThreshold(500.0); // Set threshold

    // Load training data
    std::cout << "Loading training data..." << std::endl;
    createTrain("faces");

    // Train the recognizer
    if (faces.empty() || labels.empty()) {
        std::cerr << "No training data found!" << std::endl;
        return -1;
    }

    std::cout << "Training Started" << std::endl;
    lbph->train(faces, labels);
    lbph->save("Classifiers/TrainedLBPH.yml");
    std::cout << "Training Complete!" << std::endl;

    return 0;
}
