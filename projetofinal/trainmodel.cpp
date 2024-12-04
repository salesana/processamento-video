#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

vector<pair<Mat, int>> loadTrainingData(const string &basePath) {
    vector<pair<Mat, int>> data;

    for (const auto &entry : fs::directory_iterator(basePath)) {
        int id = stoi(entry.path().filename().string());
        for (const auto &imgFile : fs::directory_iterator(entry.path())) {
            Mat img = imread(imgFile.path().string(), IMREAD_GRAYSCALE);
            if (!img.empty()) {
                data.emplace_back(img, id);
            }
        }
    }

    return data;
}

int main() {
    string facesPath = "faces";
    if (!fs::exists(facesPath)) {
        cerr << "Error: Faces directory not found.\n";
        return -1;
    }

    Ptr<face::LBPHFaceRecognizer> recognizer = face::LBPHFaceRecognizer::create();
    vector<pair<Mat, int>> trainingData = loadTrainingData(facesPath);

    vector<Mat> images;
    vector<int> labels;

    for (const auto &data : trainingData) {
        images.push_back(data.first);
        labels.push_back(data.second);
    }

    cout << "Training started...\n";
    recognizer->train(images, labels);
    recognizer->save("Classifiers/TrainedLBPH.yml");
    cout << "Training completed!\n";

    return 0;
}
