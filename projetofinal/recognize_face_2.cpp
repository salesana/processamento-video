#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>

// Function to load ID and names from a CSV file
std::map<int, std::string> loadIdNames(const std::string& csvFilePath) {
    std::map<int, std::string> idNames;
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << csvFilePath << std::endl;
        return idNames;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string idStr, name;
        if (std::getline(ss, idStr, ',') && std::getline(ss, name)) {
            int id = std::stoi(idStr);
            idNames[id] = name;
        }
    }
    file.close();
    return idNames;
}

int main() {
    // Load ID names from CSV
    std::map<int, std::string> idNames = loadIdNames("id-names.csv");

    // Load Haar Cascade Classifier
    cv::CascadeClassifier faceClassifier("Classifiers/haarface.xml");
    if (faceClassifier.empty()) {
        std::cerr << "Error loading Haar Cascade Classifier." << std::endl;
        return -1;
    }

    // Load the trained LBPH recognizer
    cv::Ptr<cv::face::LBPHFaceRecognizer> lbph = cv::face::LBPHFaceRecognizer::create();
    lbph->setThreshold(500.0);
    lbph->read("Classifiers/TrainedLBPH.yml");

    // Open the webcam
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "Error opening camera." << std::endl;
        return -1;
    }

    cv::Mat img, grey;
    while (cv::waitKey(1) != 'q') {
        camera >> img;
        if (img.empty()) continue;

        // Convert to grayscale
        cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);

        // Detect faces
        std::vector<cv::Rect> faces;
        faceClassifier.detectMultiScale(grey, faces, 1.1, 4);

        for (const auto& face : faces) {
            cv::Mat faceRegion = grey(face);
            cv::resize(faceRegion, faceRegion, cv::Size(220, 220));

            int label = -1;
            double trust = 0.0;
            lbph->predict(faceRegion, label, trust);

            auto it = idNames.find(label);
            if (it != idNames.end()) {
                const std::string& name = it->second;
                // Draw rectangle and name
                cv::rectangle(img, face, cv::Scalar(0, 0, 255), 2);
                cv::putText(img, name, cv::Point(face.x, face.y + face.height + 30),
                            cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 0, 255));
            }
        }

        // Display the video feed with detections
        cv::imshow("Recognize", img);
    }

    // Release the camera and destroy windows
    camera.release();
    cv::destroyAllWindows();
    
    return 0;
}
