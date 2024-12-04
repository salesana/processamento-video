#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace cv;
using namespace std;

vector<pair<int, string>> readFromCSV(const string &filename) {
    vector<pair<int, string>> data;
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        getline(file, line); // Skip header
        while (getline(file, line)) {
            size_t commaPos = line.find(',');
            int id = stoi(line.substr(0, commaPos));
            string name = line.substr(commaPos + 1);
            data.emplace_back(id, name);
        }
        file.close();
    }
    return data;
}

int main() {
    string csvFile = "id-names.csv";
    vector<pair<int, string>> idNames = readFromCSV(csvFile);

    Ptr<face::LBPHFaceRecognizer> recognizer = face::LBPHFaceRecognizer::create();
    recognizer->read("Classifiers/TrainedLBPH.yml");

    CascadeClassifier faceClassifier("Classifiers/haarface.xml");
    if (faceClassifier.empty()) {
        cerr << "Error: Could not load classifier cascade.\n";
        return -1;
    }

    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cerr << "Error: Could not open the camera.\n";
        return -1;
    }

    Mat frame, gray;

    while (true) {
        camera >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceClassifier.detectMultiScale(gray, faces, 1.1, 5);

        for (const auto &face : faces) {
            Mat faceRegion = gray(face);
            resize(faceRegion, faceRegion, Size(220, 220));

            int label;
            double confidence;
            recognizer->predict(faceRegion, label, confidence);

            string name = "Unknown";
            for (const auto &entry : idNames) {
                if (entry.first == label) {
                    name = entry.second;
                    break;
                }
            }

            rectangle(frame, face, Scalar(0, 0, 255), 2);
            putText(frame, name, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        imshow("Recognition", frame);
        if (waitKey(1) == 'q') break;
    }

    camera.release();
    destroyAllWindows();
    return 0;
}
