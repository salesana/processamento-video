#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

// Euclidean distance function
double distance(const vector<double>& v1, const vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += pow(v1[i] - v2[i], 2);
    }
    return sqrt(sum);
}

// Normalize feature vector
void normalize(vector<double>& vec) {
    double max_val = *max_element(vec.begin(), vec.end());
    if (max_val > 0) {
        for (auto& val : vec) {
            val /= max_val;
        }
    }
}

// KNN function
int knn(const vector<vector<double>>& train, const vector<double>& test, int k = 5) {
    vector<pair<double, int>> dist;

    for (const auto& row : train) {
        vector<double> feature_vector(row.begin(), row.end() - 1);
        int label = static_cast<int>(row.back());
        double d = distance(feature_vector, test);
        dist.push_back({d, label});
    }

    // Sort based on distance
    sort(dist.begin(), dist.end(), [](const pair<double, int>& a, const pair<double, int>& b) {
        return a.first < b.first;
    });

    // Select top k
    vector<int> labels;
    for (int i = 0; i < k; ++i) {
        labels.push_back(dist[i].second);
    }

    // Determine the most frequent label
    map<int, int> freq;
    for (int label : labels) {
        freq[label]++;
    }

    int max_count = 0, best_label = -1;
    for (const auto& [label, count] : freq) {
        if (count > max_count) {
            max_count = count;
            best_label = label;
        }
    }

    return best_label;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }

    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
        cerr << "Error: Could not load Haar Cascade file" << endl;
        return -1;
    }

    string dataset_path = "./face_dataset/";
    if (!fs::exists(dataset_path)) {
        cerr << "Error: Dataset path does not exist" << endl;
        return -1;
    }

    vector<vector<double>> face_data;
    map<int, string> names;
    int class_id = 0;

    // Load dataset
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.path().extension() == ".bin") {
            names[class_id] = entry.path().stem().string();

            // Load binary data
            ifstream file(entry.path(), ios::binary);
            if (!file.is_open()) {
                cerr << "Error: Could not open file " << entry.path() << endl;
                continue;
            }

            while (!file.eof()) {
                vector<double> row(100 * 100 + 1); // Features + label
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
                if (file.gcount() > 0) {
                    row.back() = class_id; // Assign current class_id to label
                    face_data.push_back(row);
                }
            }
            file.close();

            class_id++;
        }
    }

    cout << "Loaded " << face_data.size() << " face samples." << endl;

    Mat saved_frame;

    // Main loop
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Could not capture frame" << endl;
            continue;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);

        for (const auto& face : faces) {
            int x = face.x, y = face.y, w = face.width, h = face.height;

            // Extract face ROI
            int offset = 5;
            Rect roi(
                max(0, x - offset),
                max(0, y - offset),
                min(frame.cols - x, w + 2 * offset),
                min(frame.rows - y, h + 2 * offset)
            );
            Mat face_section = frame(roi);
            resize(face_section, face_section, Size(100, 100));

            // Draw bounding box
            rectangle(frame, face, Scalar(255, 255, 255), 2);
        }

        imshow("Faces", frame);

        int key = waitKey(1);

        // If 's' is pressed, save the last frame and predict
        if (key == 's') {
            saved_frame = frame.clone();
            if (!faces.empty()) {
                Rect face = faces[0]; // Use the first detected face
                Mat face_section = frame(face);
                resize(face_section, face_section, Size(100, 100));

                // Flatten face_section
                vector<double> face_vector;
                face_vector.reserve(100 * 100);
                for (int i = 0; i < face_section.rows; ++i) {
                    for (int j = 0; j < face_section.cols; ++j) {
                        face_vector.push_back(static_cast<double>(face_section.at<uchar>(i, j)));
                    }
                }

                // Normalize the feature vector
                normalize(face_vector);

                // Predict using KNN
                int predicted_label = knn(face_data, face_vector);
                string name = names[predicted_label];
                cout << "Predicted name: " << name << endl;

                // Save the frame to disk for reference
                imwrite("saved_frame.jpg", saved_frame);
                cout << "Saved frame as 'saved_frame.jpg'" << endl;
            } else {
                cout << "No face detected in the current frame." << endl;
            }
        }

        if (key == 'q') break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
