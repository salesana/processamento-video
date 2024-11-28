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

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open the camera" << endl;
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

    // Get the user's name
    cout << "Enter the name of the person to recognize: ";
    string person_name;
    cin >> person_name;

    string dataset_file = dataset_path + person_name + ".bin";
    if (!fs::exists(dataset_file)) {
        cout << "Error: The name '" << person_name << "' is not in the dataset." << endl;
        return -1;
    }

    // Load dataset for the specified user
    vector<vector<double>> face_data;
    ifstream file(dataset_file, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Could not open dataset file for " << person_name << endl;
        return -1;
    }

    while (!file.eof()) {
        vector<double> row(100 * 100); // Only features, no labels
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
        if (file.gcount() > 0) {
            face_data.push_back(row);
        }
    }
    file.close();

    cout << "Loaded " << face_data.size() << " samples for " << person_name << "." << endl;

    const double MIN_DISTANCE_THRESHOLD = 1000.0; // Threshold for validation

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
            Mat face_section = gray(roi);
            resize(face_section, face_section, Size(100, 100));

            // Flatten face_section
            vector<double> face_vector;
            face_vector.reserve(100 * 100);
            for (int i = 0; i < face_section.rows; ++i) {
                for (int j = 0; j < face_section.cols; ++j) {
                    face_vector.push_back(static_cast<double>(face_section.at<uchar>(i, j)));
                }
            }

            // Find the closest sample in the dataset
            double min_distance = DBL_MAX;
            for (const auto& sample : face_data) {
                double dist = distance(face_vector, sample);
                if (dist < min_distance) {
                    min_distance = dist;
                }
            }

            // Check against threshold and display result
            if (min_distance <= MIN_DISTANCE_THRESHOLD) {
                rectangle(frame, face, Scalar(0, 255, 0), 2);
                putText(frame, person_name + " Validado", Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                cout << "Recognized " << person_name << " with distance: " << min_distance << endl;
            } else {
                rectangle(frame, face, Scalar(0, 0, 255), 2);
                putText(frame, "Not Recognized", Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                cout << "Not Recognized (distance: " << min_distance << ")" << endl;
            }
        }

        imshow("Face Recognition", frame);

        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
