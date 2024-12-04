#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <ctime>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

void saveToCSV(const string &filename, const vector<pair<int, string>> &data) {
    ofstream file(filename);
    if (file.is_open()) {
        file << "id,name\n";
        for (const auto &entry : data) {
            file << entry.first << "," << entry.second << "\n";
        }
        file.close();
    }
}

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
    vector<pair<int, string>> idNames;

    // Load or initialize CSV
    if (fs::exists(csvFile)) {
        idNames = readFromCSV(csvFile);
    } else {
        saveToCSV(csvFile, idNames);
    }

    // Ensure "faces" directory exists
    if (!fs::exists("faces")) {
        fs::create_directory("faces");
    }

    cout << "Welcome!\n\nPlease put in your ID.\n";
    cout << "If this is your first time, choose a random ID between 1-10000\n";

    int id;
    string name;
    cout << "ID: ";
    cin >> id;

    auto it = find_if(idNames.begin(), idNames.end(),
                      [id](const pair<int, string> &entry) { return entry.first == id; });

    if (it != idNames.end()) {
        name = it->second;
        cout << "Welcome Back " << name << "!!\n";
    } else {
        cout << "Please Enter your name: ";
        cin.ignore();
        getline(cin, name);
        fs::create_directory("faces/" + to_string(id));
        idNames.emplace_back(id, name);
        saveToCSV(csvFile, idNames);
    }

    cout << "\nLet's capture! Press 's' to take a picture, and 'q' to quit.\n";

    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cerr << "Error: Could not open the camera.\n";
        return -1;
    }

    CascadeClassifier faceClassifier("Classifiers/haarface.xml");
    if (faceClassifier.empty()) {
        cerr << "Error: Could not load classifier cascade.\n";
        return -1;
    }

    int photosTaken = 0;
    Mat frame, gray;

    while (true) {
        camera >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceClassifier.detectMultiScale(gray, faces, 1.1, 5);

        for (const auto &face : faces) {
            rectangle(frame, face, Scalar(0, 0, 255), 2);
            Mat faceRegion = gray(face);

            if (waitKey(1) == 's' && mean(faceRegion)[0] > 50) {
                Mat resizedFace;
                resize(faceRegion, resizedFace, Size(220, 220));
                string filename = "faces/" + to_string(id) + "/face_" + to_string(photosTaken++) + ".jpg";
                imwrite(filename, resizedFace);
                cout << photosTaken << " -> Photos taken!\n";
            }
        }

        imshow("Face Capture", frame);
        if (waitKey(1) == 'q') break;
    }

    camera.release();
    destroyAllWindows();
    return 0;
}
