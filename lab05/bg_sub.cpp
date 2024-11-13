/**
 * @file bg_sub.cpp
 * @brief Background subtraction tutorial with video saving functionality, using webcam input and sequential file saving.
 */

#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

const char* params
    = "{ help h         |           | Print usage }"
      "{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";

bool recording = false;
VideoWriter videoOriginal;
VideoWriter videoBg;
int videoCounter = 1; // Counter to save videos with sequential numbers

void startVideoRecording(int width, int height, double fps) {
    stringstream originalFileName, bgFileName;
    originalFileName << "original_video_" << videoCounter << ".avi";
    bgFileName << "background_subtracted_video_" << videoCounter << ".avi";

    videoOriginal.open(originalFileName.str(), VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
    videoBg.open(bgFileName.str(), VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), false);

    if (!videoOriginal.isOpened() || !videoBg.isOpened()) {
        cerr << "Error: Could not open video files for writing." << endl;
        recording = false;
    } else {
        recording = true;
        videoCounter++; // Increment counter for the next recording
    }
}

void stopVideoRecording() {
    if (recording) {
        videoOriginal.release();
        videoBg.release();
        recording = false;
    }
}

int main(int argc, char* argv[]) {
    CommandLineParser parser(argc, argv, params);
    parser.about("This program shows how to use background subtraction methods provided by OpenCV and saves the videos with and without background.");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();

    // Open the webcam
    VideoCapture capture(0); // 0 is the default webcam ID
    if (!capture.isOpened()) {
        cerr << "Unable to open the webcam." << endl;
        return 0;
    }

    Mat frame, fgMask;
    double fps = capture.get(CAP_PROP_FPS);
    int width = capture.get(CAP_PROP_FRAME_WIDTH);
    int height = capture.get(CAP_PROP_FRAME_HEIGHT);

    while (true) {
        capture >> frame;
        if (frame.empty())
            break;

        pBackSub->apply(frame, fgMask);

        rectangle(frame, cv::Point(10, 2), cv::Point(100, 20), cv::Scalar(255, 255, 255), -1);
        stringstream ss;
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        imshow("Frame", frame);
        imshow("FG Mask", fgMask);

        char key = (char)waitKey(30);
        if (key == 'q' || key == 27)
            break;

        if (key == 'k' || key == 'K') {
            if (!recording) {
                startVideoRecording(width, height, fps);
            }
        }

        if (key == 'h' || key == 'H') {
            stopVideoRecording();
        }

        if (recording) {
            videoOriginal.write(frame);
            videoBg.write(fgMask);
        }
    }

    stopVideoRecording();
    return 0;
}

