#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <ctime>

using namespace cv;

/** Global Variables */
const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

//! [low]
static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
//! [low]

//! [high]
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
//! [high]

static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}

static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_detection_name, high_S);
}

static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}

static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_detection_name, high_V);
}

// Function to save the current frames from both windows
void save_frames(const Mat& frame, const Mat& frame_threshold)
{
    // Create a unique filename based on the current time
    time_t now = time(0);
    tm *ltm = localtime(&now);

    std::string timestamp = std::to_string(1900 + ltm->tm_year) + "_" +
                            std::to_string(1 + ltm->tm_mon) + "_" +
                            std::to_string(ltm->tm_mday) + "_" +
                            std::to_string(ltm->tm_hour) + "_" +
                            std::to_string(ltm->tm_min) + "_" +
                            std::to_string(ltm->tm_sec);

    // Save both the raw video frame and the detection frame
    std::string capture_filename = "video_capture_" + timestamp + ".png";
    std::string detection_filename = "object_detection_" + timestamp + ".png";

    // Save the images
    imwrite(capture_filename, frame);
    imwrite(detection_filename, frame_threshold);

    std::cout << "Saved frames: " << capture_filename << " and " << detection_filename << std::endl;
}

int main(int argc, char* argv[])
{
    //! [cap]
    VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);
    //! [cap]

    //! [window]
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    //! [window]

    //! [trackbar]
    // Trackbars to set thresholds for HSV values
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
    //! [trackbar]

    Mat frame, frame_HSV, frame_threshold, frame_filtered;

    while (true) {
        //! [while]
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Apply Gaussian blur with a 3x3 kernel to the frame
        GaussianBlur(frame, frame_filtered, Size(3, 3), 0);

        // Convert from BGR to HSV colorspace
        cvtColor(frame_filtered, frame_HSV, COLOR_BGR2HSV);

        // Detect the object based on HSV Range Values
        inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
        //! [while]

        //! [show]
        // Show the frames
        imshow(window_capture_name, frame);             // Original video capture
        imshow(window_detection_name, frame_threshold); // Object detection result
        //! [show]

        // Wait for key press
        char key = (char)waitKey(30);
        if (key == 'q' || key == 27) {
            break;
        }
        // If 's' is pressed, save both frames
        if (key == 's' || key == 'S') {
            save_frames(frame, frame_threshold);
        }
    }
    return 0;
}

