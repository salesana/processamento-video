#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <ctime>

using namespace cv;

//![variables]
Mat src, src_gray;
Mat dst, detected_edges;
VideoWriter video_canny, video_original;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";
const char* window_original = "Original Frame";
bool recording = false; // Flag for recording
//![variables]

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
static void CannyThreshold(int, void*)
{
    GaussianBlur(src_gray, detected_edges, Size(3, 3), 0);
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);
    
    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);
    
    imshow(window_name, dst);
    imshow(window_original, src);
}

/**
 * @function save_frames
 * @brief Save both the original and the Canny frames to files when 's' is pressed
 */
void save_frames(const Mat& original_frame, const Mat& canny_frame)
{
    time_t now = time(0);
    tm* ltm = localtime(&now);
    std::string timestamp = std::to_string(1900 + ltm->tm_year) + "_" +
                            std::to_string(1 + ltm->tm_mon) + "_" +
                            std::to_string(ltm->tm_mday) + "_" +
                            std::to_string(ltm->tm_hour) + "_" +
                            std::to_string(ltm->tm_min) + "_" +
                            std::to_string(ltm->tm_sec);

    std::string original_filename = "original_frame_" + timestamp + ".png";
    std::string canny_filename = "canny_frame_" + timestamp + ".png";
    imwrite(original_filename, original_frame);
    imwrite(canny_filename, canny_frame);

    std::cout << "Frames saved as: " << original_filename << " and " << canny_filename << std::endl;
}

/**
 * @function start_video_recording
 * @brief Starts video recording for both windows
 */
void start_video_recording(int width, int height)
{
    int fps = 20; // Frames per second
    video_original.open("original_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
    video_canny.open("canny_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
    if (video_original.isOpened() && video_canny.isOpened()) {
        std::cout << "Recording started..." << std::endl;
        recording = true;
    }
}

/**
 * @function stop_video_recording
 * @brief Stops video recording for both windows
 */
void stop_video_recording()
{
    if (recording) {
        video_original.release();
        video_canny.release();
        std::cout << "Recording stopped." << std::endl;
        recording = false;
    }
}

/**
 * @function main
 */
int main(int argc, char** argv)
{
    VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open the webcam!" << std::endl;
        return -1;
    }

    namedWindow(window_name, WINDOW_AUTOSIZE);
    namedWindow(window_original, WINDOW_AUTOSIZE);
    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

    while (true)
    {
        cap >> src;
        if (src.empty()) {
            std::cout << "Error: Could not read a frame from the webcam!" << std::endl;
            break;
        }

        cvtColor(src, src_gray, COLOR_BGR2GRAY);
        dst.create(src.size(), src.type());
        CannyThreshold(0, 0);

        char key = (char)waitKey(30);
        if (key == 'q' || key == 27) {
            stop_video_recording();
            break;
        }

        if (key == 's' || key == 'S') {
            save_frames(src, dst);
        }

        if (key == 'k' || key == 'K') {
            if (!recording) {
                start_video_recording(src.cols, src.rows);
            }
        }

        if (key == 'h' || key == 'H') {
            stop_video_recording();
        }

        if (recording) {
            video_original.write(src);
            video_canny.write(dst);
        }
    }

    return 0;
}

