/**
 * @file CannyDetector_Demo.cpp
 * @brief Sample code showing how to detect edges using the Canny Detector with a webcam and Gaussian blur
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctime>

using namespace cv;

//![variables]
Mat src, src_gray;
Mat dst, detected_edges;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";
const char* window_original = "Original Frame";
//![variables]

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
static void CannyThreshold(int, void*)
{
    //![reduce_noise]
    /// Apply Gaussian blur before Canny detector
    GaussianBlur( src_gray, detected_edges, Size(3, 3), 0 );
    //![reduce_noise]

    //![canny]
    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    //![canny]

    /// Using Canny's output as a mask, we display our result
    //![fill]
    dst = Scalar::all(0);
    //![fill]

    //![copyto]
    src.copyTo( dst, detected_edges);
    //![copyto]

    //![display]
    imshow( window_name, dst );
    imshow( window_original, src );
    //![display]
}

/**
 * @function save_frames
 * @brief Save both the original and the Canny frames to files when 's' is pressed
 */
void save_frames(const Mat& original_frame, const Mat& canny_frame)
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

    // Save the original frame
    std::string original_filename = "original_frame_" + timestamp + ".png";
    imwrite(original_filename, original_frame);

    // Save the Canny edge frame
    std::string canny_filename = "canny_frame_" + timestamp + ".png";
    imwrite(canny_filename, canny_frame);

    std::cout << "Frames saved as: " << original_filename << " and " << canny_filename << std::endl;
}

/**
 * @function main
 */
int main( int argc, char** argv )
{
  //![capture_from_webcam]
  // Open the default camera (webcam)
  VideoCapture cap(0); // 0 is typically the ID for the default webcam
  
  if(!cap.isOpened())  // Check if the camera opened successfully
  {
    std::cout << "Error: Could not open the webcam!" << std::endl;
    return -1;
  }
  //![capture_from_webcam]

  //![create_window]
  namedWindow( window_name, WINDOW_AUTOSIZE );
  namedWindow( window_original, WINDOW_AUTOSIZE );
  //![create_window]

  //![create_trackbar]
  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
  //![create_trackbar]

  while (true)
  {
    //![read_frame]
    cap >> src; // Capture a new frame from the webcam
    if (src.empty())
    {
      std::cout << "Error: Could not read a frame from the webcam!" << std::endl;
      break;
    }
    //![read_frame]

    //![convert_to_gray]
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    //![convert_to_gray]

    //![create_mat]
    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );
    //![create_mat]

    /// Apply the Canny edge detection with Gaussian blur
    CannyThreshold(0, 0);

    /// Break the loop if 'q' or 'esc' key is pressed
    char key = (char)waitKey(30);
    if (key == 'q' || key == 27) {
      break;
    }

    /// If 's' is pressed, save both the original and the edge-detected frames
    if (key == 's' || key == 'S') {
      save_frames(src, dst);  // Save the original and Canny-processed frames
    }
  }

  return 0;
}

