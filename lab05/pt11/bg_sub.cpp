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
      "{ input          | pessoas_rapido.avi | Path to a video or a sequence of image }"
      "{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);
    parser.about( "This program shows how to use background subtraction methods provided by "
                  " OpenCV. You can process both videos and images.\n" );
    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }

    //! [create]
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();
    //! [create]

    //! [capture]
    VideoCapture capture( samples::findFile( parser.get<String>("input") ) );
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }
    //! [capture]

    // Create VideoWriter objects to save the frames and foreground masks
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    string originalOutput = "output.avi";
    string fgMaskOutput = "fg_mask_output.avi";

    VideoWriter originalVideoWriter(originalOutput, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height));
    VideoWriter fgMaskWriter(fgMaskOutput, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height), false);

    if (!originalVideoWriter.isOpened() || !fgMaskWriter.isOpened()) {
        cerr << "Could not open the output video files." << endl;
        return -1;
    }

    Mat frame, fgMask;
    while (true) {
        capture >> frame;
        if (frame.empty())
            break;

        //! [apply]
        //update the background model
        pBackSub->apply(frame, fgMask);
        //! [apply]

        //! [display_frame_number]
        //get the frame number and write it on the current frame
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        stringstream ss;
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        //! [display_frame_number]

        //! [show]
        //show the current frame and the fg masks
        imshow("Frame", frame);
        imshow("FG Mask", fgMask);
        //! [show]

        // Save the current frame and the foreground mask to the respective video files
        originalVideoWriter.write(frame);
        fgMaskWriter.write(fgMask);

        // Check if 'S' is pressed to stop and save the videos
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27) // 'q' or ESC to quit
            break;
        else if (keyboard == 's' || keyboard == 'S') { // 'S' to save and stop
            cout << "Saving and exiting..." << endl;
            break;
        }
    }

    // Release video writers
    originalVideoWriter.release();
    fgMaskWriter.release();

    return 0;
}

