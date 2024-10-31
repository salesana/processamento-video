/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to calculate and equalize histogram of a grayscale image from webcam feed
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/**
 * @function main
 */
int main()
{
    //! [Open webcam]
    VideoCapture cap(0); // Open the default camera (0)
    if (!cap.isOpened())
    {
        cerr << "Error: Unable to open the webcam." << endl;
        return EXIT_FAILURE;
    }
    //! [Open webcam]

    while (true)
    {
        Mat frame;
        cap >> frame; // Capture a new frame from the webcam

        if (frame.empty())
        {
            cerr << "Error: Unable to capture frame." << endl;
            break;
        }
        
        imshow("Image original", frame);

        //! [Convert to grayscale]
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        //! [Convert to grayscale]

        //! [Compute histogram of grayscale image]
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        Mat gray_hist;
        calcHist(&gray, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange);

        //! [Draw histogram for grayscale image]
        int hist_w = 512, hist_h = 400;
        int bin_w = cvRound((double)hist_w / histSize);
        Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

        normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX);

        for (int i = 1; i < histSize; i++)
        {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
                 Point(bin_w * i, hist_h - cvRound(gray_hist.at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);
        }
        //! [Draw histogram for grayscale image]

        //! [Apply Histogram Equalization]
        Mat equalized;
        equalizeHist(gray, equalized);
        //! [Apply Histogram Equalization]

        //! [Compute histogram of equalized image]
        Mat equalized_hist;
        calcHist(&equalized, 1, 0, Mat(), equalized_hist, 1, &histSize, &histRange);

        //! [Draw histogram for equalized image]
        Mat equalizedHistImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
        normalize(equalized_hist, equalized_hist, 0, equalizedHistImage.rows, NORM_MINMAX);

        for (int i = 1; i < histSize; i++)
        {
            line(equalizedHistImage, Point(bin_w * (i - 1), hist_h - cvRound(equalized_hist.at<float>(i - 1))),
                 Point(bin_w * i, hist_h - cvRound(equalized_hist.at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);
        }
        //! [Draw histogram for equalized image]

        //! [Display results]
        imshow("Grayscale Image", gray);
        imshow("Grayscale Histogram", histImage);
        imshow("Equalized Image", equalized);
        imshow("Equalized Histogram", equalizedHistImage);

        //! [Save images on key press]
        char key = (char)waitKey(30); // Wait 30 ms for a key press
        if (key == 's' || key == 'S')
        {
            imwrite("Grayscale_Image.png", gray);
            imwrite("Equalized_Image.png", equalized);
            imwrite("Grayscale_Histogram.png", histImage);
            imwrite("Equalized_Histogram.png", equalizedHistImage);
            cout << "Images saved successfully.\n";
        }
        else if (key == 27) // Press 'Esc' to exit
        {
            break;
        }
        //! [Save images on key press]
    }

    cap.release(); // Release the webcam
    destroyAllWindows();
    return 0;
}

