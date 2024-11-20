#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, const char** argv)
{
    String face_cascade_name = "/home/ufabc/Documentos/anasales/lab07/haarcascade_frontalface_alt2.xml";
    String eyes_cascade_name = "/home/ufabc/Documentos/anasales/lab07/haarcascade_eye_tree_eyeglasses.xml";
    String image_path = "/home/ufabc/Documentos/anasales/lab07/caio_selfie.jpeg";

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };

    //-- 2. Read the image
    Mat frame = imread(image_path);

    if (frame.empty())
    {
        cout << "--(!) Error loading image\n";
        return -1;
    }

    //-- 3. Detect and display
    detectAndDisplay(frame);

    //-- 4. Save or exit
    char key = (char)waitKey(0); // Wait indefinitely for a key press
    if (key == 's' || key == 'S')
    {
        imwrite("caio.png", frame);
        cout << "Image saved successfully as 'imagemgrupo.png'.\n";
    }
    else if (key == 27) // Press 'Esc' to exit
    {
        cout << "Exiting program.\n";
    }

    return 0;
}

void detectAndDisplay(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(
        frame_gray, faces,
        1.1,        // Scale factor (smaller values -> more accurate but slower)
        3,          // Minimum neighbors (higher values -> less false positives)
        0,          // Flags (leave as 0 for default behavior)
        Size(30, 30) // Minimum size (adjust for smaller faces)
    );

    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);

        Mat faceROI = frame_gray(faces[i]);

        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(
            faceROI, eyes,
            1.1,    // Scale factor
            3,      // Minimum neighbors
            0,      // Flags
            Size(15, 15) // Minimum size
        );

        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
        }
    }

    //-- Show what you got
    imshow("Capture - Face detection", frame);
}


