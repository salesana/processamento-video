#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

void Add_salt_pepper_Noise(Mat &srcArr, float pa, float pb) {
    RNG rng; // Random number generator
    int amount1 = srcArr.rows * srcArr.cols * pa;
    int amount2 = srcArr.rows * srcArr.cols * pb;

    // Adding "salt" noise (black pixels)
    for (int counter = 0; counter < amount1; ++counter) {
        srcArr.at<uchar>(rng.uniform(0, srcArr.rows), rng.uniform(0, srcArr.cols)) = 0;
    }

    // Adding "pepper" noise (white pixels)
    for (int counter = 0; counter < amount2; ++counter) {
        srcArr.at<uchar>(rng.uniform(0, srcArr.rows), rng.uniform(0, srcArr.cols)) = 255;
    }
}

int main(int argc, char *argv[]) {
    Mat srcArr;

    // Load the image
    if (argc <= 1) {
        srcArr = imread("grupo.png");
    } else if (argc >= 2) {
        srcArr = imread(argv[1]);
    }

    // Check if the image was loaded successfully
    if (srcArr.empty()) {
        cerr << "Error: Image not found!" << endl;
        return -1;
    }

    // Convert the image to grayscale
    cvtColor(srcArr, srcArr, COLOR_RGB2GRAY, 1);
    imshow("The original Image", srcArr);

    Mat srcArr1 = srcArr.clone(); // Clone to add noise
    Mat srcArr2 = srcArr.clone(); // Clone for black and white processing
    Mat dstArr;

    // Parameters for noise
    float pa, pb;
    pa = 0.05; // Percentage of salt noise
    pb = 0.05; // Percentage of pepper noise

    // Add salt-and-pepper noise to the image
    Add_salt_pepper_Noise(srcArr1, pa, pb);
    imshow("Add salt and pepper noise to image", srcArr1);
    imwrite("salt_pepper_noise_image.jpg", srcArr1);

    // Convert the noisy image to black and white (binary image)
    double threshold_value = 128; // You can adjust this value as needed
    threshold(srcArr1, srcArr2, threshold_value, 255, THRESH_BINARY);
    imshow("Black and White Image", srcArr2);
    imwrite("black_white_image.jpg", srcArr2);

    waitKey(0);
    return 0;
}

