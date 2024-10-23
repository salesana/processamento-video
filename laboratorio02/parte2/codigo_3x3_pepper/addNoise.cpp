#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void Add_salt_pepper_Noise(Mat &srcArr, float pa, float pb) {
    RNG rng;
    int amount1 = static_cast<int>(srcArr.rows * srcArr.cols * pa);
    int amount2 = static_cast<int>(srcArr.rows * srcArr.cols * pb);

    for (int counter = 0; counter < amount1; ++counter) {
        srcArr.at<uchar>(rng.uniform(0, srcArr.rows), rng.uniform(0, srcArr.cols)) = 0; // Salt
    }
    for (int counter = 0; counter < amount2; ++counter) {
        srcArr.at<uchar>(rng.uniform(0, srcArr.rows), rng.uniform(0, srcArr.cols)) = 255; // Pepper
    }
}

int main(int argc, char *argv[]) {
    Mat srcArr;

    if (argc <= 1) {
        srcArr = imread("/home/ufabc/Documentos/anasales/lab02/saltpeppernoise/codigo_3x3_pepper/foto_grupo_1.png");
    } else {
        srcArr = imread(argv[1]);
    }

    // Check if the image was loaded successfully
    if (srcArr.empty()) {
        cerr << "Error: Image not found!" << endl;
        return -1;
    }

    cvtColor(srcArr, srcArr, cv::COLOR_BGR2GRAY); // Convert to grayscale
    imshow("The original Image", srcArr);

    Mat srcArr1 = srcArr.clone();
    Mat dstArr;

    float pa, pb;
    cout << "Input the pa and pb of the expected salt & pepper noise: ";
    cin >> pa >> pb;

    Add_salt_pepper_Noise(srcArr1, pa, pb);
    imshow("Add salt and pepper noise to image", srcArr1);
    imwrite("salt_pepper_noise_image.tif", srcArr1);

    medianBlur(srcArr1, dstArr, 5);
    imshow("The effect after median filter", dstArr);
    imwrite("filtered_image.tif", dstArr);

    waitKey(0);
    return 0;
}

