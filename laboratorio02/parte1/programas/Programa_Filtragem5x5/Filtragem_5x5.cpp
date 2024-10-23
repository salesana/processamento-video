#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

/// Global Variables
int MAX_KERNEL_LENGTH = 5;
Mat src; Mat img_blur; Mat gaussian; Mat median; Mat bilateral;

/**
 * function main
 */
int main( int argc, char ** argv )
{
    /// Load the source image
    const char* filename = argc >=2 ? argv[1] : "foto_grupo_2.png";

    src = imread( samples::findFile( filename ), IMREAD_COLOR );
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Usage:\n %s [image_name-- foto_grupo_2.png] \n", argv[0]);
        return EXIT_FAILURE;
    }

    img_blur = src.clone();
    gaussian = src.clone();
    median = src.clone();
    bilateral = src.clone();
    
    	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    {
        blur( src, img_blur, Size( i, i ), Point(-1,-1) );
    }
	imwrite("blur_5x5.png", img_blur);
	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    {
        GaussianBlur( src, gaussian, Size( i, i ), 0, 0 );
    }
    	imwrite("gaussian_5x5.png", gaussian);
    	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    {
        medianBlur ( src, median, i );
    }
    	imwrite("median_5x5.png", median);
    	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    {
        bilateralFilter ( src, bilateral, i, i*2, i/2 );
    }
    	imwrite("bilateral_5x5.png", bilateral);
}
