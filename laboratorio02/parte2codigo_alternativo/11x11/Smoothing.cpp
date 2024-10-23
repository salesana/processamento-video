#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
Mat src; Mat dst;
char window_name[] = "Smoothing Demo";

/// Function headers
int display_caption( const char* caption );
int display_dst( int delay, const string& filename );

/**
 * function main
 */
int main( int argc, char ** argv )
{
    namedWindow( window_name, WINDOW_AUTOSIZE );

    /// Load the source image
    const char* filename = argc >= 2 ? argv[1] : "original.jpg";

    src = imread( samples::findFile( filename ), IMREAD_COLOR );
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Usage:\n %s [image_name-- default lena.jpg] \n", argv[0]);
        return EXIT_FAILURE;
    }

    if( display_caption( "Original Image" ) != 0 )
    {
        return 0;
    }

    dst = src.clone();
    if( display_dst( DELAY_CAPTION, "original_image.jpg" ) != 0 )
    {
        return 0;
    }

    /// Applying Homogeneous blur with a 7x7 kernel
    if( display_caption( "Homogeneous Blur" ) != 0 )
    {
        return 0;
    }

    //![blur]
    blur( src, dst, Size( 7, 7 ), Point(-1,-1) );
    if( display_dst( DELAY_BLUR, "homogeneous_blur.jpg" ) != 0 )
    {
        return 0;
    }
    //![blur]

    /// Applying Gaussian blur with a 7x7 kernel
    if( display_caption( "Gaussian Blur" ) != 0 )
    {
        return 0;
    }

    //![gaussianblur]
    GaussianBlur( src, dst, Size( 7, 7 ), 0, 0 );
    if( display_dst( DELAY_BLUR, "gaussian_blur.jpg" ) != 0 )
    {
        return 0;
    }
    //![gaussianblur]

    /// Applying Median blur with a 7x7 kernel
    if( display_caption( "Median Blur" ) != 0 )
    {
        return 0;
    }

    //![medianblur]
    medianBlur( src, dst, 7 );
    if( display_dst( DELAY_BLUR, "median_blur.jpg" ) != 0 )
    {
        return 0;
    }
    //![medianblur]

    /// Applying Bilateral Filter with a 7x7 filter
    if( display_caption( "Bilateral Blur" ) != 0 )
    {
        return 0;
    }

    //![bilateralfilter]
    bilateralFilter( src, dst, 7, 7*2, 7/2 );
    if( display_dst( DELAY_BLUR, "bilateral_blur.jpg" ) != 0 )
    {
        return 0;
    }
    //![bilateralfilter]

    /// Done
    display_caption( "Done!" );

    return 0;
}

/**
 * @function display_caption
 */
int display_caption( const char* caption )
{
    dst = Mat::zeros( src.size(), src.type() );
    putText( dst, caption,
             Point( src.cols/4, src.rows/2),
             FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );

    return display_dst(DELAY_CAPTION, "");
}

/**
 * @function display_dst
 */
int display_dst( int delay, const string& filename )
{
    imshow( window_name, dst );
    
    // Save the file if a valid filename is provided
    if (!filename.empty()) {
        imwrite(filename, dst); // Save the image with the given filename
    }
    
    int c = waitKey( delay );
    if( c >= 0 ) { return -1; }
    return 0;
}

