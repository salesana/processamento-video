/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to use the function calcHist
 * @author
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
int main(int argc, char** argv)
{
 //! [Open webcam]
    VideoCapture cap(0); // Open the default camera (0)
    if (!cap.isOpened())
    {
        cerr << "Error: Unable to open the webcam." << endl;
        return EXIT_FAILURE;
    }
    while (true)
	{
		Mat src;
		cap >> src; // Capture a new frame from the webcam

		if (src.empty())
		{
		    cerr << "Error: Unable to capture frame." << endl;
		    break;
		}
		
		imshow("Image original", src);

	    //! [Load image]
	    //CommandLineParser parser( argc, argv, "{@input | lena.jpg | input image}" );
	    //Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
	    //! [Load image]

	    //! [Separate the image in 3 places ( B, G and R )]
	    vector<Mat> bgr_planes;
	    split( src, bgr_planes );
	    //! [Separate the image in 3 places ( B, G and R )]

	    //! [Establish the number of bins]
	    int histSize = 256;
	    //! [Establish the number of bins]

	    //! [Set the ranges ( for B,G,R) )]
	    float range[] = { 0, 256 }; //the upper boundary is exclusive
	    const float* histRange[] = { range };
	    const float* histRange2[] = { range };
	    //! [Set the ranges ( for B,G,R) )]

	    //! [Set histogram param]
	    bool uniform = true, accumulate = false;
	    //! [Set histogram param]

	    //! [Compute the histograms]
	    Mat b_hist, g_hist, r_hist;
	    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
	    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
	    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );
	    //! [Compute the histograms]

	    //! [Draw the histograms for B, G and R]
	    int hist_w = 512, hist_h = 400;
	    int bin_w = cvRound( (double) hist_w/histSize );

	    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	    //! [Draw the histograms for B, G and R]

	    //! [Normalize the result to ( 0, histImage.rows )]
	    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	    //! [Normalize the result to ( 0, histImage.rows )]

	    //! [Draw for each channel]
	    for( int i = 1; i < histSize; i++ )
	    {
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
		      Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
		      Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
		      Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
		      Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
		      Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
		      Scalar( 0, 0, 255), 2, 8, 0  );
	    }
	    //! [Draw for each channel]
	    
	    //Aplicar equalizacao nos canais
	Mat equalized_red, equalized_green, equalized_blue;
       	equalizeHist(bgr_planes[0], equalized_red);
       	equalizeHist(bgr_planes[1], equalized_green);
       	equalizeHist(bgr_planes[2], equalized_blue);
       	
       	//! [Compute the histograms]
	    Mat b_hist_eq, g_hist_eq, r_hist_eq;
	    calcHist( &equalized_red, 1, 0, Mat(), b_hist_eq, 1, &histSize, histRange2, uniform, accumulate );
	    calcHist( &equalized_green, 1, 0, Mat(), g_hist_eq, 1, &histSize, histRange2, uniform, accumulate );
	    calcHist( &equalized_blue, 1, 0, Mat(), r_hist_eq, 1, &histSize, histRange2, uniform, accumulate );
	//! [Compute the histograms]
	
	//! [Draw the histograms for B, G and R]
	    int hist_w2 = 512, hist_h2 = 400;
	    int bin_w2 = cvRound( (double) hist_w2/histSize );

	    Mat histImage2( hist_h2, hist_w2, CV_8UC3, Scalar( 0,0,0) );
	//! [Draw the histograms for B, G and R]
	
	//! [Normalize the result to ( 0, histImage.rows )]
	    normalize(b_hist_eq, b_hist_eq, 0, histImage2.rows, NORM_MINMAX, -1, Mat() );
	    normalize(g_hist_eq, g_hist_eq, 0, histImage2.rows, NORM_MINMAX, -1, Mat() );
	    normalize(r_hist_eq, r_hist_eq, 0, histImage2.rows, NORM_MINMAX, -1, Mat() );
	//! [Normalize the result to ( 0, histImage.rows )]
	
	
	//! [Draw for each channel]
	    for( int i = 1; i < histSize; i++ )
	    {
		line( histImage2, Point( bin_w2*(i-1), hist_h2 - cvRound(b_hist_eq.at<float>(i-1)) ),
		      Point( bin_w2*(i), hist_h2 - cvRound(b_hist_eq.at<float>(i)) ),
		      Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImage2, Point( bin_w2*(i-1), hist_h2 - cvRound(g_hist_eq.at<float>(i-1)) ),
		      Point( bin_w2*(i), hist_h2 - cvRound(g_hist_eq.at<float>(i)) ),
		      Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage2, Point( bin_w2*(i-1), hist_h2 - cvRound(r_hist_eq.at<float>(i-1)) ),
		      Point( bin_w2*(i), hist_h2 - cvRound(r_hist_eq.at<float>(i)) ),
		      Scalar( 0, 0, 255), 2, 8, 0  );
	    }
	//! [Draw for each channel]
	/*Mat dst;
	dst = merge((equalized_blue,equalized_green,equalized_red));
	*/
	    //! [Display]
	    //imshow("Source image", src );
	   imshow("calcHist Demo", histImage );
	   imshow("histograma_equalizado", histImage2 );
	   //imshow("Vermelho_equalizado",equalized_red);
	    
	    //! [Save images on key press]
        char key = (char)waitKey(30); // Wait 30 ms for a key press
        if (key == 's' || key == 'S')
        {
            imwrite("Imagem_Original.png", src);
            imwrite("Histograma.png",histImage);
            imwrite("Histograma_Equalizado.png",histImage2);
            cout << "Images saved successfully.\n";
        }
        else if (key == 27) // Press 'Esc' to exit
        {
            break;
        }
        //! [Save images on key press]

	    //return EXIT_SUCCESS;
	}
	cap.release(); // Release the webcam
	destroyAllWindows();
	return 0;
}
