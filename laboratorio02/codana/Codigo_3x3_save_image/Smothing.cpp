#include <iostream>
#include <sstream> // Para converter inteiros em strings
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;

Mat src; Mat dst;
char window_name[] = "Smoothing Demo";

int display_caption(const char* caption);
int display_dst(int delay, const string& filter_name);

int main(int argc, char ** argv)
{
    namedWindow(window_name, WINDOW_AUTOSIZE);

    const char* filename = argc >= 2 ? argv[1] : "foto_grupo_1.png";

    src = imread(samples::findFile(filename), IMREAD_COLOR);
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Usage:\n %s [image_name-- default lena.jpg] \n", argv[0]);
        return EXIT_FAILURE;
    }

    if (display_caption("Original Image") != 0)
    {
        return 0;
    }

    dst = src.clone();
    if (display_dst(DELAY_CAPTION, "original") != 0)
    {
        return 0;
    }

    // Aplicando o Filtro de Média (Homogeneous Blur)
    if (display_caption("Homogeneous Blur") != 0)
    {
        return 0;
    }

    blur(src, dst, Size(3, 3), Point(-1, -1));
    if (display_dst(DELAY_BLUR, "homogeneous_blur") != 0)
    {
        return 0;
    }

    // Aplicando o Filtro Gaussiano
    if (display_caption("Gaussian Blur") != 0)
    {
        return 0;
    }

    GaussianBlur(src, dst, Size(3, 3), 0, 0);
    if (display_dst(DELAY_BLUR, "gaussian_blur") != 0)
    {
        return 0;
    }

    // Aplicando o Filtro de Mediana
    if (display_caption("Median Blur") != 0)
    {
        return 0;
    }

    medianBlur(src, dst, 3);
    if (display_dst(DELAY_BLUR, "median_blur") != 0)
    {
        return 0;
    }

    // Aplicando o Filtro Bilateral
    if (display_caption("Bilateral Blur") != 0)
    {
        return 0;
    }

    bilateralFilter(src, dst, 3, 6, 1.5);
    if (display_dst(DELAY_BLUR, "bilateral_blur") != 0)
    {
        return 0;
    }

    display_caption("Done!");

    return 0;
}

int display_caption(const char* caption)
{
    dst = Mat::zeros(src.size(), src.type());
    putText(dst, caption,
        Point(src.cols / 4, src.rows / 2),
        FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));

    return display_dst(DELAY_CAPTION, "");
}

int display_dst(int delay, const string& filter_name)
{
    imshow(window_name, dst);

    if (!filter_name.empty()) {
        stringstream ss;
        ss << filter_name << ".jpg";
        imwrite(ss.str(), dst); // Salva a imagem com o nome baseado no tipo de filtro
        cout << "Imagem salva: " << ss.str() << endl;
    }

    int c = waitKey(delay);
    if (c >= 0) { return -1; }
    return 0;
}

