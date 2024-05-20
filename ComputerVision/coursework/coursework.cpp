#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <cmath>

#define TEST
#undef TEST

using namespace cv;
using namespace std;

void createSpiral(const Mat& inputImage, Mat& outputImage, bool isColor, int spiralWidth) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    Point2f center(width / 2.0f, height / 2.0f);

    outputImage = Mat::zeros(height, width, isColor ? CV_8UC3 : CV_8UC1);

    float maxRadius = hypot(width / 2.0f, height / 2.0f);
    float maxAngle = max(width, height) * CV_PI / spiralWidth;

    for (float angle = 0; angle < maxAngle; angle += 0.001f) {
        float radius = spiralWidth * angle / (2 * CV_PI);

        float x = center.x + radius * cos(angle);
        float y = center.y + radius * sin(angle);

        if (x >= 0 && x < width && y >= 0 && y < height) {
            int ix = static_cast<int>(x);
            int iy = static_cast<int>(y);

            if (isColor) {
                Vec3b color = inputImage.at<Vec3b>(iy, ix);
                outputImage.at<Vec3b>(iy, ix) = color;
            }
            else {
                uchar color = inputImage.at<uchar>(iy, ix);
                outputImage.at<uchar>(iy, ix) = color;
            }
        }
    }
}



int main(int argc) {


#ifdef TEST

    Mat inputImage1 = imread("woman.jpg", IMREAD_COLOR);
    Mat inputImage2 = imread("woman.jpg", IMREAD_COLOR);
    Mat inputImage3 = imread("woman.jpg", IMREAD_COLOR);
    Mat inputImage4 = imread("woman.jpg", IMREAD_COLOR);

    Mat grayImage1;
    Mat grayImage2;
    Mat grayImage3;
    Mat grayImage4;

    cvtColor(inputImage1, grayImage1, COLOR_BGR2GRAY);
    cvtColor(inputImage2, grayImage2, COLOR_BGR2GRAY);
    cvtColor(inputImage3, grayImage3, COLOR_BGR2GRAY);
    cvtColor(inputImage4, grayImage4, COLOR_BGR2GRAY);

    Mat outputImage1;
    Mat outputImage2;
    Mat outputImage3;
    Mat outputImage4;

    createSpiral(false ? inputImage1 : grayImage1, outputImage1, false, 1);
    createSpiral(false ? inputImage2 : grayImage2, outputImage2, false, 3);
    createSpiral(false ? inputImage3 : grayImage3, outputImage3, false, 5);
    createSpiral(false ? inputImage4 : grayImage4, outputImage4, false, 10);

    // Отображение и сохранение результата
    imshow("Spiral Image1", outputImage1);
    imshow("Spiral Image2", outputImage2);
    imshow("Spiral Image3", outputImage3);
    imshow("Spiral Image4", outputImage4);

#else
   
    string path;
    cout << "Input path to image: ";
    cin >> path;

    // Проверка наличия входного изображения
    if (path.empty()) {
        cout << "Usage: ./kursovaya <image_path>" << endl;
        return -1;
    }

    // Загрузка входного изображения
    Mat inputImage = imread(path, IMREAD_COLOR);
    if (inputImage.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Запрос параметров у пользователя
    char colorChoice;
    cout << "Do you want a color (c) or black and white (b) spiral image? ";
    cin >> colorChoice;
    bool isColor = (colorChoice == 'c');

    int spiralWidth;
    cout << "Enter the spiral width (e.g., 5): ";
    cin >> spiralWidth;

    // Преобразование изображения в черно-белое, если это необходимо
    Mat grayImage;
    if (!isColor) {
        cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    }

    // Создание спирального изображения
    Mat outputImage;
    createSpiral(isColor ? inputImage : grayImage, outputImage, isColor, spiralWidth);

    // Отображение и сохранение результата
    imshow("Spiral Image", outputImage);
    imshow("Original Image", inputImage);
    imwrite("spiral_image.jpg", outputImage);
#endif

    waitKey(0);
    return 0;
}
