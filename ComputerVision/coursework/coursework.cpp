#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <cmath>

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
    imwrite("spiral_image.jpg", outputImage);

    waitKey(0);
    return 0;
}
