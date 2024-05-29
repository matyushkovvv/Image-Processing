#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

double averageColor(const Mat& inputImage) {
    Scalar sum = cv::sum(inputImage);
    double totalSum = sum[0] + sum[1] + sum[2];
    int totalPixels = inputImage.rows * inputImage.cols;
    double average = totalSum / (totalPixels * 3);

    return average;
}

Mat increaseBrightness(const Mat& inputImage, int brightnessValue) {

    Mat brightenedImage;
    inputImage.convertTo(brightenedImage, -1, 1, brightnessValue); // Увеличиваем яркость каждого пикселя

    return brightenedImage;
}

void createSpiralImage(const Mat& inputImage, Mat& outlineImage, Mat& filledImage, int spiralWidth) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    Point2f center(width / 2.0f, height / 2.0f);

    outlineImage = Mat::ones(height, width, CV_8UC1) * 255;
    filledImage = Mat::ones(height, width, CV_8UC1) * 255; 

    float maxAngle = max(height, width) * 2 * CV_PI; // Устанавливаем число оборотов

    Point2f prevPoint1, prevPoint2;
    bool hasPrevPoint = false;

    for (float angle = 0; angle < maxAngle; angle += 0.01f) {
        float radius = spiralWidth * angle / (2 * CV_PI);

        float x = center.x + radius * cos(angle);
        float y = center.y + radius * sin(angle);

        if (x >= 0 && x < width && y >= 0 && y < height) {
            int ix = static_cast<int>(x);
            int iy = static_cast<int>(y);

            uchar intensity;
            if (inputImage.channels() == 3) {
                Vec3b color = inputImage.at<Vec3b>(iy, ix);
                intensity = (color[0] + color[1] + color[2]) / 3; // Среднее значение для RGB изображения
            }
            else {
                intensity = inputImage.at<uchar>(iy, ix); // Значение для градаций серого
            }

            float thickness = 1 + ((255 - intensity) / 255.0f) * (spiralWidth - 1); // Толщина в зависимости от интенсивности

            Point2f point1(
                center.x + (radius - thickness / 2) * cos(angle),
                center.y + (radius - thickness / 2) * sin(angle)
            );
            Point2f point2(
                center.x + (radius + thickness / 2) * cos(angle),
                center.y + (radius + thickness / 2) * sin(angle)
            );

            if (hasPrevPoint) {
                line(outlineImage, prevPoint1, point1, Scalar(0, 0, 0), 1);
                line(outlineImage, prevPoint2, point2, Scalar(0, 0, 0), 1);

                // Заполнение области между линиями
                for (float i = 0; i <= 1.0; i += 0.01f) {
                    Point2f interPoint1 = prevPoint1 * (1.0f - i) + point1 * i;
                    Point2f interPoint2 = prevPoint2 * (1.0f - i) + point2 * i;
                    line(filledImage, interPoint1, interPoint2, Scalar(0, 0, 0), 1);
                }
            }

            prevPoint1 = point1;
            prevPoint2 = point2;
            hasPrevPoint = true;
        }
        else {
            hasPrevPoint = false; // Сбрасываем флаг, если точка выходит за границы изображения
        }
    }
}

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << "Usage: ./coursework.exe <image_path> <spiral_width>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    int spiralWidth = atoi(argv[2]) >= 5 ? atoi(argv[2]) : 5;

    Mat inputImage = imread(imagePath, IMREAD_COLOR);
    if (inputImage.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Контрастирование
    if (averageColor(inputImage) < 220) {
        char answer;
        cout << "The image is too dark, do you want to make it brighter? (y/n): ";
        cin >> answer;

        if (answer == 'y') {
            inputImage = increaseBrightness(inputImage, 200 - averageColor(inputImage));
        }
        else {
            inputImage = increaseBrightness(inputImage, 0);
        }
    }

    Mat outlineImage, filledImage;
    createSpiralImage(inputImage, outlineImage, filledImage, spiralWidth);

    imshow("inputImage", inputImage);
    imshow("Spiral Outline Image", outlineImage);
    imshow("Spiral Filled Image", filledImage);
    imwrite("input_image.jpg", inputImage);
    imwrite("spiral_outline_image.jpg", outlineImage);
    imwrite("spiral_filled_image.jpg", filledImage);

    waitKey(0);
    return 0;
}