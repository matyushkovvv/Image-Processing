#include "imageGenerator.h"
#include <stdexcept>
#include <iostream>

imageGenerator::imageGenerator(int height, int width, int type, cv::Scalar color)
    : rows(height), cols(width), typeOfImage(type)
{
    try {
        if (rows < 1)
            rows = 1;

        if (cols < 1)
            cols = 1;

        cv::Mat img(rows, cols, typeOfImage, color);
        image = img;

        //throw std::runtime_error("ERROR: initialization failed.");
    }
    catch (const std::exception& e) {
        std::cout << "ERROR: initialization failed." << std::endl;
        throw;
    }
}

imageGenerator::imageGenerator(int height, int width)
    : rows(height), cols(width), typeOfImage(CV_8UC1)
{
    try {
        if (rows < 1)
            rows = 1;

        if (cols < 1)
            cols = 1;

        cv::Mat img(rows, cols, typeOfImage, cv::Scalar(0));
        image = img;

        //throw std::runtime_error("ERROR: initialization failed.");
    }
    catch (const std::exception& e) {
        std::cout << "ERROR: initialization failed." << std::endl;
        throw;
    }
}

imageGenerator::imageGenerator(const imageGenerator& obj) {
    image = obj.image;

    rows = obj.rows;
    cols = obj.cols;

    typeOfImage = obj.typeOfImage;
}

imageGenerator::imageGenerator(cv::Mat& img) {
    image = img;

    rows = image.rows;
    cols = image.cols;
}

imageGenerator::imageGenerator()
    : rows(400), cols(400), typeOfImage(CV_8UC1)
{
    try {

        cv::Mat img(rows, cols, typeOfImage, cv::Scalar(0));
        image = img;

        //throw std::runtime_error("ERROR: initialization failed.");
    }
    catch (const std::exception& e) {
        std::cout << "ERROR: initialization failed." << std::endl;
        throw;
    }
}

imageGenerator imageGenerator::gradient(int rectangleHeight, int rectangleWidth) {

    cv::Mat copyImage = image;
    int numRectangles = cols / rectangleWidth;

    int gradientStep = 255 / numRectangles;
    int gradientValue = 0;

    for (int i = 0; i < numRectangles; i++) {
        cv::Range currentRange(0, rows);
        cv::Mat currentRectangle = copyImage.colRange(i * rectangleWidth, (i + 1) * rectangleWidth);
        currentRectangle.setTo(gradientValue);

        gradientValue += gradientStep;
    }

    return imageGenerator(copyImage);
}

imageGenerator imageGenerator::gamma(float ratio) {

    cv::Mat gammaCorrected;
    cv::Mat gammaCorrectedScaled;

    // Конвертируем в формат с плавающей точкой и диапазоном 0-1
    image.convertTo(gammaCorrected, CV_32FC1, 1.0 / 255.0);

    // Гамма-коррекция
    cv::pow(gammaCorrected, ratio, gammaCorrected);

    // Конвертируем обратно в 8-битный формат
    gammaCorrected.convertTo(gammaCorrectedScaled, CV_8UC1, 255.0);

    return imageGenerator(gammaCorrectedScaled);
}



imageGenerator imageGenerator::noise(float mean, float StdDev) {

    cv::Mat mDst;
    cv::Mat mSrc_16SC;
    cv::Mat mGaussian_noise = cv::Mat(getMat().size(), CV_16SC1);
    cv::randn(mGaussian_noise, cv::Scalar::all(mean), cv::Scalar::all(StdDev));

    getMat().convertTo(mSrc_16SC, CV_16SC1);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(mDst, getMat().type());

    return imageGenerator(mDst);

}


void imageGenerator::drawCircle(POSITION pos, int radius, cv::Scalar color, int thickness) {
    switch (pos) {
    case LEFT:
        cv::circle(
            image,
            cv::Point(radius, image.rows / 2),
            radius,
            color,
            thickness);

        break;
    case TOP:
        cv::circle(
            image,
            cv::Point(image.cols / 2, radius),
            radius,
            color,
            thickness);

        break;
    case RIGHT:
        cv::circle(
            image,
            cv::Point(image.cols - radius, image.rows / 2),
            radius,
            color,
            thickness);

        break;
    case BOTTOM:
        cv::circle(
            image,
            cv::Point(image.cols / 2, image.rows - radius),
            radius,
            color,
            thickness);

        break;
    case CENTRE:
        cv::circle(
            image,
            cv::Point(image.cols / 2, image.rows / 2),
            radius,
            color,
            thickness);

        break;
    }
}

void imageGenerator::drawRect(POSITION pos, int height, int width, cv::Scalar color, int thickness) {
    switch (pos) {
    case LEFT:
        cv::rectangle(
            image,
            cv::Point(0, (image.rows / 2) - (height / 2)),
            cv::Point(width, (image.rows / 2) + (height / 2)),
            color,
            thickness);

        break;
    case TOP:
        cv::rectangle(
            image,
            cv::Point((image.cols / 2) - (width / 2), 0),
            cv::Point((image.cols / 2) + (width / 2), height),
            color,
            thickness);

        break;
    case RIGHT:
        cv::rectangle(
            image,
            cv::Point(image.cols - width, (image.rows / 2) - (height / 2)),
            cv::Point(image.cols, (image.rows / 2) + (height / 2)),
            color,
            thickness);

        break;
    case BOTTOM:
        cv::rectangle(
            image,
            cv::Point((image.cols / 2) - (width / 2), image.rows - height),
            cv::Point((image.cols / 2) + (width / 2), image.rows),
            color,
            thickness);

        break;
    case CENTRE:
        cv::rectangle(
            image,
            cv::Point((image.cols / 2) - (width / 2), (image.rows / 2) - (height / 2)),
            cv::Point((image.cols / 2) + (width / 2), (image.rows / 2) + (height / 2)),
            color,
            thickness);

        break;
    }
}

int imageGenerator::getHeight() {
    return rows;
}

int imageGenerator::getWidth() {
    return cols;
}

void imageGenerator::showImage(cv::String name) {
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, image);
    cv::waitKey(0);

    cv::destroyWindow(name);
}

bool imageGenerator::saveImage(cv::String name) {
    return cv::imwrite(name, image);
}

imageGenerator imageGenerator::loadImage(cv::String name) {

    cv::Mat img;
    img = cv::imread(name);

    return imageGenerator(img);
}

cv::Mat imageGenerator::getMat() {
    return image;
}

/*
imageGenerator imageGenerator::getHistogram(imageGenerator obj, int width, int height) {

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&obj.image, 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);

    int hist_w = width, hist_h = height;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            cv::Scalar(255), 2, 8, 0);
    }

    return imageGenerator(histImage);
}
*/

imageGenerator imageGenerator::getHistogram(imageGenerator obj, int width, int height) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&obj.image, 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);

    int hist_w = width, hist_h = height;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255));
    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // draw the intensity line for histogram
    for (int i = 0; i < 255; i++)
    {
        line(histImage, cv::Point(bin_w * (i), hist_h),
            cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            cv::Scalar(0), 1, 8, 0);
    }

    return imageGenerator(histImage);
}

imageGenerator imageGenerator::getHistogram(cv::Mat obj, int width, int height) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;

    cv::calcHist(&obj, 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);

    int hist_w = width, hist_h = height;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255));
    cv::normalize(b_hist, b_hist, 0, histImage.rows, 230, -1, cv::Mat());

    for (int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            cv::Scalar(0), 2, 8, 0);
    }

    return imageGenerator(histImage);
}

imageGenerator imageGenerator::vconcat(imageGenerator top, imageGenerator bottom) {

    if (top.cols != bottom.cols)
        return imageGenerator();

    cv::Mat result;

    cv::vconcat(top.getMat(), bottom.getMat(), result);

    return imageGenerator(result);
}

imageGenerator imageGenerator::hconcat(imageGenerator left, imageGenerator right) {

    if (left.rows != right.rows)
        return imageGenerator();

    cv::Mat result;

    cv::hconcat(left.getMat(), right.getMat(), result);

    return imageGenerator(result);
}
