#pragma once

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <map>

typedef enum {
	LEFT,
	TOP,
	RIGHT,
	BOTTOM,
	CENTRE
} POSITION;

class imageGenerator
{
	cv::Mat image;

	int rows;
	int cols;

	int typeOfImage;

public:

	imageGenerator(int height, int width, int typeOfImage, cv::Scalar color);
	imageGenerator(int height, int width);
	imageGenerator(const imageGenerator& obj);
	imageGenerator(cv::Mat& img);
	imageGenerator();

	imageGenerator gradient(int rectangleHeight = 30, int rectangleWidth = 3);
	imageGenerator gamma(float ratio = 2.4);
	imageGenerator noise(float mean, float StdDev);
	void drawCircle(POSITION pos, int radius, cv::Scalar color, int thickness = -1);
	void drawRect(POSITION, int height, int width, cv::Scalar color, int thickness = -1);

	void showImage(cv::String name = "image");
	bool saveImage(cv::String name);
	static imageGenerator loadImage(cv::String name);

	int getHeight();
	int getWidth();
	cv::Mat getMat();

	static imageGenerator getHistogram(imageGenerator obj, int width, int height);
	static imageGenerator getHistogram(cv::Mat obj, int width, int height);
	static imageGenerator vconcat(imageGenerator top, imageGenerator bottom);
	static imageGenerator hconcat(imageGenerator left, imageGenerator right);
};
