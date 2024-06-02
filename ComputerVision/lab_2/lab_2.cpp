#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include "imageGenerator.h"

int main(int argc, char* argv[]) {

	imageGenerator exampleOne(256, 256, CV_8UC1, cv::Scalar(0)), exampleTwo(256, 256, CV_8UC1, cv::Scalar(20)), exampleThree(256, 256, CV_8UC1, cv::Scalar(55)), exampleFour(256, 256, CV_8UC1, cv::Scalar(90));
	imageGenerator histogramOne, histogramTwo, histogramThree, histogramFour;
	imageGenerator resultOne, resultTwo, resultThree, resultFour, result;

	exampleOne.drawRect(CENTRE, 200, 200, cv::Scalar(127));
	exampleOne.drawCircle(CENTRE, 80, cv::Scalar(255));

	resultOne = imageGenerator::vconcat(exampleOne, exampleOne.noise(0.0, 3.0));
	histogramOne = imageGenerator::getHistogram(exampleOne.noise(0.0, 3.0), 256, 256);
	resultOne = imageGenerator::vconcat(resultOne, histogramOne);

	resultOne = imageGenerator::vconcat(resultOne, exampleOne.noise(0.0, 7.0));
	histogramOne = imageGenerator::getHistogram(exampleOne.noise(0.0, 7.0), 256, 256);
	resultOne = imageGenerator::vconcat(resultOne, histogramOne);

	resultOne = imageGenerator::vconcat(resultOne, exampleOne.noise(0.0, 15.0));
	histogramOne = imageGenerator::getHistogram(exampleOne.noise(0.0, 15.0), 256, 256);
	resultOne = imageGenerator::vconcat(resultOne, histogramOne);


	exampleTwo.drawRect(CENTRE, 200, 200, cv::Scalar(127));
	exampleTwo.drawCircle(CENTRE, 80, cv::Scalar(235));

	resultTwo = imageGenerator::vconcat(exampleTwo, exampleTwo.noise(0.0, 3.0));
	histogramTwo = imageGenerator::getHistogram(exampleTwo.noise(0.0, 3.0), 256, 256);
	resultTwo = imageGenerator::vconcat(resultTwo, histogramTwo);

	resultTwo = imageGenerator::vconcat(resultTwo, exampleTwo.noise(0.0, 7.0));
	histogramTwo = imageGenerator::getHistogram(exampleTwo.noise(0.0, 7.0), 256, 256);
	resultTwo = imageGenerator::vconcat(resultTwo, histogramTwo);

	resultTwo = imageGenerator::vconcat(resultTwo, exampleTwo.noise(0.0, 15.0));
	histogramTwo = imageGenerator::getHistogram(exampleTwo.noise(0.0, 15.0), 256, 256);
	resultTwo = imageGenerator::vconcat(resultTwo, histogramTwo);


	exampleThree.drawRect(CENTRE, 200, 200, cv::Scalar(127));
	exampleThree.drawCircle(CENTRE, 80, cv::Scalar(200));

	resultThree = imageGenerator::vconcat(exampleThree, exampleThree.noise(0.0, 3.0));
	histogramThree = imageGenerator::getHistogram(exampleThree.noise(0.0, 3.0), 256, 256);
	resultThree = imageGenerator::vconcat(resultThree, histogramThree);

	resultThree = imageGenerator::vconcat(resultThree, exampleThree.noise(0.0, 7.0));
	histogramThree = imageGenerator::getHistogram(exampleThree.noise(0.0, 7.0), 256, 256);
	resultThree = imageGenerator::vconcat(resultThree, histogramThree);

	resultThree = imageGenerator::vconcat(resultThree, exampleThree.noise(0.0, 15.0));
	histogramThree = imageGenerator::getHistogram(exampleThree.noise(0.0, 15.0), 256, 256);
	resultThree = imageGenerator::vconcat(resultThree, histogramThree);


	exampleFour.drawRect(CENTRE, 200, 200, cv::Scalar(127));
	exampleFour.drawCircle(CENTRE, 80, cv::Scalar(165));

	resultFour = imageGenerator::vconcat(exampleFour, exampleFour.noise(0.0, 3.0));
	histogramFour = imageGenerator::getHistogram(exampleFour.noise(0.0, 3.0), 256, 256);
	resultFour = imageGenerator::vconcat(resultFour, histogramFour);

	resultFour = imageGenerator::vconcat(resultFour, exampleFour.noise(0.0, 7.0));
	histogramFour = imageGenerator::getHistogram(exampleFour.noise(0.0, 7.0), 256, 256);
	resultFour = imageGenerator::vconcat(resultFour, histogramFour);

	resultFour = imageGenerator::vconcat(resultFour, exampleFour.noise(0.0, 15.0));
	histogramFour = imageGenerator::getHistogram(exampleFour.noise(0.0, 15.0), 256, 256);
	resultFour = imageGenerator::vconcat(resultFour, histogramFour);


	result = imageGenerator::hconcat(resultOne, resultTwo);
	result = imageGenerator::hconcat(result, resultThree);
	result = imageGenerator::hconcat(result, resultFour);

	result.showImage("image");
	result.saveImage("image.png");

	return 0;
}
