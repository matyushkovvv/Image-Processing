#include <iostream>
#include <string>
#include <filesystem>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>


cv::Mat getHist(cv::Mat image, int hist_h, int hist_w) {

	int typeImage = CV_8UC1;
	if (image.channels() == 3)
		typeImage = CV_8UC3;

	cv::Mat histImage(hist_h, hist_w, typeImage, cv::Scalar(200, 200, 200));

	// Разделение исходного изображения на отдельные каналы
	std::vector<cv::Mat> channels;
	cv::split(image, channels);

	for (int c = 0; c < channels.size(); c++) {
		cv::Mat hist;
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		int bin_w = cvRound((double)hist_w / histSize);

		// Вычисление гистограммы для текущего канала
		cv::calcHist(&channels[c], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

		// Нормализация гистограммы
		cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

		// Отрисовка линии интенсивности для гистограммы текущего канала
		for (int i = 0; i < histSize - 1; i++) {
			cv::line(histImage,
				cv::Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
				cv::Point(bin_w * (i + 1), hist_h - cvRound(hist.at<float>(i + 1))),
				cv::Scalar(0, 0, 0),
				1, cv::LINE_AA);
		}
	}

	return histImage;
}

bool autoContrast(cv::Mat& src, cv::Mat& dst, unsigned int quantileLow, unsigned int quantileHigh) {

	if (!src.data)
		return false;

	int rows = src.rows;
	int cols = src.cols;

	if (src.channels() == 1) {

		dst = src.clone();

		double max, min;
		cv::minMaxLoc(src, &min, &max, nullptr, nullptr);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {

				int pixel = src.at<uchar>(i, j);

				int contrastPixel = min + (pixel - quantileLow) *
					((max - min) / (quantileHigh - quantileLow));

				dst.at<uchar>(i, j) = contrastPixel;
			}
		}

		return true;
	}
	else {

		cv::Mat channels[3];
		cv::split(src, channels);

		for (auto& channel : channels) {

			double max, min;
			cv::minMaxLoc(channel, &min, &max, nullptr, nullptr);

			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {

					int pixel = channel.at<uchar>(i, j);

					int contrastPixel = min + (pixel - quantileLow) *
						((max - min) / (quantileHigh - quantileLow));

					channel.at<uchar>(i, j) = contrastPixel;
				}
			}
		}

		cv::merge(channels, 3, dst);

		return true;
	}

	return false;
}


int main(int argc, char** argv) {
	if (argc != 4) {
		std::cerr << "Usage: " << argv[0] << " <path> <quantileLow> <quantileHigh> " << std::endl;
		return 1;
	}

	std::string path = argv[1];
	unsigned int quantileLow = std::stoi(argv[2]);
	unsigned int quantileHigh = std::stoi(argv[3]);

	cv::Mat image = cv::imread(path, cv::IMREAD_ANYCOLOR);
	if (!image.data)
		return 0;

	cv::Mat originalImageWithHist;
	cv::Mat hist = getHist(image, 256, 256);
	cv::vconcat(image, hist, originalImageWithHist);

	cv::Mat contrastImage;
	bool result = autoContrast(image, contrastImage, quantileLow, quantileHigh);
	if (result) {
		cv::Mat contrastImageWithHist;
		hist = getHist(contrastImage, 256, 256);
		cv::vconcat(contrastImage, hist, contrastImageWithHist);

		cv::Mat result;
		cv::hconcat(originalImageWithHist, contrastImageWithHist, result);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
		cv::imshow("image", result);
		cv::waitKey(0);
		cv::destroyWindow("image");
	}

	return 0;
}


