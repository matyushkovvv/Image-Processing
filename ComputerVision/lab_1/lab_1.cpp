#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

cv::Mat gamma_correct(cv::Mat img, float gamma) {
    cv::Size img_dim = img.size();
    int tile_width = img_dim.width / 256.;
    for (int grad = 0; grad <= 255; grad++) {
        uchar color = pow(float(grad) / 255., gamma) * 255;
        for (int intile_coord = 0; intile_coord < tile_width; intile_coord++) {
            img.col(tile_width * grad + intile_coord).setTo(color);
        }
    }
    return img;
}

void generate_gradient_image(int width, int height, double gamma, const std::string& filename) {
    cv::Mat img1(height, width * 256, CV_8UC1);
    cv::Mat img2(height, width * 256, CV_8UC1);

    img1 = gamma_correct(img1, 1.0);
    img2 = gamma_correct(img2, gamma);

    cv::Mat final_img;
    cv::vconcat(img1, img2, final_img);

    if (!filename.empty()) {
        cv::imwrite(filename, final_img);
        std::cout << "Image saved as " << filename << std::endl;
    }
    else {
        cv::imshow("Generated Image", final_img);
        cv::waitKey(0);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <gamma> <filename>" << std::endl;
        return 1;
    }

    int width = std::atoi(argv[1]);
    int height = std::atoi(argv[2]);
    double gamma = std::atof(argv[3]);
    std::string filename = argv[4];

    if (width <= 0 || height <= 0 || gamma <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return 1;
    }

    generate_gradient_image(width, height, gamma, filename);

    return 0;
}
