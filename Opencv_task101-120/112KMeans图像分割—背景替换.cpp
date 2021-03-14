#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/Dress-Man-2-icon.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	//namedWindow("input image", WINDOW_AUTOSIZE);
	//imshow("input image", src);

	int width = src.cols;
	int height = src.rows;
	int dims = src.channels();

	// 初始化定义
	int sampleCount = width * height;
	int clusterCount = 3;
	Mat labels;
	Mat centers;

	// RGB 数据转换到样本数据
	Mat sample_data = src.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// 运行K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	int index = labels.at<int>(0, 0);
	labels = labels.reshape(1, height);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int c = labels.at<int>(row, col);
			if (c == index) {
				mask.at<uchar>(row, col) = 255;
			}
		}
	}

	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	dilate(mask, mask, se);
	GaussianBlur(mask, mask, Size(5, 5), 0);
	Mat result = Mat::zeros(src.size(), CV_8UC3);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float w1 = mask.at<uchar>(row, col) / 255.0;
			Vec3b bgr = src.at<Vec3b>(row, col);
			bgr[0] = w1 * 255.0 +  bgr[0] * (1.0 - w1);
			bgr[1] = w1 * 255.0 +  bgr[1] * (1.0 - w1);
			bgr[2] = w1 * 255.0 +  bgr[2] * (1.0 - w1);
			result.at<Vec3b>(row, col) = bgr;
		}
	}
	imshow("KMeans-image-Demo", result);
	waitKey(0);
	return 0;
}