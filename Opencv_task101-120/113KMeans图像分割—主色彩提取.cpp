#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/toux.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	int width = src.cols;
	int height = src.rows;
	int dims = src.channels();

	// 初始化定义
	int sampleCount = width * height;
	int clusterCount = 4;
	Mat labels;
	Mat centers;

	// RGB 数据转换到样本数据
	Mat sample_data = src.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// 运行K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	Mat card = Mat::zeros(Size(width, 50), CV_8UC3);
	vector<float> clusters(clusterCount);
	for (int i = 0; i < labels.rows; i++) {
		clusters[labels.at<int>(i, 0)]++;
	}
	for (int i = 0; i < clusters.size(); i++) {
		clusters[i] = clusters[i] / sampleCount;
	}
	int x_offset = 0;
	for (int x = 0; x < clusterCount; x++) {
		Rect rect;
		rect.x = x_offset;
		rect.y = 0;
		rect.height = 50;
		rect.width = round(clusters[x] * width);
		x_offset += rect.width;
		int b = centers.at<float>(x, 0);
		int g = centers.at<float>(x, 1);
		int r = centers.at<float>(x, 2);
		rectangle(card, rect, Scalar(b, g, r), -1, 8, 0);
	}

	imshow("Image Color Card", card);
	waitKey(0);
	return 0;
}