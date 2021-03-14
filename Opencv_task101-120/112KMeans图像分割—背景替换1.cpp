#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	Mat src = imread("photo/Dress-Man-2-icon.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	imshow("原图", src);

	// 1.将二维图像数据线性化
	Mat data;
	for (int i = 0; i < src.rows; i++)     //像素点线性排列
		for (int j = 0; j < src.cols; j++)
		{
			Vec3b point = src.at<Vec3b>(i, j);
			Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			data.push_back(tmp);
		}

	// 2.使用K-means聚类；分离出背景色
	int numCluster = 4;
	Mat labels;
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, numCluster, labels, criteria, 3, KMEANS_PP_CENTERS);

	// 3.背景与人物二值化
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	int index = src.rows * 2 + 2;  //获取点（2，2）作为背景色
	int cindex = labels.at<int>(index);
	/*  提取背景特征 */
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			index = row * src.cols + col;
			int label = labels.at<int>(index);
			if (label == cindex) { // 背景
				mask.at<uchar>(row, col) = 0;
			}
			else {
				mask.at<uchar>(row, col) = 255;
			}
		}
	}
	//imshow("mask", mask);

	// 4.腐蚀 + 高斯模糊：图像与背景交汇处高斯模糊化
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	erode(mask, mask, k);
	//imshow("erode-mask", mask);
	GaussianBlur(mask, mask, Size(3, 3), 0, 0);
	//imshow("Blur Mask", mask);

	// 5.更换背景色以及交汇处融合处理
	RNG rng(12345);
	Vec3b color;  //设置的背景色
	color[0] = 0;//rng.uniform(0, 255);
	color[1] = 0;// rng.uniform(0, 255);
	color[2] = 0;// rng.uniform(0, 255);
	Mat result(src.size(), src.type());

	double w = 0.0;   //融合权重
	int b = 0, g = 0, r = 0;
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;

	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			int m = mask.at<uchar>(row, col);
			if (m == 255) {
				result.at<Vec3b>(row, col) = src.at<Vec3b>(row, col); // 前景
			}
			else if (m == 0) {
				result.at<Vec3b>(row, col) = color; // 背景
			}
			else {/* 融合处理部分 */
				w = m / 255.0;
				b1 = src.at<Vec3b>(row, col)[0];
				g1 = src.at<Vec3b>(row, col)[1];
				r1 = src.at<Vec3b>(row, col)[2];

				b2 = color[0];
				g2 = color[1];
				r2 = color[2];

				b = b1 * w + b2 * (1.0 - w);
				g = g1 * w + g2 * (1.0 - w);
				r = r1 * w + r2 * (1.0 - w);

				result.at<Vec3b>(row, col)[0] = b;
				result.at<Vec3b>(row, col)[1] = g;
				result.at<Vec3b>(row, col)[2] = r;
			}
		}
	}
	imshow("背景替换", result);

	waitKey(0);
	return 0;
}