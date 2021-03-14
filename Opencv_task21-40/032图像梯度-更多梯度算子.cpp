#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int artc, char** argv) {
	Mat src = imread("photo/girl.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	Mat robert_x = (Mat_<int>(2, 2) << 1, 0, 0, -1);
	Mat robert_y = (Mat_<int>(2, 2) << 0, -1, 1, 0);

	Mat prewitt_x = (Mat_<char>(3, 3) << -1, 0, 1,
		-1, 0, 1,
		-1, 0, 1);
	Mat prewitt_y = (Mat_<char>(3, 3) << -1, -1, -1,
		0, 0, 0,
		1, 1, 1);

	Mat robert_grad_x, robert_grad_y, prewitt_grad_x, prewitt_grad_y;
	//图像卷积运算函数
	filter2D(src, robert_grad_x, CV_16S, robert_x);
	filter2D(src, robert_grad_y, CV_16S, robert_y);
	//可实现图像增强等相关操作的快速运算
	convertScaleAbs(robert_grad_x, robert_grad_x);
	convertScaleAbs(robert_grad_y, robert_grad_y);
	//图像卷积运算函数
	filter2D(src, prewitt_grad_x, CV_32F, prewitt_x);
	filter2D(src, prewitt_grad_y, CV_32F, prewitt_y);
	//可实现图像增强等相关操作的快速运算
	convertScaleAbs(prewitt_grad_x, prewitt_grad_x);
	convertScaleAbs(prewitt_grad_y, prewitt_grad_y);
	printf("image gradient...");

	imshow("robert x", robert_grad_x);
	imshow("robert y", robert_grad_y);
	imshow("prewitt x", prewitt_grad_x);
	imshow("prewitt y", prewitt_grad_y);

	waitKey(0);
	return 0;
}
