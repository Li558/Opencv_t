#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int artc, char** argv) {
	Mat image = imread("photo/girl.jpg");
	if (image.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", image);

	Mat blured, dst;
	//高斯滤波
	GaussianBlur(image, blured, Size(3, 3), 0);
	imshow("image", image);
	//Laplacian 变换
	Laplacian(blured, dst, CV_32F, 1, 1.0, 127.0, BORDER_DEFAULT);
	//可实现图像增强等相关操作的快速运算
	convertScaleAbs(dst, dst);
	imshow("result", dst);

	waitKey(0);
	return 0;
}
