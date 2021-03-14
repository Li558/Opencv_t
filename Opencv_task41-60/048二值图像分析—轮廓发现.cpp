#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{
	Mat src = imread("photo/coins.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	// 去噪声与二值化
	Mat dst, gray, binary;
	GaussianBlur(src, dst, Size(3, 3), 0, 0);
	cvtColor(dst, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary", binary);
	//imwrite("D:/binary.png", binary);

	// 轮廓发现与绘制
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//轮廓提取
	findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	for (size_t t = 0; t < contours.size(); t++) {
		//绘制轮廓
		drawContours(src, contours, t, Scalar(0, 0, 255), 2, 8);
	}
	imshow("contours", src);

	waitKey(0);
	return 0;
}