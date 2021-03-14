#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{
	Mat src = imread("photo/girl.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	// 去噪声与二值化
	Mat dst, gray, binary;
	// 彩色图像分割  滤波
	pyrMeanShiftFiltering(src, dst, 10, 100);
	imshow("dst", dst);
	// GaussianBlur(src, dst, Size(3, 3), 0, 0);
	cvtColor(dst, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary", binary);
	//imwrite("D:/binary.png", binary);

	waitKey(0);
	return 0;
}