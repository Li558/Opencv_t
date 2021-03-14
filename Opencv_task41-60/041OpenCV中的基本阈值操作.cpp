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

	// 转换为灰度图像
	int T = 127;
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	for (int i = 0; i < 5; i++) {
		//二值化阈值处理
		threshold(gray, binary, T, 255, i);
		imshow(format("binary_%d", i), binary);
		//imwrite(format("D:/binary_%d.png", i), binary);  
	}
	waitKey(0);
	return 0;
}