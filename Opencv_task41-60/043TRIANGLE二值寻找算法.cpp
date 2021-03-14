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

	// 自动阈值寻找与二值化
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	double t = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_TRIANGLE);
	printf("threshold : %.2f\n", t);


	imshow("binary", binary);
	//imwrite("D:/binary.png", binary);

	waitKey(0);
	return 0;
}