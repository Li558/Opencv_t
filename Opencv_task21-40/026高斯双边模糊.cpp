#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int artc, char** argv) {
	Mat src = imread("girl.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	Mat dst;
	//¸ßË¹Ë«±ßÂË²¨
	bilateralFilter(src, dst, 0, 100, 10, 4);
	imshow("result", dst);

	waitKey(0);
	return 0;
}
