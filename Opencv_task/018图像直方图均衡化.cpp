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
	Mat gray, dst;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	imshow("input", gray);
	equalizeHist(gray, dst);
	imshow("eq", dst);

	waitKey(0);
	return 0;
}



