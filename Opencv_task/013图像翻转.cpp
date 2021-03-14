#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("girl.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	imshow("input", src);

	Mat dst;
	// X Flip µ¹Ó°
	flip(src, dst, 0);
	imshow("x-flip", dst);

	// Y Flip ¾µÏñ
	flip(src, dst, 1);
	imshow("y-flip", dst);

	// XY Flip ¶Ô½Ç
	flip(src, dst, -1);
	imshow("xy-flip", dst);

	waitKey(0);
	return 0;
}

