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

	Mat edges;
	//Canny Ëã×Ó±ßÔµ¼ì²â
	Canny(src, edges, 100, 300, 3, false);
	imshow("edge image", edges);

	waitKey(0);
	return 0;
}
