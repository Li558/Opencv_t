#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	//Mat src = imread("photo/master.jpg");

	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		printf("could not open camera...\n");
		return -1;
	}
	while (1)
	{
		Mat src;
		cap >> src;
		namedWindow("input", WINDOW_AUTOSIZE);
		imshow("input", src);

		Mat mask = Mat::zeros(src.size(), CV_8UC1);
		Rect rect(180, 20, 180, 220);
		Mat bgdmodel = Mat::zeros(1, 65, CV_64FC1);
		Mat fgdmodel = Mat::zeros(1, 65, CV_64FC1);
		grabCut(src, mask, rect, bgdmodel, fgdmodel, 5, GC_INIT_WITH_RECT);
		Mat result;
		for (int row = 0; row < mask.rows; row++) {
			for (int col = 0; col < mask.cols; col++) {
				int pv = mask.at<uchar>(row, col);
				if (pv == 1 || pv == 3) {
					mask.at<uchar>(row, col) = 255;
				}
				else {
					mask.at<uchar>(row, col) = 0;
				}
			}
		}
		bitwise_and(src, src, result, mask);
		imshow("grabcut result", result);
		waitKey(30);
	}
	return 0;
}