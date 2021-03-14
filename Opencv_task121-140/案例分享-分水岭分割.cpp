#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/pill_002.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	// 去噪声，这步很重要
	Mat gray, binary, shifted;
	pyrMeanShiftFiltering(src, shifted, 21, 51);
	//imshow("shifted", shifted);

	cvtColor(shifted, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("binary", binary);

	// distance transform
	Mat dist;
	distanceTransform(binary, dist, DistanceTypes::DIST_L2, 3, CV_32F);
	normalize(dist, dist, 0, 1, NORM_MINMAX);
	//imshow("distance result", dist);

	// binary
	threshold(dist, dist, 0.4, 1, THRESH_BINARY);
	//imshow("distance binary", dist);

	// markers
	Mat dist_m;
	dist.convertTo(dist_m, CV_8U);
	vector<vector<Point>> contours;
	findContours(dist_m, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// create markers
	Mat markers = Mat::zeros(src.size(), CV_32SC1);
	for (size_t t = 0; t < contours.size(); t++) {
		drawContours(markers, contours, static_cast<int>(t), Scalar::all(static_cast<int>(t) + 1), -1);
	}
	circle(markers, Point(5, 5), 3, Scalar(255), -1);
	//imshow("markers", markers*10000);

	// 形态学操作 - 彩色图像，目的是去掉干扰，让结果更好
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(src, src, MORPH_ERODE, k);

	// 完成分水岭变换
	watershed(src, markers);
	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
	markers.convertTo(mark, CV_8UC1);
	bitwise_not(mark, mark, Mat());
	//imshow("watershed result", mark);

	// generate random color
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		int r = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int b = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	// 颜色填充与最终显示
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	int index = 0;
	for (int row = 0; row < markers.rows; row++) {
		for (int col = 0; col < markers.cols; col++) {
			index = markers.at<int>(row, col);
			if (index > 0 && index <= contours.size()) {
				dst.at<Vec3b>(row, col) = colors[index - 1];
			}
			else {
				dst.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
			}
		}
	}

	imshow("Final Result", dst);
	printf("number of objects : %d\n", contours.size());

	waitKey(0);
	return 0;
}