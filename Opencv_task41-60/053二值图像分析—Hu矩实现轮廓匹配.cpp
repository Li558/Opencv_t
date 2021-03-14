#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void contours_info(Mat &image, vector<vector<Point>> &pts);
int main(int argc, char** argv) {
	Mat src = imread("D:/images/abc.png");
	imshow("input", src);
	Mat src2 = imread("D:/images/a5.png");
	imshow("input2", src2);

	// 轮廓提取
	vector<vector<Point>> contours1;
	vector<vector<Point>> contours2;
	contours_info(src, contours1);
	contours_info(src2, contours2);
	// hu矩计算
	//moments()函数用于计算中心矩
	Moments mm2 = moments(contours2[0]);
	Mat hu2;
	//HuMoments--用于由中心矩计算Hu矩
	HuMoments(mm2, hu2);
	// 轮廓匹配
	for (size_t t = 0; t < contours1.size(); t++) {
		Moments mm = moments(contours1[t]);
		Mat hum;
		HuMoments(mm, hum);
		double dist = matchShapes(hum, hu2, CONTOURS_MATCH_I1, 0);
		printf("contour match distance : %.2f\n", dist);
		if (dist < 1) {
			printf("draw it \n");
			drawContours(src, contours1, t, Scalar(0, 0, 255), 2, 8);
		}
	}
	imshow("match result", src);
	waitKey(0);
	return 0;
}

void contours_info(Mat &image, vector<vector<Point>> &contours) {
	Mat gray, binary;
	vector<Vec4i> hierarchy;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
}