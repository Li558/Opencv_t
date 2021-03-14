#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	// 加载图像
	Mat src = imread("photo/blob2.png");
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// 初始化参数设置
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 200;
	params.filterByArea = true;
	params.minArea = 100;
	params.filterByCircularity = true;
	params.minCircularity = 0.1;
	params.filterByConvexity = true;
	params.minConvexity = 0.87;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	// 创建BLOB Detetor
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// BLOB分析与显示
	Mat result;
	vector<KeyPoint> keypoints;
	detector->detect(gray, keypoints);
	drawKeypoints(src, keypoints, result, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("Blob Detection Demo", result);
	waitKey(0);
}