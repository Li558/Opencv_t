#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/test1.png");
	auto keypoint_detector = GFTTDetector::create(1000, 0.01, 1.0, 3, false, 0.04);
	vector<KeyPoint> kpts;
	keypoint_detector->detect(src, kpts);
	Mat result = src.clone();
	drawKeypoints(src, kpts, result, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("GFTT-Keypoint-Detect", result);
	//imwrite("D:/result.png", result);
	waitKey(0);
	return 0;
}
