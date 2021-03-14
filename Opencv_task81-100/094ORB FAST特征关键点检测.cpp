#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/lady.jpg");
	auto orb_detector = ORB::create(1000);
	vector<KeyPoint> kpts;
	//ORB������ȡ
	orb_detector->detect(src, kpts);
	Mat result = src.clone();
	//���ƹؼ���
	drawKeypoints(src, kpts, result, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB-detector", result);
	//imwrite("D:/result.png", result);
	waitKey(0);
	return 0;
}
