#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv ;
using namespace std;

int main(int argc, char** argv) {
	Mat box = imread("photo/box.bmp");
	Mat box_in_sence = imread("photo/scene.jpg");

	if (box.empty()|| box_in_sence.empty()) {
		printf("could not load image...\n");
		return -1;
	}

	// 创建ORB
	auto orb_detector = ORB::create();
	vector<KeyPoint> kpts_01, kpts_02;
	Mat descriptors1, descriptors2;
	orb_detector->detectAndCompute(box, Mat(), kpts_01, descriptors1);
	orb_detector->detectAndCompute(box_in_sence, Mat(), kpts_02, descriptors2);

	// 定义描述子匹配 - 暴力匹配
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	std::vector< DMatch > matches;
	matcher->match(descriptors1, descriptors2, matches);

	// 绘制匹配
	Mat img_matches;
	drawMatches(box, kpts_01, box_in_sence, kpts_02, matches, img_matches);
	imshow("ORB-Matches", img_matches);
	//imwrite("D:/result.png", img_matches);

	waitKey(0);
	return 0;
}
