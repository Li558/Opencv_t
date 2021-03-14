#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void find_known_object(Mat &box, Mat &box_scene);
int main(int argc, char** argv) {

	Mat box = imread("D:/images/box.bmp");
	Mat scene = imread("D:/images/scene.jpg");
	imshow("box image", box);
	imshow("scene image", scene);
	find_known_object(box, scene);

	//Mat gray;
	//cvtColor(src, gray, COLOR_BGR2GRAY);
	auto detector = SIFT::create();
	vector<KeyPoint> keypoints_box, keypoints_scene;
	Mat descriptor_box, descriptor_scene;
	detector->detectAndCompute(box, Mat(), keypoints_box, descriptor_box);
	detector->detectAndCompute(scene, Mat(), keypoints_scene, descriptor_scene);

	Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
	vector<DMatch> matches;
	matcher->match(descriptor_box, descriptor_scene, matches);
	Mat dst;
	drawMatches(box, keypoints_box, scene, keypoints_scene, matches, dst);
	imshow("match-demo", dst);


	waitKey(0);
	return 0;
}
