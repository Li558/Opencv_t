#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#define RATIO    0.4
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat box = imread("photo/box.png");
	Mat scene = imread("photo/box_in_scene.png");
	if (scene.empty()||box.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	imshow("input image", scene);
	imshow("input image1", box);
	vector<KeyPoint> keypoints_obj, keypoints_sence;
	Mat descriptors_box, descriptors_sence;
	Ptr<ORB> detector = ORB::create();
	detector->detectAndCompute(scene, Mat(), keypoints_sence, descriptors_sence);
	detector->detectAndCompute(box, Mat(), keypoints_obj, descriptors_box);

	vector<DMatch> matches;
	// 初始化flann匹配
	// Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create(); // default is bad, using local sensitive hash(LSH)
	Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 2));
	matcher->match(descriptors_box, descriptors_sence, matches);

	// 发现匹配
	vector<DMatch> goodMatches;
	printf("total match points : %d\n", matches.size());
	float maxdist = 0;
	for (unsigned int i = 0; i < matches.size(); ++i) {
		printf("dist : %.2f \n", matches[i].distance);
		maxdist = max(maxdist, matches[i].distance);
	}
	for (unsigned int i = 0; i < matches.size(); ++i) {
		if (matches[i].distance < maxdist*RATIO)
			goodMatches.push_back(matches[i]);
	}

	Mat dst;
	drawMatches(box, keypoints_obj, scene, keypoints_sence, goodMatches, dst);
	
	//resize(dst, dst, Size(0, 0), (0.5), (0.5), 1);
	imshow("output", dst);
	waitKey(0);
	return 0;
}
