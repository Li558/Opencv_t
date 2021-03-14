#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/pedestrian_02.png");
	if (src.empty()) {
		printf("could not load image..\n");
		return -1;
	}
	imshow("input", src);
	HOGDescriptor *hog = new HOGDescriptor();
	hog->setSVMDetector(hog->getDefaultPeopleDetector());
	vector<Rect> objects;
	hog->detectMultiScale(src, objects, 0.0, Size(4, 4), Size(8, 8), 1.25);
	for (int i = 0; i < objects.size(); i++) {
		rectangle(src, objects[i], Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("result", src);
	waitKey(0);
	return 0;
}