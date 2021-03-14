#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/lena.png");
	if (src.empty()) {
		printf("could not load image..\n");
		return -1;
	}
	imshow("input", src);
	HOGDescriptor hog;
	vector<float> features;
	hog.compute(src, features, Size(8, 8), Size(0, 0));

	printf("feature sum size :%d \n", features.size());
	for (int i = 0; i < features.size(); i++) {
		printf("v: %.2f\n ", features[i]);
	}
	imshow("result", src);
	waitKey(0);
	return 0;
}