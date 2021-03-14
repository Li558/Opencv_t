#include <opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;
void process_frame(Mat &image, int opts);
RNG rng(12345);
int main(int argc, char** argv) {
	VideoCapture capture("photo/color_object.mp4");

	if (!capture.isOpened()) {
		printf("could not open camera...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("result", WINDOW_AUTOSIZE);

	int fps = capture.get(CAP_PROP_FPS);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int num_of_frames = capture.get(CAP_PROP_FRAME_COUNT);
	printf("frame width: %d, frame height: %d, FPS : %d \n", width, height, fps);

	Mat frame;
	while (true) {
		bool ret = capture.read(frame);
		imshow("input", frame);
		if (!ret) break;

		process_frame(frame, 0);
		imshow("result", frame);
		char c = waitKey(5);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}


void process_frame(Mat &image, int opts) {
	// Detector parameters
	int maxCorners = 100;
	double quality_level = 0.01;
	double minDistance = 0.04;

	// Detecting corners
	Mat gray, dst;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	vector<Point2f> corners;
	//Shi TomasiËã·¨µÄ½Çµã¼ì²â
	goodFeaturesToTrack(gray, corners, maxCorners, quality_level, minDistance, Mat(), 3, false);
	for (int i = 0; i < corners.size(); i++) {
		int b = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int r = rng.uniform(0, 256);
		circle(image, corners[i], 5, Scalar(b, g, r), 3, 8, 0);
	}
}
