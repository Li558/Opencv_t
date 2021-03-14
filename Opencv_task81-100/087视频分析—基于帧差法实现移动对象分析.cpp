#include <opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;
RNG rng(12345);
int main(int argc, char** argv) {
	VideoCapture capture("photo/bike.avi");

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

	Mat preFrame, preGray;
	capture.read(preFrame);
	cvtColor(preFrame, preGray, COLOR_BGR2GRAY);
	GaussianBlur(preGray, preGray, Size(0, 0), 15);
	Mat binary;
	Mat frame, gray;
	Mat k = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
	while (true) {
		bool ret = capture.read(frame);
		if (!ret) break;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, gray, Size(0, 0), 15);
		//œ‡ºı
		subtract(gray, preGray, binary);
		threshold(binary, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
		//ø™‘ÀÀ„
		morphologyEx(binary, binary, MORPH_OPEN, k);
		imshow("input", frame);
		imshow("result", binary);

		gray.copyTo(preGray);
		char c = waitKey(5);
		if (c == 27) {
			break;
		}
	}

	waitKey(0);
	return 0;
}
