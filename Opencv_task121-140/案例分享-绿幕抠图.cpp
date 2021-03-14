#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat background_01;
Mat background_02;
Mat replace_and_blend(Mat &frame, Mat &mask);
Mat grabcut_greeen(Mat &image);
int main(int argc, char** argv) {

	background_01 = imread("photo/twolines.png");
	background_02 = imread("photo/twolines.png");
	VideoCapture capture;
	capture.open("photo/01.mp4");
	if (!capture.isOpened()) {
		printf("could not load video file...\n");
		return -1;
	}
	VideoCapture captureTwo;
	captureTwo.open("photo/dushuhu_02.mp4");


	const char* title = "input video";
	const char* matting_title = "video matting result";
	namedWindow(title,WINDOW_AUTOSIZE);
	namedWindow(matting_title, WINDOW_AUTOSIZE);
	Mat frame, hsv, mask;
	Mat frame2;
	int count = 0;
	Size S = Size((int)capture.get(CAP_PROP_FRAME_WIDTH),
		(int)capture.get(CAP_PROP_FRAME_HEIGHT));
	while (capture.read(frame)) {
		if (captureTwo.read(frame2)) {
			resize(frame2, background_01, frame.size());
		}
		else {
			background_02.copyTo(background_01);
		}
		cvtColor(frame, hsv, COLOR_BGR2HSV);
		inRange(hsv, Scalar(35, 43, 46), Scalar(155, 255, 255), mask);

		// 形态学腐蚀操作
		//Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		//morphologyEx(mask, mask, MORPH_CLOSE, k);
		//erode(mask, mask, k);
		GaussianBlur(mask, mask, Size(3, 3), 0, 0);

		Mat result = replace_and_blend(frame, mask);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
		imshow(title, frame);
		imshow(matting_title, result);
	}

	waitKey(0);
	return 0;
}

Mat grabcut_greeen(Mat &image) {
	Mat left = imread("photo/tyt.png");
	resize(left, background_01, image.size());
	Mat hsv, mask;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(35, 43, 46), Scalar(155, 255, 255), mask);

	// 形态学腐蚀操作
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(mask, mask, MORPH_CLOSE, k);
	erode(mask, mask, k);
	GaussianBlur(mask, mask, Size(3, 3), 0, 0);

	Mat result = replace_and_blend(image, mask);
	imshow("result", result);
	return result;
}

Mat replace_and_blend(Mat &frame, Mat &mask) {
	Mat result = Mat::zeros(frame.size(), frame.type());
	int h = frame.rows;
	int w = frame.cols;
	int dims = frame.channels();

	// replace and blend
	int m = 0;
	double wt = 0.0;
	int r = 0, g = 0, b = 0;

	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;

	for (int row = 0; row < h; row++) {
		uchar* current = frame.ptr<uchar>(row);
		uchar* bgrow = background_01.ptr<uchar>(row);
		uchar* maskrow = mask.ptr<uchar>(row);
		uchar* targetrow = result.ptr<uchar>(row);
		for (int col = 0; col < w; col++) {
			m = *maskrow++;
			if (m == 255) { // 背景
				*targetrow++ = *bgrow++;
				*targetrow++ = *bgrow++;
				*targetrow++ = *bgrow++;
				current += 3;
			}
			else if (m == 0) {
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				bgrow += 3;
			}
			else {
				// 背景像素
				b1 = *bgrow++;
				g1 = *bgrow++;
				r1 = *bgrow++;

				// 目标前景像素
				b2 = *current++;
				g2 = *current++;
				r2 = *current++;

				// 混合权重
				wt = m / 255.0;

				// 混合
				b = b1 * wt + b2 * (1.0 - wt);
				g = g1 * wt + g2 * (1.0 - wt);
				r = r1 * wt + r2 * (1.0 - wt);

				*targetrow++ = b;
				*targetrow++ = g;
				*targetrow++ = r;
			}
		}
	}

	return result;
}