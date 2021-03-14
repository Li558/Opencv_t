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
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// Detecting corners
	Mat gray, dst;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	//cornerHarris����----������Ҫ���ڼ��ͼ��Ĺ���˹��Harris���ǵ��⣬���жϳ�ĳһ���ǲ���ͼ��Ľǵ�
	cornerHarris(gray, dst, blockSize, apertureSize, k);
	// Normalizing
	Mat dst_norm = Mat::zeros(dst.size(), dst.type());
	//ͼ���һ��--- ��һ������Ҫ����Ҫ��������ݾ��������ͨ��ĳ���㷨������������Ҫ��һ����Χ��
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX);
	//ͼ����ǿ
	convertScaleAbs(dst_norm, dst_norm);
	// Drawing a circle around corners
	for (int row = 0; row < dst_norm.rows; row++) {
		for (int col = 0; col < dst_norm.cols; col++) {
			int rsp = dst_norm.at<uchar>(row, col);
			if (rsp > 150) {
				int b = rng.uniform(0, 256);
				int g = rng.uniform(0, 256);
				int r = rng.uniform(0, 256);
				circle(image, Point(row, col), 5, Scalar(b, g, r), 2);
			}
		}
	}
}
