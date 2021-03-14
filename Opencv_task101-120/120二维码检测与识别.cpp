#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/qrcode.jpg");
	if (src.empty())
		return -1;
	imshow("image", src);
	Mat gray, qrcode_roi;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	QRCodeDetector qrcode_detector;
	vector<Point> pts;
	string detect_info;
	bool det_result = qrcode_detector.detect(gray, pts);
	if (det_result) {
		detect_info = qrcode_detector.decode(gray, pts, qrcode_roi);
	}
	vector< vector<Point> > contours;
	contours.push_back(pts);
	drawContours(src, contours, 0, Scalar(0, 0, 255), 2);
	putText(src, detect_info.c_str(), Point(20, 200), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	printf("qrcode info %s \n", detect_info.c_str());
	imshow("result", src);
	waitKey(0);
	return 0;
}