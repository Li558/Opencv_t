#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int main(int argc, char** argv) {
	VideoCapture capture;
	capture.open("photo/vtest.avi");
	Mat preFrame, preGray;
	capture.read(preFrame);
	cvtColor(preFrame, preGray, COLOR_BGR2GRAY);
	Mat hsv = Mat::zeros(preFrame.size(), preFrame.type());
	Mat frame, gray;
	Mat_<Point2f> flow;
	vector<Mat> mv;
	split(hsv, mv);
	Mat mag = Mat::zeros(hsv.size(), CV_32FC1);
	Mat ang = Mat::zeros(hsv.size(), CV_32FC1);
	Mat xpts = Mat::zeros(hsv.size(), CV_32FC1);
	Mat ypts = Mat::zeros(hsv.size(), CV_32FC1);
	while (true) {
		bool ret = capture.read(frame);
		if (!ret) break;
		imshow("frame", frame);
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		/*
		1:calcOpticalFlowPyrLK---ͨ��������Lucas-Kanade ������������ĳЩ�㼯�Ĺ�����ϡ�������
		2:calcOpticalFlowFarneback---��Gunnar Farneback ���㷨������ܹ�������ͼ�����������ص�Ĺ��������������
		3:CalcOpticalFlowBM---ͨ����ƥ��ķ������������
		4:CalcOpticalFlowHS---��Horn-Schunck ���㷨������ܹ���
		5:calcOpticalFlowSF---�򵥹�����
		*/
		calcOpticalFlowFarneback(preGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		for (int row = 0; row < flow.rows; row++)
		{
			for (int col = 0; col < flow.cols; col++)
			{
				const Point2f& flow_xy = flow.at<Point2f>(row, col);
				xpts.at<float>(row, col) = flow_xy.x;
				ypts.at<float>(row, col) = flow_xy.y;
			}
		}
		cartToPolar(xpts, ypts, mag, ang);
		ang = ang * 180.0 / CV_PI / 2.0;
		normalize(mag, mag, 0, 255, NORM_MINMAX);
		convertScaleAbs(mag, mag);
		convertScaleAbs(ang, ang);
		mv[0] = ang;
		mv[1] = Scalar(255);
		mv[2] = mag;
		merge(mv, hsv);
		Mat bgr;
		cvtColor(hsv, bgr, COLOR_HSV2BGR);
		imshow("result", bgr);
		int ch = waitKey(5);
		if (ch == 27) {
			break;
		}
	}
	waitKey(0);
	return 0;
}