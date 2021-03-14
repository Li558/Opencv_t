#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{
	Mat src = imread("photo/hand.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	// 二值化
	Mat dst, gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// 删除干扰块
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	//形态学变换 morphologyEx函数 (开运算，闭运算，顶帽，黑帽)
	/*
	MORPH_OPEN – 开运算（Opening operation）
	MORPH_CLOSE – 闭运算（Closing operation）
	MORPH_GRADIENT - 形态学梯度（Morphological gradient）
	MORPH_TOPHAT - 顶帽（Top hat）
	MORPH_BLACKHAT - 黑帽（Black hat）
	*/
	morphologyEx(binary, binary, MORPH_OPEN, k);
	imshow("binary", binary);

	// 轮廓发现与绘制
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	for (size_t t = 0; t < contours.size(); t++) {
		vector<Point> hull;
		convexHull(contours[t], hull);
		bool isHull = isContourConvex(contours[t]);
		printf("test convex of the contours %s", isHull ? "Y" : "N");
		int len = hull.size();
		for (int i = 0; i < hull.size(); i++) {
			circle(src, hull[i], 4, Scalar(255, 0, 0), 2, 8, 0);
			line(src, hull[i%len], hull[(i + 1) % len], Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	imshow("contours", src);

	waitKey(0);
	return 0;
}