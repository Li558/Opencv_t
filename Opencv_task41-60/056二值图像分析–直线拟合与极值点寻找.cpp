#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{
	Mat src = imread("photo/twolines.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	// 去噪声与二值化
	Mat dst, gray, binary;
	//canny 轮廓提取
	Canny(src, binary, 80, 160, 3, false);
	imshow("binary", binary);
	//imwrite("D:/binary.png", binary);

	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	dilate(binary, binary, k);

	// 轮廓发现与绘制
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

	for (size_t t = 0; t < contours.size(); t++) {
		// 最大外接轮廓
		Rect rect = boundingRect(contours[t]);
		int m = max(rect.width, rect.height);
		if (m < 30) continue;
		// 直线拟合
		Vec4f oneline;
		fitLine(contours[t], oneline, DIST_L1, 0, 0.01, 0.01);
		float vx = oneline[0];
		float vy = oneline[1];
		float x0 = oneline[2];
		float y0 = oneline[3];

		// 直线参数斜率k与截矩b
		float k = vy / vx;
		float b = y0 - k * x0;

		// 寻找轮廓极值点
		int minx = 0, miny = 10000;
		int maxx = 0, maxy = 0;
		for (int i = 0; i < contours[t].size(); i++) {
			Point pt = contours[t][i];
			if (miny > pt.y) {
				miny = pt.y;
			}
			if (maxy < pt.y) {
				maxy = pt.y;
			}
		}
		maxx = (maxy - b) / k;
		minx = (miny - b) / k;
		line(src, Point(maxx, maxy), Point(minx, miny), Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("contours", src);

	waitKey(0);
	return 0;
}