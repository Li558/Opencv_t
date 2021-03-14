#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
RNG rng(50000);
int main(int argc, const char *argv[])
{
	Mat src = imread("photo/contours.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	// 去噪声与二值化
	Mat dst, gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary", binary);
	//imwrite("D:/binary.png", binary);

	// 轮廓发现与绘制
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	Scalar color = Scalar(255, 0, 0);
	for (size_t t = 0; t < contours.size(); t++) {
		RotatedRect rrt = minAreaRect(contours[t]);
		Point2f cpt = rrt.center;
		circle(src, cpt, 2, Scalar(0, 255, 0), 2, 8, 0);

		Mat result;
		//approxPolyDP----一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合
		approxPolyDP(contours[t], result, 4, true);
		printf("corners : %d\n", result.rows);
		if (result.rows == 6) {
			putText(src, "poly", cpt, FONT_HERSHEY_SIMPLEX, .7, color, 1, 8);
		}
		if (result.rows == 3) {
			putText(src, "triangle", cpt, FONT_HERSHEY_SIMPLEX, .7, color, 1, 8);
		}
		if (result.rows == 4) {
			putText(src, "rectangle", cpt, FONT_HERSHEY_SIMPLEX, .7, color, 1, 8);
		}
		if (result.rows > 10) {
			putText(src, "circle", cpt, FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 255, 0), 1, 8);
		}
	}
	imshow("contours", src);

	waitKey(0);
	return 0;
}