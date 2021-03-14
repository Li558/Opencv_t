#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
void open_demo();
void close_demo();
int main(int argc, char** argv)
{
	close_demo();
	open_demo();
	waitKey(0);
	return 0;
}

void open_demo() {
	Mat dresult, eresult;
	Mat src = imread("photo/fill.png");
	imshow("input2", src);

	// 二值图像
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("binary", binary);

	// 定义结构元素
	Mat se = getStructuringElement(MORPH_RECT, Size(20, 1), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, se);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarhy;
	findContours(binary, contours, hierarhy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	for (size_t t = 0; t < contours.size(); t++) {
		Rect roi = boundingRect(contours[t]);
		roi.y = roi.y - 10;
		roi.height = 12;
		rectangle(src, roi, Scalar(0, 0, 255), 1, 8, 0);
	}
	// 显示
	imshow("open demo", src);
}
//闭运算
void close_demo() {
	Mat src = imread("photo/morph3.png");
	namedWindow("input1", WINDOW_AUTOSIZE);
	imshow("input1", src);

	// 图像二值化
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// 闭操作
	// Mat se = getStructuringElement(MORPH_ELLIPSE, Size(15, 15), Point(-1, -1));
	Mat se = getStructuringElement(MORPH_RECT, Size(25, 25), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_CLOSE, se);
	//imwrite("D:/close.png", binary);
	imshow("close", binary);
}