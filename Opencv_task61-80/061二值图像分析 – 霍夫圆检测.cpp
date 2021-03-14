#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat gray;
	int dp = 2; // 在其它参数保持不变的情况下。dp的取值越高，越容易检测到圆，
	int min_radius = 20;
	int max_radius = 100;
	int minDist = 10;
	Mat src = imread("photo/coins.jpg");
	imshow("input", src);
	cvtColor(src, gray, COLOR_BGR2GRAY);
	//高斯滤波
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);
	vector<Vec3f> circles;
	//HoughCircles---函数可以利用霍夫变换算法检测出灰度图中的圆  不需要图像是二值的
	HoughCircles(gray, circles, HOUGH_GRADIENT, dp, minDist, 100, 100, min_radius, max_radius);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// 绘制圆
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
	namedWindow("circles", 1);
	imshow("circles", src);
	waitKey(0);
	return 0;
}