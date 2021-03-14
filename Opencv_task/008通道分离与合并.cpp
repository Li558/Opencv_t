#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{
	Mat src = imread("girl.jpg");
	
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);

	vector<Mat> mv;
	Mat dst1, dst2, dst3;
	// 蓝色通道为零
	//split(通道分离)
	split(src, mv);
	mv[0] = Scalar(0);
	//merge(通道合并)
	merge(mv, dst1);
	imshow("output1", dst1);

	// 绿色通道为零
	split(src, mv);
	mv[1] = Scalar(0);
	merge(mv, dst2);
	imshow("output2", dst2);

	// 红色通道为零
	split(src, mv);
	mv[2] = Scalar(0);
	merge(mv, dst3);
	imshow("output3", dst3);

	waitKey();
	return 0;
}