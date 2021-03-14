#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/Dress-Man-2-icon.jpg");
	Mat background = imread("photo/2.jpg");
	if (src.empty())
	{
		printf("could not load image...\n");
		return 0;
	}
	resize(background, background, Size(800, 800));
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	imshow("background", background);
	int height = src.rows;
	int width = src.cols;
	int ch = src.channels();
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	Rect rect(53, 12, width - 100, height - 12);
	Mat bgdmodel = Mat::zeros(1, 65, CV_64FC1);
	Mat fgdmodel = Mat::zeros(1, 65, CV_64FC1);
	grabCut(src, mask, rect, bgdmodel, fgdmodel, 5, GC_INIT_WITH_RECT);
	Mat result;
	for (int row = 0; row < mask.rows; row++) {
		for (int col = 0; col < mask.cols; col++) {
			int pv = mask.at<uchar>(row, col);
			if (pv == 1 || pv == 3) {
				mask.at<uchar>(row, col) = 255;
			}
			else {
				mask.at<uchar>(row, col) = 0;
			}
		}
	}

	Mat mask2;
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	dilate(mask, mask2, se, Point(-1, -1), 1, 0);
	imshow("mask", mask2);
	GaussianBlur(background, background, Size(0, 0), 15);
	 result = Mat::zeros(src.size(), CV_8UC3);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float w1 = mask2.at<uchar>(row, col) / 255.0;
			Vec3b bgr = src.at<Vec3b>(row, col);
			Vec3b bgr1 = background.at<Vec3b>(row, col);

			bgr[0] = (1.0 - w1) * bgr1[0] + w1 * bgr[0];
			bgr[1] = (1.0 - w1) * bgr1[1] + w1 * bgr[1];
			bgr[2] = (1.0 - w1) * bgr1[2] + w1 * bgr[2];
			result.at<Vec3b>(row, col) = bgr;
		}
	}
	imshow("result", result);

	waitKey(0);
	return 0;
}