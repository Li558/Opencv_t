#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("photo/Dress-Man-2-icon.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", src);

	Scalar colorTab[] = {
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 0, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255)
	};

	int width = src.cols;
	int height = src.rows;
	int dims = src.channels();

	// 初始化定义
	int sampleCount = width * height;
	int clusterCount = 3;
	Mat labels;
	Mat centers;

	// RGB 数据转换到样本数据
	Mat sample_data = src.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// 运行K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	// 显示图像分割结果
	int index = 0;
	Mat result = Mat::zeros(src.size(), src.type());
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row * width + col;
			int label = labels.at<int>(index, 0);
			result.at<Vec3b>(row, col)[0] = colorTab[label][0];
			result.at<Vec3b>(row, col)[1] = colorTab[label][1];
			result.at<Vec3b>(row, col)[2] = colorTab[label][2];
		}
	}

	imshow("KMeans-image-Demo", result);
	waitKey(0);
	return 0;
}