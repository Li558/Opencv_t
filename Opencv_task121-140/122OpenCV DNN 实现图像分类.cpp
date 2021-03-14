#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
/******************************************************
*
********************************************************/
using namespace cv;
using namespace cv::dnn;
using namespace std;

String labels_txt_file = "models/inception5h/imagenet_comp_graph_label_strings.txt";
String tf_pb_file = "models/inception5h/tensorflow_inception_graph.pb";
vector<String> readClassNames();
int main(int argc, char** argv) {
	Mat src = imread("photo/tyt.png");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	vector<String> labels = readClassNames();

	Mat rgb;
	cvtColor(src, rgb, COLOR_BGR2RGB);
	int w = 224;
	int h = 224;

	// ��������
	Net net = readNetFromTensorflow(tf_pb_file);
	if (net.empty()) {
		printf("read caffe model data failure...\n");
		return -1;
	}
	Mat inputBlob = blobFromImage(src, 1.0f, Size(224, 224), Scalar(), true, false);
	inputBlob -= 117.0; // ��ֵ

						// ִ��ͼ�����
	Mat prob;
	net.setInput(inputBlob, "input");
	prob = net.forward("softmax2");

	// �õ�����ܷ������
	Mat probMat = prob.reshape(1, 1);
	Point classNumber;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;
	printf("\n current image classification : %s, possible : %.2f", labels.at(classidx).c_str(), classProb);

	// ��ʾ�ı�
	putText(src, labels.at(classidx), Point(20, 20), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("Image Classification", src);
	//imwrite("D:/result.png", src);
	waitKey(0);
	return 0;
}

std::vector<String> readClassNames()
{
	std::vector<String> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}
