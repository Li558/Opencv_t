#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

string label_map = "models/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt";
string model = "models/tensorflow/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb";
string config = "models/tensorflow/faster_rcnn_resnet50_coco_2018_01_28/graph.pbtxt";

std::map<int, String> readLabelMaps();
int main(int argc, char** argv) {
	Mat src = imread("photo/persons.png");
	int width = src.cols;
	int height = src.rows;
	if (src.empty()) {
		printf("could not load image...\n");
		return 0;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	map<int, string> names = readLabelMaps();

	// 加载Faster-RCNN
	Net net = readNetFromTensorflow(model, config);
	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(), true, false);
	net.setInput(blob);

	// 预测
	Mat detection = net.forward();
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float threshold = 0.5;

	// 处理输出数据，绘制预测框与文本
	for (int row = 0; row < detectionMat.rows; row++) {
		float confidence = detectionMat.at<float>(row, 2);
		if (confidence > threshold) {

			// base zero
			int object_class = detectionMat.at<float>(row, 1) + 1;

			// predict box
			int left = detectionMat.at<float>(row, 3) * width;
			int top = detectionMat.at<float>(row, 4) * height;
			int right = detectionMat.at<float>(row, 5) * width;
			int bottom = detectionMat.at<float>(row, 6) * height;

			Rect rect;
			rect.x = left;
			rect.y = top;
			rect.width = (right - left);
			rect.height = (bottom - top);

			// render bounding box and label name
			rectangle(src, rect, Scalar(255, 0, 255), 4, 8, 0);
			map<int, string>::iterator it = names.find(object_class);
			printf("id : %d, display name : %s \n", object_class, (it->second).c_str());
			putText(src, (it->second).c_str(), Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
		}
	}
	imshow("faster-rcnn-demo", src);
	waitKey(0);
	return 0;
}

std::map<int, string> readLabelMaps()
{
	std::map<int, string> labelNames;
	std::ifstream fp(label_map);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	string one_line;
	string display_name;
	while (!fp.eof())
	{
		std::getline(fp, one_line);
		std::size_t found = one_line.find("id:");
		if (found != std::string::npos) {
			int index = found;
			string id = one_line.substr(index + 4, one_line.length() - index);

			std::getline(fp, display_name);
			std::size_t  found = display_name.find("display_name:");

			index = found + 15;
			string name = display_name.substr(index, display_name.length() - index);
			name = name.replace(name.length() - 1, name.length(), "");
			// printf("id : %d, name: %s \n", stoi(id.c_str()), name.c_str());
			labelNames[stoi(id)] = name;
		}
	}
	fp.close();
	return labelNames;
}