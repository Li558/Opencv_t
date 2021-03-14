#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main(int argc, char** argv) {
	string bin_model = "models/googlenet/bvlc_googlenet.caffemodel";
	string protxt = "models/googlenet/bvlc_googlenet.prototxt";

	// load CNN model
	Net net = dnn::readNet(bin_model, protxt);

	// 获取各层信息
	vector<String> layer_names = net.getLayerNames();
	for (int i = 0; i < layer_names.size(); i++) {
		int id = net.getLayerId(layer_names[i]);
		auto layer = net.getLayer(id);
		printf("layer id:%d, type: %s, name:%s \n", id, layer->type.c_str(), layer->name.c_str());
	}
	return 0;
}