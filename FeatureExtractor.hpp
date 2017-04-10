#ifndef _FEATURE_EXTRACTOR_HPP_
#define _FEATURE_EXTRACTOR_HPP_

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <caffe/net.hpp>
#include <caffe/caffe.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;

const int INPUT_IMAGE_WIDTH = 227;
const int INPUT_IMAGE_HEIGHT = 227;
const int NUM_CHANNELS = 3;
const std::string OUTPUT_BLOB_NAME = "fc10_merge";

class FeatureExtractor {
private:
	boost::shared_ptr<caffe::Net<float> > net;
	void OpenCV2Blob(const std::vector<cv::Mat>& imgs);
	void SetMean(const std::string& mean_file);
	cv::Mat mean_;
public:
 FeatureExtractor(const std::string& protoFilePath, 
				const std::string& modelFilePath, 
				const std::string& meanFilePath,
				bool useGpu, int deviceId);
	std::vector<float> extractFeatures(cv::Mat img);
	std::vector<float> extractFeatures2(cv::Mat img);
};

inline FeatureExtractor::FeatureExtractor(const std::string& protoFilePath, 
				const std::string& modelFilePath, 
				const std::string& meanFilePath,
				bool useGpu, int deviceId) {
	if (useGpu) {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(deviceId);
	} else {
		Caffe::set_mode(Caffe::CPU);
	}
	SetMean(meanFilePath);
	net.reset(new caffe::Net<float>(protoFilePath, caffe::TEST));
	net->CopyTrainedLayersFrom(modelFilePath);
}

inline std::vector<float> FeatureExtractor::extractFeatures(cv::Mat img) {
	CHECK(img.type() == CV_8UC3);
	cv::resize(img, img, cv::Size(INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), cv::INTER_AREA);
	img.convertTo(img, CV_32FC3);
	//std::cout << cv::mean(img)[0] << " " << cv::mean(img)[1] << " " << cv::mean(img)[2] << std::endl;
	//std::cout << mean_.at<cv::Vec3f>(0,0)[0] << " " << mean_.at<cv::Vec3f>(0,0)[1] << " " << mean_.at<cv::Vec3f>(0,0)[2] << std::endl;
	cv::subtract(img, mean_, img);
	std::vector<cv::Mat> imgs;
	imgs.push_back(img);
	OpenCV2Blob(imgs);
	const boost::shared_ptr<Blob<float> > output_blob = net->blob_by_name(OUTPUT_BLOB_NAME);
	net->ForwardPrefilled();
	int dim_blob = output_blob->count();
	const float* blob_data = output_blob->cpu_data() + output_blob->offset(0);
	std::vector<float> features;
	for (int i = 0; i < dim_blob; ++i) {
		features.push_back(blob_data[i]);
		std::cout << blob_data[i] << " ";
	}
	std::cout << std::endl;
	return features;
}

inline std::vector<float> FeatureExtractor::extractFeatures2(cv::Mat img) {
	CHECK(img.type() == CV_8UC3);
	cv::resize(img, img, cv::Size(INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), cv::INTER_AREA);
	img.convertTo(img, CV_32FC3);
	//std::cout << cv::mean(img)[0] << " " << cv::mean(img)[1] << " " << cv::mean(img)[2] << std::endl;
	//std::cout << mean_.at<cv::Vec3f>(0,0)[0] << " " << mean_.at<cv::Vec3f>(0,0)[1] << " " << mean_.at<cv::Vec3f>(0,0)[2] << std::endl;
	cv::subtract(img, mean_, img);
	std::vector<cv::Mat> imgs;
	imgs.push_back(img);
	OpenCV2Blob(imgs);
	std::vector<boost::shared_ptr<Blob<float> > > blobs = {
	(net->blob_by_name("fc9_BalancingElement")),
	(net->blob_by_name("fc9_ColorHarmony")),
	(net->blob_by_name("fc9_Content")),
	(net->blob_by_name("fc9_DoF")),
	(net->blob_by_name("fc9_Light")),
	(net->blob_by_name("fc9_MotionBlur")),
	(net->blob_by_name("fc9_Object")),
	(net->blob_by_name("fc9_Repetition")),
	(net->blob_by_name("fc9_RuleOfThirds")),
	(net->blob_by_name("fc9_Symmetry")),
	(net->blob_by_name("fc9_VividColor")) };
	net->ForwardPrefilled();

	std::vector<float> features;
	for (size_t i = 0; i < blobs.size(); ++i) {
		const float* blob_data = blobs[i]->cpu_data() + blobs[i]->offset(0);
		std::cout << blob_data[0] << " ";
		features.push_back(blob_data[0]);
	}
	std::cout << std::endl;
	return features;
}


inline void FeatureExtractor::OpenCV2Blob(const std::vector<cv::Mat>& imgs) {
    if (imgs.empty()) {
        throw std::string("Error: Image is invalid");
    }
    caffe::Blob<float>* input_layer = net->input_blobs()[0];
    float* input_data = input_layer->mutable_cpu_data();

    int img_number  = imgs.size();
    int img_channel = imgs[0].channels();
    int img_width   = imgs[0].cols;
    int img_height  = imgs[0].rows;
    int index = 0;
    for (int i = 0; i < img_number; i++) {
        for (int c = 0; c < img_channel; c++) {
            for (int h = 0; h < img_height; h++) {
                for (int w = 0; w < img_width; w++) {
					if (img_channel == 1) {
						*(input_data + index) = imgs[i].at<float>(h, w);
					} else if (img_channel == 3) {
							*(input_data + index) = imgs[i].at<cv::Vec<float, 3> >(h, w)[c];
					}
                    index++;
                }
            }
        }
    }
}

inline void FeatureExtractor::SetMean(const std::string& mean_file) {
	caffe::BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), NUM_CHANNELS)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float *data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < NUM_CHANNELS; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	//cv::Mat mean;
	cv::merge(channels, mean_);
	mean_ = mean_(cv::Range(14, 241), cv::Range(14, 241));
	//std::cout << mean_.cols << " " << mean_.rows << std::endl;

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	//cv::Scalar channel_mean = cv::mean(mean);
	//mean_ = cv::Mat(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, mean.type(), channel_mean);
}

#endif //_FEATURE_EXTRACTOR_HPP_