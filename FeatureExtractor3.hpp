#ifndef _FEATURE_EXTRACTOR3_HPP_
#define _FEATURE_EXTRACTOR3_HPP_

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

class FeatureExtractor3 {
private:
	boost::shared_ptr<caffe::Net<float> > net;
	void OpenCV2Blob(const std::vector<cv::Mat>& imgs);
	void SetMean(const std::string& mean_file);
	cv::Mat mean_;
public:
	FeatureExtractor3(const std::string& protoFilePath, 
				const std::string& modelFilePath, 
				const std::string& meanFilePath,
				bool useGpu, int deviceId);
	std::vector<float> extractFeatures(cv::Mat img);

};

inline FeatureExtractor3::FeatureExtractor3(const std::string& protoFilePath, 
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

inline std::vector<float> FeatureExtractor3::extractFeatures(cv::Mat img) {
	CHECK(img.type() == CV_8UC3);
	cv::resize(img, img, cv::Size(224, 224), cv::INTER_AREA);
	img.convertTo(img, CV_32FC3);
	cv::subtract(img, mean_, img);
	std::vector<cv::Mat> imgs;
	imgs.push_back(img);
	OpenCV2Blob(imgs);
	const boost::shared_ptr<Blob<float> > output_blob = net->blob_by_name("fc7");
	net->ForwardPrefilled();
	int dim_blob = output_blob->count();
	const float* blob_data = output_blob->cpu_data() + output_blob->offset(0);
	std::vector<float> result;
	for (int i = 0; i < dim_blob; ++i) {
		result.push_back(blob_data[i]);
	}
	return result;
}

inline void FeatureExtractor3::OpenCV2Blob(const std::vector<cv::Mat>& imgs) {
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

inline void FeatureExtractor3::SetMean(const std::string& mean_file) {
	caffe::BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), 3)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float *data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < 3; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(224, 224, mean.type(), channel_mean);
}

#endif //_FEATURE_EXTRACTOR3_HPP_