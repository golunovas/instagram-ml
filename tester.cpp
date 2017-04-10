#include <map>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <thread>

#include <jsonity.hpp>
#include <restless.hpp>
#include <opencv2/opencv.hpp>
#include <FeatureExtractor.hpp>
#include <FeatureExtractor2.hpp>
#include <FeatureExtractor3.hpp>
	
using Http = asoni::Handle;
using namespace jsonity;

const std::string PASSWD = "alex-golunov@***";
const std::string LIVE_URL = "http://challenges.instagram.unpossib.ly/api/live";
const std::string SUBMISSIONS_URL = "http://challenges.instagram.unpossib.ly/api/submissions";


cv::PCA loadPCA(const std::string &file_name) {
	cv::PCA pca_;
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    fs["mean"] >> pca_.mean ;
    fs["e_vectors"] >> pca_.eigenvectors ;
    fs["e_values"] >> pca_.eigenvalues ;
    fs.release();
    return pca_;
}

bool sendSubmission(const std::string& postID, float prediction) {
	Json::Object json;
	json["post"] = postID;
	json["likes"] = static_cast<int>(prediction);
	std::string jsonString;
	Json::encode(json, jsonString);
	std::string responseJsonString = Http().post(SUBMISSIONS_URL, PASSWD)
                    .content("text/plain", jsonString)
                    .exec().body;
    std::cout << responseJsonString << std::endl;
    Json::Value responseJson;
    Json::decode(responseJsonString, responseJson);
    return responseJson.getObject()["success"].getBoolean();

}

int main(int argc, char** argv) {
	std::set<std::string> submittedPosts;
	// aesthetics analysis
	FeatureExtractor featureExtractor("./model/initModel.prototxt", 
				  	 "./model/initModel.caffemodel", 
					 "./model/mean.binaryproto", 
					 false, 
					 0);
	// VGG
	FeatureExtractor2 featureExtractor2("./model3/deploy.prototxt", 
					  "./model3/model.caffemodel",							  
					  "./model3/mean.binaryproto", 
					  false, 
					  0);
	// GoogleNet
	FeatureExtractor3 featureExtractor3("./model2/deploy.prototxt", 
					  "./model2/model.caffemodel", 
					  "./model2/mean.binaryproto", 
					  false, 0);


	while (true) {
		try {
			std::string inJsonString = Http().get(LIVE_URL, PASSWD).exec().body;
			std::cout << "Received: " << inJsonString << std::endl;
			Json::Value inJsonValue;
			Json::decode(inJsonString, inJsonValue);
			auto& jsonAccounts = inJsonValue.getObject()["accounts"].getArray();
			for (size_t i = 0; i < jsonAccounts.size(); ++i) {
				std::string username = jsonAccounts[i]["username"].getString();
				std::cout << "Username: " << username << std::endl;
				auto& postsJson = jsonAccounts[i]["posts"].getArray();
				if (postsJson.empty()) {
					continue;
				}
				cv::PCA PCA = loadPCA(username + std::string(".pca"));
				CvSVM SVM;
				SVM.load((username + std::string(".svm")).c_str());
				for (size_t j = 0; j < postsJson.size(); ++j) {
					auto& postJson = postsJson[j].getObject();
					std::string postID = postsJson[j]["instagram"].getObject()["id"].getString();
					std::cout << "PostID: " << postID << std::endl;
					if (submittedPosts.find(postID) != submittedPosts.end()) {
						std::cout << "Skip postID: " << postID << std::endl;
						continue;
					}
					std::string imageURL = postsJson[j]["instagram"].getObject()["display_src"].getString();
					std::cout << "ImageURL: " << imageURL << std::endl;
					std::string dataStr = Http().get(imageURL).exec().body;
					if (dataStr.length() == 0) {
						std::cout << "Couldn't download the image" << std::endl;
						continue;
					}
					std::vector<char> dataVec;
					std::copy( dataStr.begin(), dataStr.end(), std::back_inserter(dataVec));
					cv::Mat img = cv::imdecode(dataVec, CV_LOAD_IMAGE_COLOR);
					if (img.empty()) {
						std::cout << "Invalid image" << std::endl;
						continue;
					}
					std::cout << "Image width: " << img.cols << " height: " << img.rows << std::endl;
					std::vector<float> f = featureExtractor.extractFeatures(img);
					std::vector<float> f2 = featureExtractor2.extractFeatures(img);
					std::vector<float> f3 = featureExtractor3.extractFeatures(img);
					cv::Mat fm = cv::Mat(f).t();
					cv::hconcat(fm, cv::Mat(f2).t(), fm);
					cv::hconcat(fm, cv::Mat(f3).t(), fm);
					cv::Mat features = PCA.project(fm);
					float prediction = SVM.predict(features);
					bool isSuccess = sendSubmission(postID, prediction);
					if (isSuccess) {
						submittedPosts.insert(postID);
					}
					std::cout << "Prediction: " << prediction << std::endl;
					std::cout << "isSuccess: " << isSuccess << std::endl;
				}

			}
		} catch (...) { 
			// we don't want to die in the middle of the competition
		}	
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 * 60));
	}
	return 0;
}