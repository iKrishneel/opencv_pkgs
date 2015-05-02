
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <include/svm.h>
#include <iostream>
#include <string>

struct svm_problem libSVMWrapper(
    const cv::Mat &feature_mat, const cv::Mat &label_mat,
    const svm_parameter &param) {
    if (feature_mat.rows != label_mat.rows) {
       std::cout << "--TRAINING SET IS NOT EQUIVALENT.." << std::endl;
       std::_Exit(EXIT_FAILURE);
    }
    svm_problem svm_prob_vector;
    const int feature_lenght = static_cast<int>(feature_mat.rows);
    const int feature_dimensions = static_cast<int>(feature_mat.cols);
    svm_prob_vector.l = feature_lenght;
    svm_prob_vector.y = new double[feature_lenght];
    svm_prob_vector.x = new svm_node*[feature_lenght];
    for (int i = 0; i < feature_lenght; i++) {
       svm_prob_vector.x[i] = new svm_node[feature_dimensions];
    }
    for (int j = 0; j < feature_lenght; j++) {
       svm_prob_vector.y[j] = static_cast<double>(label_mat.at<float>(j, 0));
       for (int i = 0; i < feature_dimensions; i++) {
          svm_prob_vector.x[j][i].index = i + 1;
          svm_prob_vector.x[j][i].value = static_cast<double>(
             feature_mat.at<float>(j, i));
       }
       svm_prob_vector.x[j][feature_dimensions].index = -1;
    }
    return svm_prob_vector;
}

int main(int argc, char *argv[]) {

    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    cv::Mat labelsMat(4, 1, CV_32FC1, labels);
    float trainingData[4][2] = {{501, 10}, {255, 10},
                                {501, 255}, {10, 501}};
    cv::Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-6;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    svm_problem svm_prob_vector = libSVMWrapper(
       trainingDataMat, labelsMat, param);
    struct svm_model *model = new svm_model;
    if (svm_check_parameter(&svm_prob_vector, &param)) {
       std::cout << "ERROR" << std::endl;
    } else {
       model = svm_train(&svm_prob_vector, &param);
    }

    std::string model_file_name = "svm";
    bool save_model = true;
    if (save_model) {
       try {
          svm_save_model(model_file_name.c_str(), model);
          std::cout << "Model file Saved Successfully..." << std::endl;
       } catch(std::exception& e) {
          std::cout << e.what() << std::endl;
       }
    }
    cv::Vec3b green(0, 255, 0);
    cv::Vec3b blue(255, 0, 0);
    int width = 512, height = 512;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i < image.rows; ++i) {
       for (int j = 0; j < image.cols; ++j) {
          cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << j, i);

          int dims = sampleMat.cols;
          svm_node* test_pt = new svm_node[dims];
          for (int k = 0; k < dims; k++) {
             test_pt[k].index = k + 1;
             test_pt[k].value = static_cast<double>(sampleMat.at<float>(0, k)); 
          }
          test_pt[dims].index = -1;
          float response = svm_predict(model, test_pt);
          
          if (response == 1)
             image.at<cv::Vec3b>(i, j)  = green;
          else if (response == -1)
             image.at<cv::Vec3b>(i, j)  = blue;
       }
    }

    cv::imshow("image", image);
    cv::waitKey(0);
    
    return 0;
}
