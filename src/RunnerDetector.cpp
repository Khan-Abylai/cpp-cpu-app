//
// Created by kartykbayev on 8/23/22.
//

#include <iostream>
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

template<typename T>
T vectorProduct(const std::vector<T> &v) {
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os,
                         const ONNXTensorElementDataType &type) {
    switch (type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}



std::vector<float> prepareImage(const cv::Mat &frame) {
    vector<float> flattenedImage;
    int IMG_WIDTH = 512, IMG_HEIGHT = 512, INPUT_SIZE = 3*IMG_WIDTH*IMG_HEIGHT;
    flattenedImage.resize(INPUT_SIZE, 0);
    int PIXEL_MAX_VALUE = 255;
    cv::Mat resizedFrame;
    resize(frame, resizedFrame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    for (int row = 0; row < resizedFrame.rows; row++) {
        for (int col = 0; col < resizedFrame.cols; col++) {
            uchar *pixels = resizedFrame.data + resizedFrame.step[0] * row + resizedFrame.step[1] * col;
            flattenedImage[row * IMG_WIDTH + col] =
                    static_cast<float>(2 * (pixels[0] / PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * (pixels[1] / PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + 2 * IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * (pixels[2] / PIXEL_MAX_VALUE - 0.5));
        }
    }
    return move(flattenedImage);
}


int main() {
    const int64_t batchSize = 1;
    cout << "Hello World" << endl;
    std::string instanceName{"lp-detection"};
    std::string modelFilepath{"../weights/detector_base.onnx"};
    std::string imageFilePath{"../data/plate_images/01AM055_08-42-59.jpeg"};
    cout << modelFilepath << endl;
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);

    sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    const char *inputName = session.GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1) {
        std::cout << "Got dynamic batch size. Setting input batch size to "
                  << batchSize << "." << std::endl;
        inputDims.at(0) = batchSize;
    }

    const char *outputName = session.GetOutputName(0, allocator);

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1) {
        std::cout << "Got dynamic batch size. Setting output batch size to "
                  << batchSize << "." << std::endl;
        outputDims.at(0) = batchSize;
    }
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    std::cout << "Input Name: " << inputName << std::endl;
    std::cout << "Input Type: " << inputType << std::endl;
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    std::cout << "Output Name: " << outputName << std::endl;
    std::cout << "Output Type: " << outputType << std::endl;
    std::cout << "Output Dimensions: " << outputDims << std::endl;


    cv::Mat image = cv::imread("../data/plate_images/image-3.jpg");
    auto flattened = prepareImage(image);
    auto stop = 1;

    return 0;
}