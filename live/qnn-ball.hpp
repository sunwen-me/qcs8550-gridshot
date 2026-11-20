#pragma once
#ifdef GRIDSHOT_HAVE_QNN
#include <qnn_runtime/session.hpp>
#endif
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double percentile(std::vector<double> timings, double fraction) {
    std::sort(timings.begin(), timings.end());
    return timings[static_cast<size_t>(std::ceil(timings.size() * fraction)) - 1];
}

struct Input {
    cv::Mat data;
    double scale;
    int left;
    int top;
};

Input letterbox(const cv::Mat& image) {
    constexpr int size = 640;
    const double scale = std::min(double(size) / image.cols, double(size) / image.rows);
    const int width = std::lround(image.cols * scale), height = std::lround(image.rows * scale);
    const int left = (size - width) / 2, top = (size - height) / 2;
    cv::Mat padded(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat region = padded(cv::Rect(left, top, width, height));
    cv::resize(image, region, region.size(), 0, 0, cv::INTER_LINEAR);
    return {padded, scale, left, top};
}

cv::Size nv12_frame_size(const cv::Mat& packed) {
    if (packed.type() != CV_8UC1 || packed.cols <= 0 || packed.cols % 2 ||
        packed.rows <= 0 || packed.rows % 3)
        throw std::runtime_error("Expected packed NV12 with even dimensions");
    return {packed.cols, (packed.rows / 3) * 2};
}

Input letterbox_nv12(const cv::Mat& packed, cv::Size source) {
    constexpr int size = 640;
    const double scale = std::min(double(size) / source.width, double(size) / source.height);
    const int width = std::max(1, int(std::lround(source.width * scale)));
    const int height = std::max(1, int(std::lround(source.height * scale)));
    const int left = (size - width) / 2, top = (size - height) / 2;
    const cv::Size even((width + 1) & ~1, (height + 1) & ~1);
    const auto sample = nv12_frame_size(packed);
    const cv::Mat y = packed.rowRange(0, sample.height);
    // OpenCV's GStreamer backend supplies packed Y rows followed by interleaved UV rows.
    const cv::Mat uv = packed.rowRange(sample.height, packed.rows).reshape(2);
    cv::Mat small_y = y, small_uv = uv;
    if (sample != even) {
        cv::resize(y, small_y, even, 0, 0, cv::INTER_LINEAR);
        cv::resize(uv, small_uv, {even.width / 2, even.height / 2}, 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat padded(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat region = padded(cv::Rect(left, top, width, height));
    if (even == region.size()) {
        cv::cvtColorTwoPlane(small_y, small_uv, region, cv::COLOR_YUV2BGR_NV12);
    } else {
        cv::Mat even_bgr;
        cv::cvtColorTwoPlane(small_y, small_uv, even_bgr, cv::COLOR_YUV2BGR_NV12);
        cv::resize(even_bgr, region, region.size(), 0, 0, cv::INTER_LINEAR);
    }
    return {padded, scale, left, top};
}

#ifdef GRIDSHOT_HAVE_QNN
std::array<std::uint16_t, 256> normalization_table() {
    std::array<std::uint16_t, 256> table;
    for (size_t value = 0; value < table.size(); ++value)
        table[value] = cv::float16_t(static_cast<float>(value) * (1.f / 255)).bits();
    return table;
}

void write_input(qnn_runtime::Tensor& tensor, const cv::Mat& image,
                 const std::array<std::uint16_t, 256>& table, std::vector<std::uint16_t>& buffer,
                 bool nhwc) {
    constexpr size_t pixels = 640 * 640;
    const auto* bgr = image.ptr<cv::Vec3b>();
    if (nhwc) {
        for (size_t index = 0; index < pixels; ++index) {
            buffer[3 * index] = table[bgr[index][2]];
            buffer[3 * index + 1] = table[bgr[index][1]];
            buffer[3 * index + 2] = table[bgr[index][0]];
        }
    } else {
        for (size_t index = 0; index < pixels; ++index) {
            buffer[index] = table[bgr[index][2]];
            buffer[pixels + index] = table[bgr[index][1]];
            buffer[2 * pixels + index] = table[bgr[index][0]];
        }
    }
    std::memcpy(tensor.bytes.data(), buffer.data(), tensor.bytes.size());
}

#endif

struct Detection {
    double x1, y1, x2, y2;
    float confidence;
};

std::vector<Detection> decode(const std::vector<float>& output, bool channel_last,
                              const Input& input, cv::Size source) {
    constexpr int anchors = 8400;
    constexpr float confidence_threshold = 0.25f;
    const auto feature = [&](int feature_index, int anchor) {
        return output[channel_last ? anchor * 5 + feature_index : feature_index * anchors + anchor];
    };
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    for (int anchor = 0; anchor < anchors; ++anchor) {
        const float score = feature(4, anchor);
        if (score <= confidence_threshold) continue;
        const float width = feature(2, anchor), height = feature(3, anchor);
        boxes.emplace_back(feature(0, anchor) - width / 2, feature(1, anchor) - height / 2, width, height);
        scores.push_back(score);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, 0.45f, indices);
    std::vector<Detection> result;
    for (const int index : indices) {
        const auto& box = boxes[index];
        const auto x = [&](double coordinate) { return std::clamp((coordinate - input.left) / input.scale, 0., double(source.width)); };
        const auto y = [&](double coordinate) { return std::clamp((coordinate - input.top) / input.scale, 0., double(source.height)); };
        result.push_back({x(box.x), y(box.y), x(box.x + box.width), y(box.y + box.height), scores[index]});
    }
    return result;
}


inline std::vector<Detection> decode(const std::vector<float>& output, bool channel_last,
                                     const Input& input, const cv::Mat& image) {
    return decode(output, channel_last, input, image.size());
}
