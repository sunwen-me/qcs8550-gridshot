#include "qnn-ball.hpp"

#include <opencv2/videoio.hpp>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>
#include <unistd.h>

using namespace std::chrono_literals;

volatile std::sig_atomic_t running = 1;
void stop(int) { running = 0; }

struct Frame {
    cv::Mat image;
    Clock::time_point captured;
};

class Camera {
public:
    Camera(const std::string& source, int backend) : camera(source, backend) {
        if (!camera.isOpened()) throw std::runtime_error("Cannot open HDMI capture source");
        if (backend == cv::CAP_V4L2) camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
        worker = std::jthread([this](std::stop_token stop_token) {
            while (!stop_token.stop_requested()) {
                cv::Mat image;
                if (!camera.read(image)) break;
                {
                    std::lock_guard guard(mutex);
                    latest = Frame{std::move(image), Clock::now()};
                }
                ready.notify_one();
            }
            {
                std::lock_guard guard(mutex);
                ended = true;
            }
            ready.notify_one();
        });
    }

    std::optional<Frame> next() {
        std::unique_lock guard(mutex);
        while (running && !ended && !latest) ready.wait_for(guard, 100ms);
        if (running && !latest && !ended) throw std::runtime_error("HDMI capture stopped");
        return std::exchange(latest, std::nullopt);
    }

private:
    cv::VideoCapture camera;
    std::optional<Frame> latest;
    std::mutex mutex;
    std::condition_variable ready;
    bool ended = false;
    std::jthread worker;
};

#ifdef GRIDSHOT_HAVE_QNN
class QnnDetector {
public:
    QnnDetector(const std::string& runtime, const std::string& model) : session(runtime) {
        session.load(model);
        const auto inputs = session.inputs(), outputs = session.outputs();
        if (inputs.size() != 1 || outputs.size() != 1) throw std::runtime_error("Expected one input and output");
        input_name = inputs.front();
        output_name = outputs.front();
        const auto& input = session.tensor(input_name);
        if (input.shape != qnn_runtime::Shape{1, 640, 640, 3} || input.type != qnn_runtime::Type::Float16 ||
            input.bytes.size() != 3 * 640 * 640 * sizeof(std::uint16_t) ||
            session.tensor(output_name).shape != qnn_runtime::Shape{1, 5, 8400})
            throw std::runtime_error("Expected our FP16 NHWC 640x640 model");
        buffer.resize(input.elements());
    }

    std::vector<Detection> operator()(const Input& input, cv::Size source) {
        write_input(session.tensor(input_name), input.data, table, buffer, true);
        session.run();
        return decode(session.read_f32(output_name), false, input, source);
    }

private:
    qnn_runtime::Session session;
    std::string input_name, output_name;
    std::vector<std::uint16_t> buffer;
    const std::array<std::uint16_t, 256> table = normalization_table();
};
#endif

class Detector {
public:
    Detector(const std::string& runtime, const std::string& model) {
        if (std::filesystem::path(model).extension() == ".onnx") {
            net = cv::dnn::readNetFromONNX(model);
            if (net.empty()) throw std::runtime_error("Could not load ONNX model");
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } else {
#ifdef GRIDSHOT_HAVE_QNN
            qnn.emplace(runtime, model);
#else
            (void)runtime;
            throw std::runtime_error("For a QNN context, build with GRIDSHOT_WITH_QNN=ON; otherwise supply an ONNX model");
#endif
        }
    }

    std::vector<Detection> operator()(const Input& input, cv::Size source) {
#ifdef GRIDSHOT_HAVE_QNN
        if (qnn) return (*qnn)(input, source);
#endif
        net.setInput(cv::dnn::blobFromImage(input.data, 1. / 255., {640, 640}, {}, true));
        const auto out = net.forward();
        if (out.type() != CV_32F || out.dims != 3 || out.size[0] != 1 ||
            out.size[1] != 5 || out.size[2] != 8400 || !out.isContinuous())
            throw std::runtime_error("Expected a one-class YOLO output [1,5,8400]");
        const auto* values = out.ptr<float>();
        return decode(std::vector<float>(values, values + out.total()), false, input, source);
    }

private:
    cv::dnn::Net net;
#ifdef GRIDSHOT_HAVE_QNN
    std::optional<QnnDetector> qnn;
#endif
};

struct HidBurst {
    cv::Point moved{};
    int reports = 0;
};

class HidMouse {
public:
    explicit HidMouse(const std::string& path) {
        fd = open(path.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error(path + ": " + std::strerror(errno));
    }
    HidMouse(const HidMouse&) = delete;
    HidMouse& operator=(const HidMouse&) = delete;
    ~HidMouse() { close(fd); }

    HidBurst move(cv::Point delta, Clock::time_point until) {
        const int extent = std::max(std::abs(delta.x), std::abs(delta.y));
        const int packets = (extent + 126) / 127;
        const auto deadline = std::min(until, Clock::now() + 20ms);
        HidBurst burst;
        for (int i = 1; i <= packets; ++i) {
            const cv::Point prefix(std::lround(double(delta.x) * i / packets),
                                   std::lround(double(delta.y) * i / packets));
            while (running && Clock::now() < deadline) {
                if (write_report(0, prefix - burst.moved)) {
                    burst.moved = prefix;
                    ++burst.reports;
                    break;
                }
                std::this_thread::sleep_for(1ms);
            }
            if (burst.reports != i) break;
        }
        return burst;
    }

    bool click() {
        if (!write_report(1, {})) return false;
        std::this_thread::sleep_for(8ms);
        const auto deadline = Clock::now() + 40ms;
        while (Clock::now() < deadline) {
            if (write_report(0, {})) return true;
            std::this_thread::sleep_for(1ms);
        }
        throw std::runtime_error("Could not release HID left button");
    }

private:
    bool write_report(std::uint8_t buttons, cv::Point delta) {
        const std::array<std::int8_t, 3> report{
            static_cast<std::int8_t>(buttons), static_cast<std::int8_t>(delta.x), static_cast<std::int8_t>(delta.y)
        };
        const auto sent = write(fd, report.data(), report.size());
        const int error = errno;
        if (sent == static_cast<ssize_t>(report.size())) return true;
        if (sent < 0 && (error == EAGAIN || error == EINTR)) return false;
        throw std::runtime_error(sent < 0 ? std::strerror(error) : "Incomplete mouse HID report");
    }

    int fd;
};

struct AimSettings {
    double counts_per_pixel_x = 0.3;
    double counts_per_pixel_y = 0.3;
    int feedback_ms = 120;
};

class Controller {
public:
    explicit Controller(AimSettings settings) : settings(settings) {}

    std::optional<cv::Point> update(const std::vector<Detection>& detections, cv::Size size,
                                    Clock::time_point captured, Clock::time_point now) {
        feedback_status = "idle";
        confirmed_ms.reset();
        if (now - captured > 100ms) return std::nullopt;
        centers.clear();
        for (const auto& box : detections) {
            if (box.confidence >= 0.55f)
                centers.emplace_back((box.x1 + box.x2) * 0.5, (box.y1 + box.y2) * 0.5);
        }
        std::optional<cv::Point2d> predicted;
        if (pending) {
            const auto age = captured - pending->sent_at;
            const auto fallback = std::max(65ms, std::chrono::milliseconds(settings.feedback_ms));
            feedback_status = "waiting";
            if (age < 65ms) return std::nullopt;
            auto observed = shared_motion(*pending);
            if (observed) {
                feedback_status = "shared_motion";
            } else if (age >= fallback) {
                observed = locked_motion(*pending);
                if (observed) feedback_status = "locked_motion";
            }
            if (observed) {
                predicted = pending->anchor + *observed;
                confirmed_ms = elapsed_ms(pending->sent_at, captured);
                const double match_radius = std::clamp(0.07 * cv::norm(*observed), 13., 38.);
                if (feedback_status == "shared_motion" && missing(*predicted, match_radius) &&
                    surviving_neighbors(pending->before, pending->anchor, *observed, match_radius) >= 2) {
                    locked.reset();
                    predicted.reset();
                    feedback_status = "target_replaced";
                }
                pending.reset();
            } else if (age >= fallback) {
                const bool scene_changed = !centers.empty() && missing(pending->anchor, 18.) &&
                    std::any_of(centers.begin(), centers.end(), [&](const auto& point) {
                        return std::none_of(pending->before.begin(), pending->before.end(),
                            [&](const auto& old) { return cv::norm(point - old) <= 18.; });
                    });
                // Expire the unconfirmed move, then compute a normal correction from this fresh frame.
                feedback_status = scene_changed ? "scene_changed" : "feedback_timeout";
                locked.reset();
                centered_anchor.reset();
                pending.reset();
            } else {
                return std::nullopt;
            }
        }

        const cv::Point2d crosshair(size.width * 0.5, size.height * 0.5);
        if (centered_anchor) {
            // With the view settled, two stationary neighbors distinguish a hit from camera motion.
            if (locked && missing(*centered_anchor, 12.) &&
                surviving_neighbors(centered_scene, *centered_anchor, {}, 12.) >= 2) {
                locked.reset();
                feedback_status = "target_replaced";
            }
            centered_anchor.reset();
        }
        const auto nearest = [&](cv::Point2d origin, double radius) {
            std::optional<cv::Point2d> best;
            for (const auto& center : centers) {
                const double distance = cv::norm(center - origin);
                if (distance < radius) {
                    best = center;
                    radius = distance;
                }
            }
            return best;
        };
        auto selected = locked ? nearest(predicted.value_or(*locked), 80.) : std::nullopt;
        if (!selected) {
            locked.reset();
            selected = nearest(crosshair, std::numeric_limits<double>::infinity());
        }
        if (!selected) return std::nullopt;
        if (!locked) {
            ++generation;
        }
        locked = selected;
        const auto error = *locked - crosshair;
        const double error_distance = cv::norm(error);
        if (error_distance <= 3.0) {
            centered_anchor = locked;
            centered_scene = centers;
            return std::nullopt;
        }
        centered_anchor.reset();
        const auto counts = [](double error_px, double scale) {
            return std::lround(std::clamp(error_px * scale, -512., 512.));
        };
        const cv::Point rounded(counts(error.x, settings.counts_per_pixel_x),
                                counts(error.y, settings.counts_per_pixel_y));
        if (rounded == cv::Point()) return std::nullopt;
        return rounded;
    }

    void sent(Clock::time_point time, cv::Point moved) {
        if (locked) {
            pending = Pending{time, *locked, {-moved.x / settings.counts_per_pixel_x,
                                              -moved.y / settings.counts_per_pixel_y}, centers};
        }
    }
    std::optional<cv::Point2d> locked_center() const { return locked; }
    std::size_t lock_generation() const { return generation; }
    std::string_view feedback() const { return feedback_status; }
    std::optional<double> feedback_latency() const { return confirmed_ms; }

private:
    struct Pending {
        Clock::time_point sent_at;
        cv::Point2d anchor;
        cv::Point2d expected;
        std::vector<cv::Point2d> before;
    };

    bool missing(const cv::Point2d& old, double radius) const {
        return std::none_of(centers.begin(), centers.end(),
            [&](const auto& point) { return cv::norm(point - old) <= radius; });
    }

    int surviving_neighbors(const std::vector<cv::Point2d>& before, cv::Point2d anchor,
                            cv::Point2d motion, double radius) const {
        // Count distinct old and new detections; ignore the anchor and its immediate vicinity.
        if (before.size() > 6 || centers.size() > 6) return 0;
        std::array<bool, 6> used{};
        int matched = 0;
        for (const auto& old : before) {
            if (cv::norm(old - anchor) <= radius) continue;
            std::size_t index = centers.size();
            double best = radius;
            for (std::size_t i = 0; i < centers.size(); ++i) {
                if (used[i]) continue;
                const double error = cv::norm(centers[i] - (old + motion));
                if (error < best) { index = i; best = error; }
            }
            if (index < centers.size()) { used[index] = true; ++matched; }
        }
        return matched;
    }

    static bool plausible(const cv::Point2d& actual, const cv::Point2d& expected, double length) {
        if (!std::isfinite(length) || length < 0.01) return false;
        const auto direction = expected * (1. / length);
        const double along = actual.dot(direction);
        return along >= std::max(0.65, 0.68 * length) &&
               along <= std::max(length + 1.5, 1.60 * length) &&
               std::abs(actual.cross(direction)) <= std::max(1.5, 0.22 * length);
    }

    std::optional<cv::Point2d> shared_motion(const Pending& move) const {
        const double length = cv::norm(move.expected);
        if (length < 6. || centers.size() < 2 || move.before.size() < 2 ||
            centers.size() > 6 || move.before.size() > 6) return std::nullopt;
        const double tolerance = std::clamp(0.12 * length, 2., 55.);
        std::optional<cv::Point2d> best;
        double error = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < move.before.size(); ++i) {
            if (!missing(move.before[i], 1.)) continue;
            for (std::size_t j = 0; j < centers.size(); ++j) {
                const auto shift = centers[j] - move.before[i];
                if (!plausible(shift, move.expected, length)) continue;
                const double cost = cv::norm(shift - move.expected);
                if (cost >= error) continue;
                bool supported = false;
                for (std::size_t k = 0; k < move.before.size() && !supported; ++k) {
                    if (k == i || !missing(move.before[k], 1.)) continue;
                    for (std::size_t l = 0; l < centers.size(); ++l) {
                        if (l == j) continue;
                        const auto other = centers[l] - move.before[k];
                        if (cv::norm(other - shift) <= tolerance && plausible(other, move.expected, length)) {
                            supported = true;
                            break;
                        }
                    }
                }
                if (supported) { best = shift; error = cost; }
            }
        }
        return best;
    }

    std::optional<cv::Point2d> locked_motion(const Pending& move) const {
        const double length = cv::norm(move.expected);
        if (!missing(move.anchor, std::clamp(0.24 * length, 0.5, 5.))) return std::nullopt;
        std::optional<cv::Point2d> best;
        double error = std::max(1.5, 0.38 * length);
        for (const auto& point : centers) {
            const auto shift = point - move.anchor;
            const double cost = cv::norm(shift - move.expected);
            if (cost < error && plausible(shift, move.expected, length)) {
                best = shift;
                error = cost;
            }
        }
        return best;
    }

    AimSettings settings;
    std::optional<cv::Point2d> locked;
    std::optional<cv::Point2d> centered_anchor;
    std::vector<cv::Point2d> centered_scene;
    std::vector<cv::Point2d> centers;
    std::optional<Pending> pending;
    std::string_view feedback_status = "idle";
    std::optional<double> confirmed_ms;
    std::size_t generation = 0;
};

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        cv::setNumThreads(4);
        if (argc != 5 && argc != 8 && argc != 9 && argc != 10)
            throw std::runtime_error(
                "Usage: aim-live RUNTIME MODEL --replay DIRECTORY\n"
                "       aim-live RUNTIME MODEL --camera /dev/videoN [HID_PATH COUNTS_X COUNTS_Y [FEEDBACK_MS [TEST_MS]]]\n"
                "       aim-live RUNTIME MODEL --gst PIPELINE [HID_PATH COUNTS_X COUNTS_Y [FEEDBACK_MS [TEST_MS]]]\n"
                "       aim-live RUNTIME MODEL --gst-hdmi1080-nv12 PIPELINE [HID_PATH COUNTS_X COUNTS_Y [FEEDBACK_MS [TEST_MS]]]");
        const std::string_view mode = argv[3];
        const bool nv12 = mode == "--gst-hdmi1080-nv12";
        const bool live = mode == "--camera" || mode == "--gst" || nv12;
        const bool active = argc >= 8;
        if (!live && (mode != "--replay" || active))
            throw std::runtime_error("HID is only available with an explicit live camera");
        AimSettings settings;
        int test_ms = 0;
        if (active) {
            settings.counts_per_pixel_x = std::stod(argv[6]);
            settings.counts_per_pixel_y = std::stod(argv[7]);
            const auto parse_ms = [](const char* text) {
                std::size_t consumed = 0;
                const int value = std::stoi(text, &consumed);
                if (text[consumed] != '\0') throw std::runtime_error("Expected an integer duration in ms");
                return value;
            };
            if (argc >= 9) settings.feedback_ms = parse_ms(argv[8]);
            if (argc == 10) {
                test_ms = parse_ms(argv[9]);
                if (test_ms < 1 || test_ms > 3000)
                    throw std::runtime_error("Specify test duration in the 1–3000 ms range");
            }
            const auto calibrated = [](double value) { return std::isfinite(value) && value != 0; };
            if (!calibrated(settings.counts_per_pixel_x) || !calibrated(settings.counts_per_pixel_y)
                || settings.feedback_ms < 0 || settings.feedback_ms > 250)
                throw std::runtime_error("Specify nonzero finite calibration and feedback in the 0–250 ms range");
        }
        Detector detector(argv[1], argv[2]);
        Controller controller(settings);
        std::optional<HidMouse> mouse;
        if (active) mouse.emplace(argv[5]);
        std::optional<Clock::time_point> test_start;
        bool gridshot_seen = false;
        const auto test_window = std::chrono::milliseconds(test_ms);
        constexpr auto settle_window = 300ms;

        const auto process = [&](const Frame& frame) {
            if (frame.image.empty()) throw std::runtime_error("Cannot decode a frame");
            if (nv12 && nv12_frame_size(frame.image) != cv::Size(640, 360))
                throw std::runtime_error("Expected 640x360 NV12 from the HDMI scaler");
            const auto source = nv12 ? cv::Size(1920, 1080) : frame.image.size();
            const auto input = nv12 ? letterbox_nv12(frame.image, source) : letterbox(frame.image);
            const auto detections = detector(input, source);
            const auto now = Clock::now();
            const auto visible_targets = std::count_if(detections.begin(), detections.end(),
                [](const Detection& box) { return box.confidence >= 0.55f; });
            const bool has_target = visible_targets > 0;
            if (has_target && now - frame.captured <= 100ms) {
                gridshot_seen = true;
                if (test_ms && !test_start) test_start = now;
            }
            const bool test_can_move = !test_ms || (test_start && has_target && now - *test_start < test_window);
            const cv::Point2d crosshair(source.width * 0.5, source.height * 0.5);
            const bool frame_fresh = now - frame.captured <= 100ms;
            const bool fire_aligned = frame_fresh && std::any_of(detections.begin(), detections.end(),
                [&](const Detection& box) {
                    if (box.confidence < 0.55f) return false;
                    const cv::Point2d center((box.x1 + box.x2) * 0.5, (box.y1 + box.y2) * 0.5);
                    return cv::norm(center - crosshair) < 30.0;
                });
            const auto delta = test_can_move
                ? controller.update(detections, source, frame.captured, now)
                : std::nullopt;
            const auto deadline = test_start ? *test_start + test_window : Clock::time_point::max();
            const auto burst = mouse && gridshot_seen && test_can_move && running && !fire_aligned && delta
                ? mouse->move(*delta, deadline) : HidBurst{};
            const bool sent = burst.reports != 0;
            if (sent) controller.sent(Clock::now(), burst.moved);
            const bool fired = mouse && gridshot_seen && has_target && test_can_move && running &&
                               fire_aligned && mouse->click();
            const auto locked = controller.locked_center();
            std::cout << "{\"detections\":" << detections.size()
                      << ",\"frame_ms\":" << elapsed_ms(frame.captured, now)
                      << ",\"width\":" << source.width << ",\"height\":" << source.height
                      << ",\"sent\":" << (sent ? "true" : "false")
                      << ",\"dx\":" << (delta ? delta->x : 0)
                      << ",\"dy\":" << (delta ? delta->y : 0)
                      << ",\"hid_reports\":" << burst.reports
                      << ",\"moved_x\":" << burst.moved.x
                      << ",\"moved_y\":" << burst.moved.y
                      << ",\"fired\":" << (fired ? "true" : "false")
                      << ",\"feedback\":\"" << (test_can_move ? controller.feedback() : "idle") << '\"'
                      << ",\"feedback_ms\":" << (test_can_move ? controller.feedback_latency().value_or(-1.) : -1.)
                      << ",\"test_ms\":" << (test_start ? elapsed_ms(*test_start, now) : -1.0)
                      << ",\"lock_generation\":" << controller.lock_generation()
                      << ",\"locked_center\":";
            if (locked) std::cout << '[' << locked->x << ',' << locked->y << ']';
            else std::cout << "null";
            std::cout << ",\"centers\":[";
            bool first = true;
            for (const auto& box : detections) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << '[' << (box.x1 + box.x2) * .5 << ','
                          << (box.y1 + box.y2) * .5 << ',' << box.confidence << ']';
            }
            std::cout << "]}\n" << std::flush;
        };
        if (live) {
            Camera camera(argv[4], mode == "--camera" ? cv::CAP_V4L2 : cv::CAP_GSTREAMER);
            while (running) {
                const auto frame = camera.next();
                if (!frame) break;
                process(*frame);
                if (test_start && Clock::now() - *test_start >= test_window + settle_window) break;
            }
        } else {
            std::vector<std::filesystem::path> paths;
            for (const auto& entry : std::filesystem::directory_iterator(argv[4])) {
                const auto ext = entry.path().extension();
                if (entry.is_regular_file() && (ext == ".jpg" || ext == ".png")) paths.push_back(entry.path());
            }
            std::sort(paths.begin(), paths.end());
            if (paths.empty()) throw std::runtime_error("Replay directory has no images");
            for (const auto& path : paths) {
                if (!running) break;
                process({cv::imread(path.string()), Clock::now()});
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
