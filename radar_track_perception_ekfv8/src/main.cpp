// main.cpp (v3 modified: Fail-Safe Soft Ego)
// 逻辑更新：
// 软车速模块现在会返回一个 bool 状态。
// 如果 候选点不足(启动失败) -> 跳过背景过滤 -> 所有点直接给 XGBoost。
// 如果 启动成功 -> 算出车速 -> 过滤掉静止背景 -> 剩余点给 XGBoost。

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/make_shared.h>
#include <xgboost/c_api.h>
#include <nlohmann/json.hpp>

#include <unordered_map>
#include <map>
#include <set>
#include <deque>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <memory>
#include <filesystem>
#include <chrono>

#ifdef __linux__
#include <X11/Xlib.h>
#undef Success
#undef Status
extern "C" __attribute__((constructor)) void x11_thread_init() {
    setenv("QT_X11_NO_MITSHM", "1", 1);
    int ok = XInitThreads();
    if (!ok) fprintf(stderr, "[X11] XInitThreads() failed\n");
}
#endif

using json = nlohmann::json;
using namespace std;
using namespace Eigen;
namespace fs = std::filesystem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
constexpr double DEFAULT_PROCESS_NOISE = 0.1;
constexpr double DEFAULT_MEAS_NOISE = 0.5;
constexpr int HISTORY_SIZE = 10;
} // namespace

// ===================== Data Types =====================
struct RadarObject {
    int radar_frame = 0;
    int closest_lidar_frame = 0;
    int closest_cam_frame = 0;
    int id = -1;
    float x = 0.f, y = 0.f;
    float vx = 0.f, vy = 0.f;
    float ax = 0.f, ay = 0.f;
    float rcs = 0.f;
    float length = 0.f, width = 0.f;
    float orientation = 0.f; // rad
    int trackType = 0;
    float existProbability = 0.f;
};

struct ExportRow {
    int frame;
    int internal_id;
    float x, y, vx, vy;
    float prob;
    std::string status;
    std::string color;
};

// ===================== Config Struct =====================
enum class DynPropSource { Zero, TrackType, ExistProb };

struct AppConfig {
    // Paths
    std::string radar_csv;
    std::string lidar_dir;
    std::string cam_dir;
    std::string models_dir;
    std::string radar_extrinsic;
    std::string lidar_extrinsic;
    std::string export_csv;

    // ROI
    float det_x_max = 100.0f;
    float det_y_max = 5.0f;

    // Thresholds
    double prob_thres = 0.6;
    double thr_high = 0.8;
    double thr_low = 0.6;
    int max_misses = 6;
    bool orientation_in_deg = true;
    DynPropSource dyn_prop_src = DynPropSource::TrackType;
    bool export_enabled = false;

    // --- Soft Ego Velocity Params ---
    bool enable_soft_ego_est = false; // 开关
    double ego_rcs_min = -5.0;
    double ego_length_min = 0.0;
    double ego_speed_abs_max = 10.0; 
    int ego_min_candidates = 2;
    
    double ego_bin_width = 0.5;   // 聚类窗口宽度
    double ego_weight_rcs = 1.0;  // RCS权重
    double ego_weight_len = 2.0;  // 长度权重

    double ego_alpha_mean = 0.8;
    double ego_decay = 0.99;

    // --- Static Filter Params ---
    double static_speed_thres = 0.8;
    double static_rcs_thres = 0.0;
};

// ===================== EKF =====================
class EKF {
public:
    EKF(const MatrixXd& A, const MatrixXd& H,
        const MatrixXd& Q, const MatrixXd& R,
        const VectorXd& x0, const MatrixXd& P0)
        : A_(A), H_(H), Q_(Q), R_(R), x_(x0), P_(P0) {}
    void predict() { x_ = A_ * x_; P_ = A_ * P_ * A_.transpose() + Q_; }
    void update(const VectorXd& z) {
        predict();
        MatrixXd S = H_ * P_ * H_.transpose() + R_;
        MatrixXd K = P_ * H_.transpose() * S.inverse();
        x_ += K * (z - H_ * x_);
        P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_) * P_;
    }
    VectorXd getState() const { return x_; }
private:
    MatrixXd A_, H_, Q_, R_;
    VectorXd x_;
    MatrixXd P_;
};

// ===================== Helper: SE3 =====================
struct SE3f { Matrix4f T = Matrix4f::Identity(); bool valid = false; };
static float yaw_from_R(const Matrix3f& R) { return std::atan2(R(1,0), R(0,0)); }
static SE3f load_extrinsic_json(const std::string& path) {
    SE3f se3; if (path.empty()) return se3;
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr<<"Cannot open extrinsic: "<<path<<"\n"; return se3; }
    json j; f>>j;
    if (j.contains("T")) {
        auto vec = j["T"].get<std::vector<float>>();
        if (vec.size()==16) { for (int r=0;r<4;++r) for (int c=0;c<4;++c) se3.T(r,c)=vec[r*4+c]; se3.valid=true; return se3; }
    }
    if (j.contains("xyz") && j.contains("rpy_deg")) {
        auto xyz=j["xyz"].get<std::vector<float>>();
        auto rpy=j["rpy_deg"].get<std::vector<float>>();
        if (xyz.size()==3 && rpy.size()==3) {
            float rx=rpy[0]*M_PI/180.f, ry=rpy[1]*M_PI/180.f, rz=rpy[2]*M_PI/180.f;
            AngleAxisf Rx(rx, Vector3f::UnitX()), Ry(ry, Vector3f::UnitY()), Rz(rz, Vector3f::UnitZ());
            Matrix3f R=(Rz*Ry*Rx).toRotationMatrix();
            se3.T.setIdentity();
            se3.T.block<3,3>(0,0)=R;
            se3.T.block<3,1>(0,3)=Vector3f(xyz[0],xyz[1],xyz[2]);
            se3.valid=true; return se3;
        }
    }
    return se3;
}
static Vector3f apply_T(const SE3f& se6, const Vector3f& p) { if (!se6.valid) return p; Vector4f ph; ph<<p,1.f; return (se6.T*ph).head<3>(); }
static Vector3f rotate_T(const SE3f& se6, const Vector3f& v) { if (!se6.valid) return v; return se6.T.block<3,3>(0,0)*v; }

// ===================== Main Viewer =====================
class RadarLidarViewer {
public:
    struct TrackingState { int non_detection_count=0; std::unique_ptr<EKF> tracker; };

    RadarLidarViewer(const AppConfig& config,
                     const SE3f& T_radar_to_world,
                     const SE3f& T_lidar_to_world)
        : cfg_(config),
          T_radar_(T_radar_to_world),
          T_lidar_(T_lidar_to_world) {

        initFeatureMapping();
        initEKFMats();
        loadXGBoost();
        loadRadarCSV();

        prob_thres_ = cfg_.prob_thres;

        viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("Radar Viewer");
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->addCoordinateSystem(3.0);
        viewer_->initCameraParameters();
        viewer_->registerKeyboardCallback([this](const pcl::visualization::KeyboardEvent& e){ this->KeyboardEvent(e); });

        if (!cfg_.export_csv.empty()) {
            export_stream_.open(cfg_.export_csv);
            if (export_stream_.is_open()) export_stream_<<"frame,internal_id,x,y,vx,vy,prob,status,color\n";
        }
    }

    ~RadarLidarViewer(){ if(booster_) XGBoosterFree(booster_); if(export_stream_.is_open()) export_stream_.close(); }

    void Run() {
        if (frame_data_.empty()) { std::cerr << "No radar frames loaded. Exiting.\n"; return; }
        current_frame_index_ = frame_data_.begin()->first;
        while (!viewer_->wasStopped()) {
            renderCurrentFrame();
            viewer_->spinOnce(30);
            if (paused_) continue;
            auto it = frame_data_.upper_bound(current_frame_index_);
            current_frame_index_ = (it==frame_data_.end()) ? frame_data_.begin()->first : it->first;
        }
    }

    void KeyboardEvent(const pcl::visualization::KeyboardEvent& e) {
        if (!e.keyDown()) return;
        auto k = e.getKeySym();
        auto clamp01 = [](double v){ return std::max(0.0, std::min(1.0, v)); };

        if (k=="space") paused_ = !paused_;
        else if (k=="Right" || k=="d") { auto it=frame_data_.upper_bound(current_frame_index_); if (it!=frame_data_.end()) current_frame_index_=it->first; }
        else if (k=="Left") { auto it=frame_data_.lower_bound(current_frame_index_); if (it==frame_data_.begin()) current_frame_index_=frame_data_.begin()->first; else {--it; current_frame_index_=it->first;} }
        else if (k=="e") exporting_enabled_runtime_ = !exporting_enabled_runtime_;
        else if (k=="s") viewer_->saveScreenshot("viewer_capture.png");
        else if (k=="r") viewer_->resetCamera();
        else if (k=="Escape") std::exit(0);

        if (k=="period") { prob_thres_ = clamp01(prob_thres_ + 0.02); std::cout << "Prob Thres: " << prob_thres_ << std::endl; }
        if (k=="comma")  { prob_thres_ = clamp01(prob_thres_ - 0.02); std::cout << "Prob Thres: " << prob_thres_ << std::endl; }
    }

private:
    AppConfig cfg_;
    SE3f T_radar_, T_lidar_;

    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
    pcl::visualization::ImageViewer::Ptr cam_viewer_;
    bool paused_ = true;

    std::map<int, std::vector<RadarObject>> frame_data_;
    std::map<int,int> radar_to_lidar_;
    std::map<int,int> radar_to_cam_;
    int current_frame_index_ = 0;

    // runtime
    double prob_thres_ = 0.6;
    Eigen::Vector2d estimated_ego_velocity_ = Eigen::Vector2d::Zero();

    // EKF
    MatrixXd A_, H_, Q_, R_;

    // XGBoost
    BoosterHandle booster_{nullptr};
    VectorXd mean_, scale_;
    vector<int> selected_features_;
    unordered_map<string,int> feature_name_to_index_;

    // tracking
    int next_internal_id_ = 1000;
    unordered_map<int,int> radar_id_to_internal_id_, internal_id_to_radar_id_;
    unordered_map<int, deque<Vector2d>> position_history_, velocity_history_;
    unordered_map<int, TrackingState> tracked_objects_; 
    unordered_map<int,bool> latch_red_;
    unordered_map<int,int>  miss_count_;

    ofstream export_stream_; set<int> exported_frames_; bool exporting_enabled_runtime_ = true;

    void initFeatureMapping() {
        feature_name_to_index_ = {
            {"frame",0},{"id",1},{"dist_long",2},{"dist_lat",3},
            {"vrel_long",4},{"vrel_lat",5},{"rcs",6},
            {"arel_long",7},{"arel_lat",8},{"length",9},
            {"width",10},{"orientation_angle",11},{"dyn_prop",12}
        };
    }
    void initEKFMats() {
        double dt=0.1;
        A_.resize(4,4); H_.resize(2,4);
        Q_ = MatrixXd::Identity(4,4)*DEFAULT_PROCESS_NOISE;
        R_ = MatrixXd::Identity(2,2)*DEFAULT_MEAS_NOISE;
        A_ << 1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1;
        H_ << 1,0,0,0, 0,1,0,0;
    }

    static vector<string> splitCSVLine(const string& line) {
        vector<string> out; string tok; stringstream ss(line);
        while (getline(ss,tok,',')) out.push_back(tok); return out;
    }

    void loadRadarCSV() {
        ifstream fin(cfg_.radar_csv);
        if (!fin.is_open()) throw runtime_error("Cannot open radar CSV: "+cfg_.radar_csv);
        string header; if (!getline(fin, header)) throw runtime_error("Empty radar CSV");
        auto headers = splitCSVLine(header);
        unordered_map<string,int> col_idx; for (int i=0;i<(int)headers.size();++i) col_idx[headers[i]]=i;
        auto get = [&](const vector<string>& cols, const string& name, double def=0.0){
            auto it=col_idx.find(name); if (it==col_idx.end()) return def;
            if (it->second>=(int)cols.size()) return def;
            const string& s=cols[it->second]; if (s.empty()) return def;
            return stod(s);
        };
        size_t count=0; string line;
        while (getline(fin, line)) {
            if (line.empty()) continue;
            auto cols = splitCSVLine(line);
            RadarObject obj;
            obj.radar_frame = (int)get(cols,"radar_frame",0);
            obj.closest_lidar_frame = (int)get(cols,"closest_lidar_frame",0);
            obj.closest_cam_frame   = (int)get(cols,"closest_cam_frame",0);
            obj.id=(int)get(cols,"id",-1);
            obj.x=get(cols,"x",0); obj.y=get(cols,"y",0);
            obj.vx=get(cols,"vx",0); obj.vy=get(cols,"vy",0);
            obj.ax=get(cols,"ax",0); obj.ay=get(cols,"ay",0);
            obj.rcs=get(cols,"rcs",0);
            obj.length=get(cols,"length",0); obj.width=get(cols,"width",0);
            obj.orientation=get(cols,"orientation",0);
            obj.trackType=(int)get(cols,"trackType",0);
            obj.existProbability=get(cols,"existProbability",0);

            if (cfg_.orientation_in_deg) obj.orientation = obj.orientation * M_PI / 180.f;
            if (T_radar_.valid) {
                Vector3f p = apply_T(T_radar_, Vector3f(obj.x,obj.y,0));
                obj.x=p.x(); obj.y=p.y();
                Vector3f v = rotate_T(T_radar_, Vector3f(obj.vx,obj.vy,0));
                obj.vx=v.x(); obj.vy=v.y();
                Vector3f a = rotate_T(T_radar_, Vector3f(obj.ax,obj.ay,0));
                obj.ax=a.x(); obj.ay=a.y();
                float yaw_add = yaw_from_R(T_radar_.T.block<3,3>(0,0));
                obj.orientation += yaw_add;
            }
            frame_data_[obj.radar_frame].push_back(obj);
            if (!radar_to_lidar_.count(obj.radar_frame)) radar_to_lidar_[obj.radar_frame] = obj.closest_lidar_frame;
            if (!radar_to_cam_.count(obj.radar_frame)) radar_to_cam_[obj.radar_frame] = obj.closest_cam_frame;
            ++count;
        }
        cout<<"Loaded "<<frame_data_.size()<<" frames from "<<cfg_.radar_csv<<endl;
    }

    void loadXGBoost() {
        const std::string model_path  = (fs::path(cfg_.models_dir) / "xgboost_model.json").string();
        const std::string scaler_path = (fs::path(cfg_.models_dir) / "scaler.json").string();
        const std::string rfe_path    = (fs::path(cfg_.models_dir) / "RFE.json").string();

        if (XGBoosterCreate(nullptr, 0, &booster_) != 0) throw std::runtime_error("XGBoosterCreate failed");
        if (XGBoosterLoadModel(booster_, model_path.c_str()) != 0) throw std::runtime_error("Model load failed: " + model_path);

        {
            std::ifstream f(scaler_path);
            if (!f.is_open()) throw std::runtime_error("Cannot open scaler: " + scaler_path);
            json j; f >> j;
            vector<double> mv, sv;
            for (auto& v : j["mean"]) mv.push_back(v.get<double>());
            for (auto& v : j["scale"]) sv.push_back(v.get<double>());
            mean_  = Eigen::Map<Eigen::VectorXd>(mv.data(), mv.size());
            scale_ = Eigen::Map<Eigen::VectorXd>(sv.data(), sv.size());
        }
        {
            selected_features_.clear();
            std::ifstream f(rfe_path);
            json j;
            if (f.is_open()) {
                f >> j;
                for (auto& it : j) {
                    std::string s = it.get<std::string>();
                    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF) s.erase(0, 3);
                    if (feature_name_to_index_.count(s)) selected_features_.push_back(feature_name_to_index_[s]);
                }
            }
            if (selected_features_.size() != 10) {
                vector<string> fb={"dist_long","dist_lat","vrel_long","vrel_lat","rcs","dyn_prop","arel_long","arel_lat","orientation_angle","length"};
                selected_features_.clear();
                for(auto& s : fb) selected_features_.push_back(feature_name_to_index_[s]);
            }
        }
    }

    float predictObject(const RadarObject& o) {
        VectorXd in(13);
        double dyn_prop = 0.0;
        if (cfg_.dyn_prop_src==DynPropSource::TrackType) dyn_prop=o.trackType;
        else if (cfg_.dyn_prop_src==DynPropSource::ExistProb) dyn_prop=o.existProbability;
        in << 0, o.id, o.x,o.y, o.vx,o.vy, o.rcs, o.ax,o.ay, o.length,o.width, o.orientation, dyn_prop;

        vector<string> cols = {"dist_long","dist_lat","vrel_long","vrel_lat","rcs","dyn_prop","arel_long","arel_lat","orientation_angle","length","width"};
        MatrixXd proc = in.transpose();
        for (size_t i=0;i<cols.size();++i) { int idx=feature_name_to_index_[cols[i]]; proc(0,idx)=(proc(0,idx)-mean_[i])/scale_[i]; }
        MatrixXd fin(1, selected_features_.size());
        for (size_t i=0;i<selected_features_.size();++i) fin(0,i)=proc(0, selected_features_[i]);

        vector<float> mat(fin.data(), fin.data()+fin.size());
        DMatrixHandle dmat=nullptr; XGDMatrixCreateFromMat(mat.data(), fin.rows(), fin.cols(), NAN, &dmat);
        bst_ulong out_len=0; const float* out=nullptr; XGBoosterPredict(booster_, dmat, 0,0,0, &out_len,&out);
        float prob = (out && out_len>0) ? out[0] : 0.0f; XGDMatrixFree(dmat); return prob;
    }

    int reIdentifyTarget(const RadarObject& o) {
        int rid=o.id; if(radar_id_to_internal_id_.count(rid)) return radar_id_to_internal_id_[rid];
        Vector2d cp(o.x,o.y), cv(o.vx,o.vy);
        for (const auto& kv:internal_id_to_radar_id_) {
            int iid=kv.first;
            if (!position_history_[iid].empty()) {
                Vector2d lp=position_history_[iid].back(); 
                Vector2d lv=velocity_history_[iid].empty()?Vector2d(0,0):velocity_history_[iid].back();
                if ((cp-lp).norm()<1.5 && (cv-lv).norm()<0.8) { 
                    radar_id_to_internal_id_[rid]=iid; internal_id_to_radar_id_[iid]=rid; return iid; 
                }
            }
        }
        int nid=next_internal_id_++; radar_id_to_internal_id_[rid]=nid; internal_id_to_radar_id_[nid]=rid;
        position_history_[nid]=deque<Vector2d>(); velocity_history_[nid]=deque<Vector2d>(); return nid;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadLidarBin(int lidar_frame) {
        if (lidar_frame <= 0) return nullptr;
        char name[64]; snprintf(name, sizeof(name), "frame_%06d.bin", lidar_frame);
        std::string path = (fs::path(cfg_.lidar_dir) / name).string();
        std::ifstream fin(path, std::ios::binary);
        if (!fin.is_open()) return nullptr;
        fin.seekg(0, std::ios::end); size_t bytes = fin.tellg(); fin.seekg(0, std::ios::beg);
        size_t npts = bytes / (sizeof(float) * 4);
        std::vector<float> buf(npts * 4); fin.read(reinterpret_cast<char*>(buf.data()), bytes);
        auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(); cloud->reserve(npts);
        for (size_t i = 0; i < npts; ++i) {
            pcl::PointXYZRGB p; Vector3f pt(buf[i*4], buf[i*4+1], buf[i*4+2]);
            if (T_lidar_.valid) pt = apply_T(T_lidar_, pt);
            p.x=pt.x(); p.y=pt.y(); p.z=pt.z(); p.r=255; p.g=255; p.b=255; cloud->push_back(p);
        }
        return cloud;
    }

    // ================= 核心算法：加权滑动窗口车速估算 (返回 bool 状态) =================
    bool estimateEgoVelocityWeightedWindow(const std::vector<RadarObject>& objs) {
        struct Vote { double vx; double weight; };
        std::vector<Vote> votes;

        // 1. 收集投票
        for (const auto& o : objs) {
            if (o.x > 2.0 && o.x < 100.0 && std::abs(o.y) > 0.5) {
                // 宽松过滤，允许弱反射点
                if (o.rcs >= cfg_.ego_rcs_min && o.length >= cfg_.ego_length_min) {
                    // 计算权重：RCS + Length
                    double w_rcs = std::max(0.0, o.rcs + 10.0) * cfg_.ego_weight_rcs; 
                    double w_len = o.length * cfg_.ego_weight_len;                   
                    double w = 1.0 + w_rcs + w_len;
                    votes.push_back({o.vx, w});
                }
            }
        }

        // 【核心修改】如果候选点不足，返回 false，后续将跳过过滤
        if ((int)votes.size() < cfg_.ego_min_candidates) {
            estimated_ego_velocity_ *= cfg_.ego_decay; 
            return false; // 指示本帧估算失败
        }

        // 2. 排序
        std::sort(votes.begin(), votes.end(), [](const Vote& a, const Vote& b){ return a.vx < b.vx; });

        // 3. 滑动窗口聚类
        double max_w = 0.0;
        double best_vx_sum = 0.0;
        double best_w_sum = 0.0;
        
        int right = 0;
        double cur_w = 0.0;
        double cur_vx_w = 0.0;

        for (int left = 0; left < (int)votes.size(); ++left) {
            if (left > 0) {
                cur_w -= votes[left-1].weight;
                cur_vx_w -= votes[left-1].vx * votes[left-1].weight;
            }
            while (right < (int)votes.size() && (votes[right].vx - votes[left].vx) <= cfg_.ego_bin_width) {
                cur_w += votes[right].weight;
                cur_vx_w += votes[right].vx * votes[right].weight;
                right++;
            }
            if (cur_w > max_w) {
                max_w = cur_w;
                best_w_sum = cur_w;
                best_vx_sum = cur_vx_w;
            }
        }

        // 4. 更新结果
        if (best_w_sum > 0.001) {
            double bg_vx = best_vx_sum / best_w_sum;
            double ego_vx = -bg_vx; 
            
            double alpha = cfg_.ego_alpha_mean;
            estimated_ego_velocity_.x() = alpha * estimated_ego_velocity_.x() + (1.0-alpha) * ego_vx;
            estimated_ego_velocity_.y() = 0;
            return true; // 估算成功
        }
        
        return false; // 权重太小，认为不可靠
    }

    void renderCurrentFrame() {
        auto it = frame_data_.find(current_frame_index_); if (it==frame_data_.end()) return;
        const auto& objs=it->second;
        cout<<"Frame "<<current_frame_index_<<" objects: "<<objs.size()<<"\n";

        // ===================== [Step 1] 软车速估算 =====================
        // ego_est_success 标志位：表示本帧是否成功估算了车速
        bool ego_est_success = false;
        
        if (cfg_.enable_soft_ego_est) {
            ego_est_success = estimateEgoVelocityWeightedWindow(objs);
        } else {
            estimated_ego_velocity_ = Eigen::Vector2d::Zero();
        }
        // =============================================================

        int lidar_idx=0; if(radar_to_lidar_.count(current_frame_index_)) lidar_idx=radar_to_lidar_[current_frame_index_];
        auto lidar_cloud = loadLidarBin(lidar_idx);
        int cam_idx=0; if(radar_to_cam_.count(current_frame_index_)) cam_idx=radar_to_cam_[current_frame_index_];

        auto radar_cloud=boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        viewer_->removeAllPointClouds(); viewer_->removeAllShapes();
        viewer_->addLine(pcl::PointXYZ(-cfg_.det_x_max, cfg_.det_y_max, 0), pcl::PointXYZ(cfg_.det_x_max, cfg_.det_y_max, 0), 0.5,0.5,0.5,"area_top");
        viewer_->addLine(pcl::PointXYZ(-cfg_.det_x_max, -cfg_.det_y_max, 0), pcl::PointXYZ(cfg_.det_x_max, -cfg_.det_y_max, 0), 0.5,0.5,0.5,"area_bottom");
        viewer_->addLine(pcl::PointXYZ( cfg_.det_x_max, -cfg_.det_y_max, 0), pcl::PointXYZ( cfg_.det_x_max, cfg_.det_y_max, 0), 0.5,0.5,0.5,"area_right");

        vector<ExportRow> rows;
        set<int> detected_iids; 
        int cnt_det=0;

        // ===================== [Step 2] 逐目标检测 =====================
        for (const auto& o:objs) {
            double x=o.x, y=o.y;
            bool in_zone = (fabs(y)<cfg_.det_y_max && x<cfg_.det_x_max);
            int iid = reIdentifyTarget(o);

            Vector2d pos(x,y); position_history_[iid].push_back(pos); if((int)position_history_[iid].size()>HISTORY_SIZE) position_history_[iid].pop_front();
            Vector2d vel(o.vx,o.vy); velocity_history_[iid].push_back(vel); if((int)velocity_history_[iid].size()>HISTORY_SIZE) velocity_history_[iid].pop_front();

            pcl::PointXYZRGB pt; pt.x=x; pt.y=y; pt.z=0;
            string status="undetected", color="green"; float prob=0.0f;

            if (!in_zone) {
                pt.r=0; pt.g=255; pt.b=0;
                radar_cloud->push_back(pt); addText3D(iid, pt);
                rows.push_back({current_frame_index_, iid, pt.x,pt.y, (float)vel.x(),(float)vel.y(), prob, "out_of_zone","green"});
                continue;
            }

            // ----------- 背景过滤逻辑 (Fail-Safe) -----------
            bool is_background = false;
            
            // 【关键逻辑】：只有当开关开启，并且本帧车速估算成功时，才进行背景过滤。
            // 如果估算失败(candidates < 2)，我们跳过过滤，防止因车速为0导致误杀真实背景(显示为运动)
            if (cfg_.enable_soft_ego_est && ego_est_success) {
                double abs_vx = o.vx + estimated_ego_velocity_.x();
                double abs_vy = o.vy + estimated_ego_velocity_.y();
                double abs_spd = std::sqrt(abs_vx*abs_vx + abs_vy*abs_vy);
                
                // 如果绝对速度很小，且RCS符合条件，认为是背景
                if (abs_spd < cfg_.static_speed_thres && o.rcs > cfg_.static_rcs_thres) {
                    is_background = true;
                }
            }
            // ----------------------------------------------

            if (is_background) {
                // 被过滤：强制概率为0
                prob = 0.0f;
            } else {
                // 未被过滤：交给 XGBoost
                prob = predictObject(o);
            }

            bool detected = (prob >= prob_thres_);
            bool xgb_strong = (prob >= cfg_.thr_high);
            bool xgb_weak   = (!xgb_strong && prob >= cfg_.thr_low);

            if (detected) {
                detected_iids.insert(iid);
                cnt_det++;

                auto it_tr=tracked_objects_.find(iid);
                if (it_tr==tracked_objects_.end()) {
                    Vector4d x0(x,y,o.vx,o.vy); auto ekf=std::make_unique<EKF>(A_,H_,Q_,R_,x0,Matrix4d::Identity());
                    tracked_objects_[iid]=TrackingState{0,std::move(ekf)};
                } else {
                    Vector2d z(x,y); it_tr->second.tracker->update(z); it_tr->second.non_detection_count=0;
                }
                latch_red_[iid] = true;
                miss_count_[iid] = 0;

                pt.r=255; pt.g=0; pt.b=0;
                status = xgb_strong? "detected_strong" : (xgb_weak? "detected_weak" : "detected");
                color  = "red";
            } else {
                if (latch_red_[iid]) { pt.r=255; pt.g=0; pt.b=0; status="latched_red"; color="red"; }
                else { pt.r=0; pt.g=255; pt.b=0; status="undetected"; color="green"; }
            }

            radar_cloud->push_back(pt); addText3D(iid, pt);
            rows.push_back({current_frame_index_, iid, pt.x,pt.y, (float)vel.x(),(float)vel.y(), prob, status, color});
        }

        // 3) 帧末丢失处理
        for (auto &kv : latch_red_) {
            int iid = kv.first; if (!kv.second) continue;
            if (detected_iids.find(iid) == detected_iids.end()) {
                miss_count_[iid] += 1;
                auto it_tr = tracked_objects_.find(iid);
                if (it_tr != tracked_objects_.end()) {
                    it_tr->second.tracker->predict();
                    it_tr->second.non_detection_count++;
                }
                if (miss_count_[iid] > cfg_.max_misses) {
                    latch_red_[iid] = false;
                    miss_count_[iid] = 0;
                    tracked_objects_.erase(iid);
                }
            }
        }

        if (lidar_cloud && !lidar_cloud->empty()) {
            viewer_->addPointCloud(lidar_cloud, "lidar");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "lidar");
        }
        viewer_->addPointCloud(radar_cloud, "radar");
        viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "radar");

        if (cam_idx>0 && !cfg_.cam_dir.empty()) {
            char nm[64]; snprintf(nm,sizeof(nm),"cam_%06d.jpg",cam_idx);
            string cam_path=(fs::path(cfg_.cam_dir)/nm).string();
            cv::Mat bgr=cv::imread(cam_path, cv::IMREAD_COLOR);
            if (!bgr.empty()) {
                if (!cam_viewer_) cam_viewer_ = boost::make_shared<pcl::visualization::ImageViewer>("Cam");
                cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
                cam_viewer_->showRGBImage(rgb.data, rgb.cols, rgb.rows, "cam");
                cam_viewer_->spinOnce(1, true);
            }
        }

        std::ostringstream oss;
        oss<<"Frame: "<<current_frame_index_
           <<" thr="<<fixed<<setprecision(2)<<prob_thres_
           <<" SoftEgo: "<<(ego_est_success ? "VALID" : (cfg_.enable_soft_ego_est ? "INVALID(Skip)" : "OFF"))
           <<" EgoV="<<estimated_ego_velocity_.x();
        viewer_->addText(oss.str(), 10, 10, 16, 1.0, 1.0, 1.0, "frame_info");

        if (export_stream_.is_open() && exporting_enabled_runtime_) {
            if (!exported_frames_.count(current_frame_index_)) {
                for (const auto& r : rows) {
                    export_stream_ << r.frame<<','<<r.internal_id<<','<<r.x<<','<<r.y<<','<<r.vx<<','<<r.vy<<','<<r.prob<<','<<r.status<<','<<r.color<<'\n';
                }
                export_stream_.flush();
                exported_frames_.insert(current_frame_index_);
            }
        }
    }

    void addText3D(int iid, const pcl::PointXYZRGB& p) {
        string key="id_"+to_string(iid)+"_"+to_string(rand());
        viewer_->addText3D(to_string(iid), p, 0.5, 1.0, 1.0, 1.0, key);
    }
};

// ===================== Loader Logic =====================
AppConfig loadConfig(const std::string& path) {
    AppConfig c;
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot open config: " << path << "\n"; return c; }
    
    try {
        json j; f >> j;
        // Paths
        if (j.contains("radar_csv")) c.radar_csv = j["radar_csv"];
        if (j.contains("lidar_dir")) c.lidar_dir = j["lidar_dir"];
        if (j.contains("cam_dir")) c.cam_dir = j["cam_dir"];
        if (j.contains("models_dir")) c.models_dir = j["models_dir"];
        if (j.contains("radar_extrinsic")) c.radar_extrinsic = j["radar_extrinsic"];
        if (j.contains("lidar_extrinsic")) c.lidar_extrinsic = j["lidar_extrinsic"];
        if (j.contains("export_csv")) c.export_csv = j["export_csv"];

        // ROI
        if (j.contains("det_x_max")) c.det_x_max = j["det_x_max"];
        if (j.contains("det_y_max")) c.det_y_max = j["det_y_max"];

        // Thresholds
        if (j.contains("prob_thres")) c.prob_thres = j["prob_thres"];
        if (j.contains("thr_high")) c.thr_high = j["thr_high"];
        if (j.contains("thr_low")) c.thr_low = j["thr_low"];
        if (j.contains("max_misses")) c.max_misses = j["max_misses"];
        if (j.contains("orientation_in_deg")) c.orientation_in_deg = j["orientation_in_deg"];
        
        if (j.contains("dyn_prop")) {
            std::string s = j["dyn_prop"];
            if (s == "trackType") c.dyn_prop_src = DynPropSource::TrackType;
            else if (s == "existProb") c.dyn_prop_src = DynPropSource::ExistProb;
            else c.dyn_prop_src = DynPropSource::Zero;
        }

        // Soft Ego
        if (j.contains("enable_soft_ego_est")) c.enable_soft_ego_est = j["enable_soft_ego_est"];
        if (j.contains("ego_rcs_min")) c.ego_rcs_min = j["ego_rcs_min"];
        if (j.contains("ego_length_min")) c.ego_length_min = j["ego_length_min"];
        if (j.contains("ego_speed_abs_max")) c.ego_speed_abs_max = j["ego_speed_abs_max"];
        if (j.contains("ego_min_candidates")) c.ego_min_candidates = j["ego_min_candidates"];
        if (j.contains("ego_bin_width")) c.ego_bin_width = j["ego_bin_width"];
        if (j.contains("ego_weight_rcs")) c.ego_weight_rcs = j["ego_weight_rcs"];
        if (j.contains("ego_weight_len")) c.ego_weight_len = j["ego_weight_len"];
        if (j.contains("ego_alpha_mean")) c.ego_alpha_mean = j["ego_alpha_mean"];
        if (j.contains("ego_decay")) c.ego_decay = j["ego_decay"];

        // Static Filter
        if (j.contains("static_speed_thres")) c.static_speed_thres = j["static_speed_thres"];
        if (j.contains("static_rcs_thres")) c.static_rcs_thres = j["static_rcs_thres"];

    } catch (const std::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << "\n";
    }
    return c;
}

static void printUsage(const char* argv0) {
    std::cout << "Usage:\n" << argv0 << " --config config.json\n";
}

int main(int argc, char** argv) {
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i+1 < argc) config_path = argv[++i];
        else if (a == "--help" || a == "-h") { printUsage(argv[0]); return 0; }
    }

    if (config_path.empty()) {
        std::cerr << "Error: Please specify a config file with --config\n";
        return 1;
    }

    AppConfig config = loadConfig(config_path);
    if (config.radar_csv.empty() || config.models_dir.empty()) {
        std::cerr << "Error: Invalid config (missing radar_csv or models_dir)\n";
        return 1;
    }

    try {
        SE3f T_radar = load_extrinsic_json(config.radar_extrinsic);
        SE3f T_lidar = load_extrinsic_json(config.lidar_extrinsic);
        RadarLidarViewer app(config, T_radar, T_lidar);
        app.Run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
