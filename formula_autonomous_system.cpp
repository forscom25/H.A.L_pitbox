/**
 * @file formula_autonomous_system.cpp
 * @author Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * @author MinKyu Cho (chomk2000@hanyang.ac.kr)
 * @brief 
 * @version 0.1
 * @date 2025-07-21
 * 
 * @copyright Copyright (c) 2025
 */

#include "formula_autonomous_system.hpp"

// ==================== Algorithm ====================

// =================================================================================================
// ========================================= 1. PERCEPTION =========================================
// =================================================================================================

// =================== RoiExtractor Implementation ===================

RoiExtractor::RoiExtractor(const std::shared_ptr<PerceptionParams> params)
    : params_(params) {
}

void RoiExtractor::extractRoi(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& roi_cloud) {
    roi_cloud->clear();
    roi_cloud->header = input_cloud->header;

    for (const auto& point : input_cloud->points) {
        
        // Check if point is within ROI
        if (point.x > params_->lidar_roi_x_min_ && point.x < params_->lidar_roi_x_max_ &&
            point.y > params_->lidar_roi_y_min_ && point.y < params_->lidar_roi_y_max_ &&
            point.z > params_->lidar_roi_z_min_ && point.z < params_->lidar_roi_z_max_) {
            roi_cloud->points.push_back(point);
        }
    }
}

// =================== GroundRemoval Implementation ===================

GroundRemoval::GroundRemoval(const std::shared_ptr<PerceptionParams> params) 
    : params_(params), rng_(std::random_device{}()) {
}

GroundRemoval::GroundRemoval(double distance_threshold, int max_iterations) 
    : rng_(std::random_device{}()) {
    // Create parameters from legacy constructor
    params_->lidar_ransac_distance_threshold_ = distance_threshold;
    params_->lidar_ransac_iterations_ = max_iterations;
}

void GroundRemoval::removeGround(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_points,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground_points) {
    
    // Clear output clouds
    ground_points->clear();
    non_ground_points->clear();
    
    if (input_cloud->empty()) return;
    
    Eigen::Vector4f best_plane; // ax + by + cz + d = 0
    std::vector<int> best_inliers;
    
    if (fitPlane(input_cloud, best_plane, best_inliers)) {
        // create checklist for ground points
        std::vector<bool> is_ground(input_cloud->size(), false);
        // Mark inliers as ground points
        for (int idx : best_inliers) {
            is_ground[idx] = true;
        }
        
        // Extract ground and non-ground points
        for (size_t i = 0; i < input_cloud->size(); ++i) {
            if (is_ground[i]) {
                ground_points->push_back(input_cloud->points[i]);
            } else {
                non_ground_points->push_back(input_cloud->points[i]);
            }
        }
    } else {
        // If plane fitting fails, treat all points as ground
        *ground_points = *input_cloud;
    }
    
    ground_points->header = input_cloud->header;
    non_ground_points->header = input_cloud->header;
}

bool GroundRemoval::fitPlane(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    Eigen::Vector4f& plane_coefficients,
    std::vector<int>& inliers) {
    
    if (cloud->size() < 3) return false;
    
    // Initialize best inlier count
    int best_inlier_count = 0;
    // Random number generator for selecting points
    std::uniform_int_distribution<int> dist(0, cloud->size() - 1);
    
    // RANSAC algorithm
    for (int iter = 0; iter < params_->lidar_ransac_iterations_; ++iter) {
        // Randomly select 3 points
        std::vector<int> sample_indices(3);
        for (int i = 0; i < 3; ++i) {
            sample_indices[i] = dist(rng_);
        }
        
        // Calculate plane from 3 points
        const auto& p1 = cloud->points[sample_indices[0]];
        const auto& p2 = cloud->points[sample_indices[1]];
        const auto& p3 = cloud->points[sample_indices[2]];
        
        // Calculate normal vector
        Eigen::Vector3f v1(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z); // Vector from p1 to p2
        Eigen::Vector3f v2(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z); // Vector from p1 to p3
        Eigen::Vector3f normal = v1.cross(v2);
        if (normal.norm() < 1e-6) continue; // Skip degenerate case
        normal.normalize();
        
        float d = -(normal.x() * p1.x + normal.y() * p1.y + normal.z() * p1.z);
        Eigen::Vector4f current_plane(normal.x(), normal.y(), normal.z(), d);
        
        // Count inliers
        std::vector<int> current_inliers;
        for (size_t i = 0; i < cloud->size(); ++i) {
            if (pointToPlaneDistance(cloud->points[i], current_plane) < params_->lidar_ransac_distance_threshold_) {
                current_inliers.push_back(i);
            }
        }
        
        // Update best plane if current plane has more inliers
        if (current_inliers.size() > best_inlier_count) {
            best_inlier_count = current_inliers.size();
            plane_coefficients = current_plane;
            inliers = current_inliers;
        }
    }
    return best_inlier_count > 0; // If the number of inliers is greater than 0, return true
}

double GroundRemoval::pointToPlaneDistance(const pcl::PointXYZ& point, const Eigen::Vector4f& plane) {
    return std::abs(plane[0] * point.x + plane[1] * point.y + plane[2] * point.z + plane[3]);
}

// =================== Clustering Implementation ===================

Clustering::Clustering(const std::shared_ptr<PerceptionParams> params)
    : params_(params) {
    vehicle_to_lidar_transform_ = Eigen::Matrix4f::Identity();
    double x = params_->lidar_translation_[0];
    double y = params_->lidar_translation_[1];
    double z = params_->lidar_translation_[2];
    double roll = params_->lidar_rotation_[0];
    double pitch = params_->lidar_rotation_[1];
    double yaw = params_->lidar_rotation_[2];

    double cr = cos(roll), sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw), sy = sin(yaw);

    // Create transformation matrix from vehicle base to lidar
    vehicle_to_lidar_transform_ << 
        cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x,
        sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y,
        -sp,   cp*sr,            cp*cr,            z,
        0,    0,                0,                1;
    
    std::cout << "Vehicle-to-Lidar transformation matrix:" << std::endl << vehicle_to_lidar_transform_ << std::endl;

}

Clustering::Clustering(double eps, int min_points, double min_cone_height, double max_cone_height) {
    // Create parameters from legacy constructor
    params_->lidar_dbscan_eps_ = eps;
    params_->lidar_dbscan_min_points_ = min_points;
    params_->lidar_cone_detection_min_height_ = min_cone_height;
    params_->lidar_cone_detection_max_height_ = max_cone_height;
    params_->lidar_cone_detection_min_points_ = 5;  // Default value
}

bool Clustering::extractCones(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_points, std::vector<Cone>& cones) {
    if (input_points->empty()) return false;
    
    // Perform DBSCAN clustering: cluster = [cluster_indices]
    std::vector<std::vector<int>> clusters = dbscan(input_points);
    cones.clear();
    cones.reserve(clusters.size());
    
    // Extract cones from clusters
    for (const auto& cluster : clusters) {
        if (isValidCone(cluster, input_points)) {
            Cone cone;
            // Calculate centroid in lidar frame
            Eigen::Vector3f centroid = calculateCentroid(cluster, input_points);
            
            // Convert centroid to vehicle base frame
            Eigen::Vector4f centroid_4d(centroid.x(), centroid.y(), centroid.z(), 1.0f);
            Eigen::Vector4f centroid_vehicle = vehicle_to_lidar_transform_ * centroid_4d;
            
            // Assign centroid to cone
            cone.center = pcl::PointXYZ(centroid_vehicle.x(), centroid_vehicle.y(), centroid_vehicle.z());
            cone.color = "unknown"; // Initial color is unknown
            cone.confidence = std::min(1.0f, static_cast<float>(cluster.size()) / 50.0f);
            
            // Store cluster points
            for (int idx : cluster) {
                cone.points.push_back(input_points->points[idx]);
            }
            
            cones.push_back(cone);
        }
    }
    
    return true;
}

std::vector<std::vector<int>> Clustering::dbscan(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<std::vector<int>> clusters;
    std::vector<bool> visited(cloud->size(), false);
    std::vector<bool> clustered(cloud->size(), false);

    // Create KD-tree for efficient neighbor search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(cloud);
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (visited[i]) continue;        // Skip if already visited
        visited[i] = true;
        std::vector<int> neighbors_indices = regionQuery(cloud, i, kdtree); // Find neighbors

        if (neighbors_indices.size() < params_->lidar_dbscan_min_points_) {
            continue; // Skip if less than min points
        }
        
        // Start new cluster
        std::vector<int> cluster;
        cluster.push_back(i);
        clustered[i] = true;
        // Expand cluster
        for (size_t j = 0; j < neighbors_indices.size(); ++j) {
            int neighbor_idx = neighbors_indices[j];
            
            if (!visited[neighbor_idx]) {
                visited[neighbor_idx] = true;
                std::vector<int> neighbor_neighbors = regionQuery(cloud, neighbor_idx, kdtree);
                // Add neighbor to cluster if it has more than min points
                if (neighbor_neighbors.size() >= params_->lidar_dbscan_min_points_) {
                    neighbors_indices.insert(neighbors_indices.end(), neighbor_neighbors.begin(), neighbor_neighbors.end());
                }
            }
            // Add neighbor to cluster if not already clustered
            if (!clustered[neighbor_idx]) {
                cluster.push_back(neighbor_idx);
                clustered[neighbor_idx] = true;
            }
        }
        
        clusters.push_back(cluster);
    }
    
    return clusters;
}

std::vector<int> Clustering::regionQuery(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    int point_idx,
    const pcl::search::KdTree<pcl::PointXYZ>::Ptr& kdtree) {
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    kdtree->radiusSearch(point_idx, params_->lidar_dbscan_eps_, indices, distances);
    
    return indices;
}

bool Clustering::isValidCone(const std::vector<int>& cluster_indices, 
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    bool is_valid = true;
    if (cluster_indices.size() < params_->lidar_cone_detection_min_points_) 
        is_valid = false; // Too few points
    
    // Calculate cone detection range
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();

    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();

    for (int idx : cluster_indices) {
        min_x = std::min(min_x, cloud->points[idx].x);
        max_x = std::max(max_x, cloud->points[idx].x);

        min_y = std::min(min_y, cloud->points[idx].y);
        max_y = std::max(max_y, cloud->points[idx].y);

        min_z = std::min(min_z, cloud->points[idx].z);
        max_z = std::max(max_z, cloud->points[idx].z);
    }

    float height = max_z - min_z;
    
    if (height < params_->lidar_cone_detection_min_height_ || height > params_->lidar_cone_detection_max_height_) {
        is_valid = false;
    }
    
    float x_range = max_x - min_x;
    float y_range = max_y - min_y;

    float range = std::sqrt(x_range * x_range + y_range * y_range);

    if (range < params_->lidar_cone_detection_min_radius_ || range > params_->lidar_cone_detection_max_radius_)
        is_valid = false;

    return is_valid;
}

Eigen::Vector3f Clustering::calculateCentroid(const std::vector<int>& cluster_indices,
                                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    Eigen::Vector3f centroid;
    centroid.x() = centroid.y() = centroid.z() = 0.0f;
    
    for (int idx : cluster_indices) {
        centroid.x() += cloud->points[idx].x;
        centroid.y() += cloud->points[idx].y;
        centroid.z() += cloud->points[idx].z;
    }
    
    float size = static_cast<float>(cluster_indices.size());
    centroid.x() /= size;
    centroid.y() /= size;
    centroid.z() /= size;
    
    return centroid;
} 

// =================== ColorDetection Implementation ===================

ColorDetection::ColorDetection(const std::shared_ptr<PerceptionParams>& params)
    : params_(params) {
    // Initialize camera parameters from params
    initializeCameraParameters();
    
    // Compute camera to lidar transformation
    computeCameraToLidarTransform();
    
    std::cout << "ColorDetection initialized with camera-to-lidar transformation" << std::endl;
}

void ColorDetection::initializeCameraParameters() {
    // Create camera intrinsic matrix
    camera_matrix_ = (cv::Mat_<double>(3, 3) << 
        params_->camera_fx_, 0, params_->camera_cx_,
        0, params_->camera_fy_, params_->camera_cy_,
        0, 0, 1);
    
    // Create distortion coefficients matrix
    std::cout << "Camera intrinsics initialized: fx=" << params_->camera_fx_ 
              << ", fy=" << params_->camera_fy_ 
              << ", cx=" << params_->camera_cx_ 
              << ", cy=" << params_->camera_cy_ << std::endl;
}

void ColorDetection::computeCameraToLidarTransform() {
    // Create transformation matrices from vehicle base to each sensor
    
    // Convert degrees to radians
    double deg_to_rad = CV_PI / 180.0;
    
    // Camera1 transformation (vehicle -> camera1)
    cv::Mat T_base_to_camera1 = createTransformationMatrix(
        params_->camera1_translation_[0],  // x
        params_->camera1_translation_[1],  // y
        params_->camera1_translation_[2],  // z
        params_->camera1_rotation_[0] * deg_to_rad,  // roll (deg -> rad)
        params_->camera1_rotation_[1] * deg_to_rad,  // pitch (deg -> rad)
        params_->camera1_rotation_[2] * deg_to_rad   // yaw (deg -> rad)
    );

    // Camera2 transformation (vehicle -> camera2)
    cv::Mat T_base_to_camera2 = createTransformationMatrix(
        params_->camera2_translation_[0],  // x
        params_->camera2_translation_[1],  // y
        params_->camera2_translation_[2],  // z
        params_->camera2_rotation_[0] * deg_to_rad,  // roll (deg -> rad)
        params_->camera2_rotation_[1] * deg_to_rad,  // pitch (deg -> rad)
        params_->camera2_rotation_[2] * deg_to_rad   // yaw (deg -> rad)
    );

    // Create transformation matrix for x-front, y-left, z-up_(vehicle frame) to z-front, x-right, y-down frame(camera frame)
    cv::Mat T_x_front_to_z_front = createTransformationMatrix(
        0, 0, 0,
        -90.0 * deg_to_rad, 0, -90.0 * deg_to_rad
    );

    // Combine transformations: vehicle -> camera
    T_base_to_camera1 = T_base_to_camera1 * T_x_front_to_z_front;
    T_base_to_camera2 = T_base_to_camera2 * T_x_front_to_z_front;

    // Compute inverse transformation: camera -> vehicle
    cv::Mat T_camera1_to_base = T_base_to_camera1.inv();
    cv::Mat T_camera2_to_base = T_base_to_camera2.inv();

    // Final transformation: camera1 -> vehicle
    camera1_to_base_rotation_ = T_camera1_to_base.rowRange(0, 3).colRange(0, 3);
    camera1_to_base_translation_ = T_camera1_to_base.rowRange(0, 3).col(3);
    std::cout << "Camera1-to-vehicle transformation computed:" << std::endl;
    std::cout << "Camera1 Rotation matrix:" << std::endl << camera1_to_base_rotation_ << std::endl;
    std::cout << "Camera1 Translation vector:" << std::endl << camera1_to_base_translation_ << std::endl;

    // Final transformation: camera2 -> vehicle
    camera2_to_base_rotation_ = T_camera2_to_base.rowRange(0, 3).colRange(0, 3);
    camera2_to_base_translation_ = T_camera2_to_base.rowRange(0, 3).col(3);
    std::cout << "Camera2-to-vehicle transformation computed:" << std::endl;
    std::cout << "Camera2 Rotation matrix:" << std::endl << camera2_to_base_rotation_ << std::endl;
    std::cout << "Camer2 Translation vector:" << std::endl << camera2_to_base_translation_ << std::endl;
}

cv::Mat ColorDetection::createTransformationMatrix(double x, double y, double z, double roll, double pitch, double yaw) {
    // Create transformation matrix from Euler angles (ZYX convention)
    double cr = cos(roll), sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw), sy = sin(yaw);
    
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
        cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x,
        sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y,
        -sp,   cp*sr,            cp*cr,            z,
        0,    0,                0,                1);
    
    return T;
}

cv::Mat ColorDetection::ConesColor(std::vector<Cone>& cones, sensor_msgs::Image& camera1_msg, sensor_msgs::Image& camera2_msg) {
    cv::Mat camera1_image, camera2_image;
    getCameraImage(camera1_msg, camera1_image);
    getCameraImage(camera2_msg, camera2_image);

    if (!camera1_image.empty() && !camera2_image.empty()) {
        cones = classifyConesColor(cones, camera1_image, camera2_image);
    }

    // for debugging: visualize cones
    cv::Mat debug_img1 = visualizeProjection(cones, camera1_image);
    cv::Mat debug_img2 = visualizeProjection(cones, camera2_image);
    cv::Mat combined_debug_image;

    // Check if debug images are empty
    if (!debug_img1.empty() && !debug_img2.empty()) {
        // Resize debug images to match height
        if (debug_img1.rows != debug_img2.rows) {
            // Calculate the ratio and resize debug_img2 to match debug_img1 height
            double ratio = (double)debug_img1.rows / (double)debug_img2.rows;
            int new_width = (int)((double)debug_img2.cols * ratio);
        
            //  Resize debug_img2 to match debug_img1 height
            cv::resize(debug_img2, debug_img2, cv::Size(new_width, debug_img1.rows));
        }

        cv::hconcat(debug_img1, debug_img2, combined_debug_image);
        
    // show one of the debug images if the other is empty
    } else if (!debug_img1.empty()) {
        combined_debug_image = debug_img1;
    } else if (!debug_img2.empty()) {
        combined_debug_image = debug_img2;
    }
    
    return combined_debug_image;
}

void ColorDetection::getCameraImage(sensor_msgs::Image& msg, cv::Mat& image){
    // Convert ROS Image message to OpenCV Mat
    cv_bridge::CvImagePtr cv_ptr;
    try {
        // Convert ROS message to cv_bridge format
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        // Extract OpenCV Mat from cv_bridge
        image = cv_ptr->image.clone();
        
        // Check if image is valid
        if (image.empty()) {
            ROS_WARN("Received empty camera image");
            return;
        }
        
        // Optional: Log image dimensions for debugging
        ROS_DEBUG("Camera image received: %dx%d, channels: %d", 
                  image.cols, image.rows, image.channels());
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception while converting camera image: %s", e.what());
        // Initialize empty image on error
        image = cv::Mat();
    }
    return;
}

std::vector<Cone> ColorDetection::classifyConesColor(const std::vector<Cone>& cones, const cv::Mat& rgb_image1, const cv::Mat& rgb_image2) {
    std::vector<Cone> classified_cones = cones;
    
    for (auto& cone : classified_cones) { //
        bool is_left_cone = (cone.center.y >= 0); // Positive y is left side

        if (is_left_cone) {
            // Project to camera 1 (left)
            cv::Point2f pt1 = projectToCamera(cone.center, 1);
            if (isPointInImage(pt1, rgb_image1.size())) {
                cone.color = detectConeColor(cone, rgb_image1, pt1);
                continue; // Skip if projected point is valid
            }

        // fallback to camera 2 (right) if projection fails
            cv::Point2f pt2 = projectToCamera(cone.center, 2);
            if (isPointInImage(pt2, rgb_image2.size())) {
                cone.color = detectConeColor(cone, rgb_image2, pt2);
                continue; // Skip if projected point is valid
            }
        } else {
            // Project to camera 2 (right)
            cv::Point2f pt2 = projectToCamera(cone.center, 2);
            if (isPointInImage(pt2, rgb_image2.size())) {
                cone.color = detectConeColor(cone, rgb_image2, pt2);
                continue; // Skip if projected point is valid
            }

            // fallback to camera 1 (left) if projection fails
            cv::Point2f pt1 = projectToCamera(cone.center, 1);
            if (isPointInImage(pt1, rgb_image1.size())) {
                cone.color = detectConeColor(cone, rgb_image1, pt1);
                continue; // Skip if projected point is valid
            }
        }
        // If both projections fail, set color to unknown
        cone.color = "out of image";
    }
    return classified_cones;
}

cv::Point2f ColorDetection::projectToCamera(const pcl::PointXYZ& point_3d, int camera_id) {
    // Transform from Vehicle base to camera coordinate system
    cv::Mat cone_point_in_base = (cv::Mat_<double>(3, 1) << point_3d.x, point_3d.y, point_3d.z);
    cv::Mat cone_point_in_camera;
    
    if (camera_id == 1) { // Left Camera
        cone_point_in_camera = camera1_to_base_rotation_ * cone_point_in_base + camera1_to_base_translation_;
    } else { // Right Camera (camera_id == 2)
        cone_point_in_camera = camera2_to_base_rotation_ * cone_point_in_base + camera2_to_base_translation_;
    }

    // Check if point is in front of camera
    if (cone_point_in_camera.at<double>(2, 0) <= 0) {
        return cv::Point2f(-1, -1); // Invalid projection
    }
    
    // camera coordinates (x_cam, y_cam, z_cam) for projection
    double x_cam = cone_point_in_camera.at<double>(0, 0);
    double y_cam = cone_point_in_camera.at<double>(1, 0);
    double z_cam = cone_point_in_camera.at<double>(2, 0);

    // Transform to image plane
    double x_img = x_cam / z_cam;
    double y_img = y_cam / z_cam;
    
    // Apply camera intrinsics for transform to pixel coordinates from image coordinates
    double u = params_->camera_fx_ * x_img + params_->camera_cx_;
    double v = params_->camera_fy_ * y_img + params_->camera_cy_;
    
    return cv::Point2f(static_cast<float>(u), static_cast<float>(v));
}

bool ColorDetection::isPointInImage(const cv::Point2f& point, const cv::Size& image_size) {
    return point.x >= 0 && point.x < image_size.width && 
           point.y >= 0 && point.y < image_size.height;
}

std::string ColorDetection::detectConeColor(const Cone& cone, const cv::Mat& rgb_image, const cv::Point2f& projection_point) {
    if (rgb_image.empty()) {
        std::cerr << "Warning: Empty image provided for color detection" << std::endl;
        return "unknown";
    }
    
    // Preprocess image if enabled
    cv::Mat processed_image = preprocessImage(rgb_image);
    
    // Convert to HSV for color analysis
    cv::Mat hsv_image;
    cv::cvtColor(processed_image, hsv_image, cv::COLOR_BGR2HSV);
    
    // Analyze color in window around projected point
    int window_size = params_->camera_hsv_window_size_;
    ColorConfidence confidence = analyzeColorWindow(hsv_image, projection_point, window_size);

    // Select best color based on confidence
    std::string best_color = selectBestColor(confidence);
    
    return best_color;
}

cv::Mat ColorDetection::preprocessImage(const cv::Mat& rgb_image) {
    if (!params_->camera_enable_preprocessing_) {
        return rgb_image;
    }
    
    cv::Mat processed = rgb_image.clone();
    
    // Apply Gaussian blur for noise reduction
    if (params_->camera_gaussian_blur_sigma_ > 0) {
        int kernel_size = static_cast<int>(2 * params_->camera_gaussian_blur_sigma_ * 3 + 1);
        if (kernel_size % 2 == 0) kernel_size++;
        
        cv::GaussianBlur(processed, processed, 
                        cv::Size(kernel_size, kernel_size), 
                        params_->camera_gaussian_blur_sigma_);
    }
    
    // Apply bilateral filter for edge-preserving smoothing
    if (params_->camera_bilateral_filter_d_ > 0) {
        cv::Mat temp;
        cv::bilateralFilter(processed, temp, 
                           params_->camera_bilateral_filter_d_, 
                           80, 80);
        processed = temp;
    }
    
    return processed;
}

ColorConfidence ColorDetection::analyzeColorWindow(const cv::Mat& hsv_image, const cv::Point2f& center, int window_size) {
    ColorConfidence confidence;
    
    // Get safe window around the projected point
    cv::Rect window = getSafeWindow(center, window_size, hsv_image.size());
    
    if (window.area() <= 0) {
        return confidence; // Return default (unknown = 1.0)
    }
    
    // Extract HSV region of interest
    cv::Mat hsv_roi = hsv_image(window);
    
    // Calculate confidence for each color
    confidence.yellow_confidence = static_cast<double>(countYellowPixels(hsv_roi));
    confidence.blue_confidence = static_cast<double>(countBluePixels(hsv_roi));
    confidence.orange_confidence = static_cast<double>(countOrangePixels(hsv_roi));
    
    return confidence;
}

cv::Rect ColorDetection::getSafeWindow(const cv::Point2f& center, int window_size, const cv::Size& image_size) {
    int half_size = window_size / 2;
    
    // Select window left-top corner
    int x = static_cast<int>(center.x) - half_size;
    int y = static_cast<int>(center.y) - half_size;
    
    // Clamp to image boundaries
    x = std::max(0, std::min(x, image_size.width - window_size));
    y = std::max(0, std::min(y, image_size.height - window_size));
    
    // Select window width and height
    int width = std::min(window_size, image_size.width - x);
    int height = std::min(window_size, image_size.height - y);
    
    // Return window [left, top, width, height]
    return cv::Rect(x, y, width, height);
}

int ColorDetection::countYellowPixels(const cv::Mat& hsv_roi) {
    int yellow_pixels = 0;
    
    for (int y = 0; y < hsv_roi.rows; ++y) {
        for (int x = 0; x < hsv_roi.cols; ++x) {
            cv::Vec3b hsv_pixel = hsv_roi.at<cv::Vec3b>(y, x);
            
            if (isInHSVRange(hsv_pixel, 
                           params_->camera_yellow_hue_min_, 
                           params_->camera_yellow_hue_max_,
                           params_->camera_yellow_sat_min_, 
                           params_->camera_yellow_val_min_)) {
                yellow_pixels++;
            }
        }
    }
    
    return yellow_pixels;
}

int ColorDetection::countBluePixels(const cv::Mat& hsv_roi) {
    int blue_pixels = 0;
    
    for (int y = 0; y < hsv_roi.rows; ++y) {
        for (int x = 0; x < hsv_roi.cols; ++x) {
            cv::Vec3b hsv_pixel = hsv_roi.at<cv::Vec3b>(y, x);
            
            if (isInHSVRange(hsv_pixel, 
                           params_->camera_blue_hue_min_, 
                           params_->camera_blue_hue_max_,
                           params_->camera_blue_sat_min_, 
                           params_->camera_blue_val_min_)) {
                blue_pixels++;
            }
        }
    }
    
    return blue_pixels;
}

int ColorDetection::countOrangePixels(const cv::Mat& hsv_roi) {
    int orange_pixels = 0;
    
    for (int y = 0; y < hsv_roi.rows; ++y) {
        for (int x = 0; x < hsv_roi.cols; ++x) {
            cv::Vec3b hsv_pixel = hsv_roi.at<cv::Vec3b>(y, x);
            
            if (isInHSVRange(hsv_pixel, 
                           params_->camera_orange_hue_min_, 
                           params_->camera_orange_hue_max_,
                           params_->camera_orange_sat_min_, 
                           params_->camera_orange_val_min_)) {
                orange_pixels++;
            }
        }
    }
    
    return orange_pixels;
}

bool ColorDetection::isInHSVRange(const cv::Vec3b& hsv_pixel, int hue_min, int hue_max, int sat_min, int val_min) {
    int hue = hsv_pixel[0];
    int sat = hsv_pixel[1];
    int val = hsv_pixel[2];
    
    // Handle hue wraparound (e.g., red: 170-180 and 0-10)
    bool hue_in_range;
    if (hue_min <= hue_max) {
        hue_in_range = (hue >= hue_min && hue <= hue_max);
    } else {
        hue_in_range = (hue >= hue_min || hue <= hue_max);
    }
    
    return hue_in_range && sat >= sat_min && val >= val_min;
}

std::string ColorDetection::selectBestColor(const ColorConfidence& confidence) {
    double max_confidence = 0.0;
    std::string best_color = "unknown";
    
    if (confidence.yellow_confidence > max_confidence) {
        max_confidence = confidence.yellow_confidence;
        best_color = "yellow";
    }
    
    if (confidence.blue_confidence > max_confidence) {
        max_confidence = confidence.blue_confidence;
        best_color = "blue";
    }
    
    if (confidence.orange_confidence > max_confidence) {
        max_confidence = confidence.orange_confidence;
        best_color = "orange";
    }
    
    // Require minimum confidence threshold to avoid false positives
    if (max_confidence < 0.001) {
        return "unknown";
    }
    
    return best_color;
}

cv::Mat ColorDetection::visualizeProjection(const std::vector<Cone>& cones, const cv::Mat& rgb_image) {
    cv::Mat visualization = rgb_image.clone();
    
    for (const auto& cone : cones) {
         // Determine camera based on y coordinate
        int primary_camera_id = (cone.center.y >= 0) ? 1 : 2;
        cv::Point2f projected = projectToCamera(cone.center, primary_camera_id);
        
        if (isPointInImage(projected, rgb_image.size())) {
            // Draw circle at projected position
            cv::Scalar color;
            if (cone.color == "yellow") {
                color = cv::Scalar(0, 255, 255); // Yellow in BGR
            } else if (cone.color == "blue") {
                color = cv::Scalar(255, 0, 0); // Blue in BGR
            } else if (cone.color == "orange") {
                color = cv::Scalar(0, 165, 255); // Orange in BGR
            } else {
                color = cv::Scalar(128, 128, 128); // Gray for unknown
            }

            cv::circle(visualization, projected, 10, color, 2);
            cv::putText(visualization, cone.color, 
                       cv::Point(projected.x + 15, projected.y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }
    
    return visualization;
}

// =================================================================================================
// ======================================== 2. Localization ========================================
// =================================================================================================

// =================== Localization Implementation ===================

Localization::Localization(const std::shared_ptr<LocalizationParams>& params)
    : params_(params),
      state_(8, 0.0),
      last_time_sec_(0.0),
      ref_wgs84_position_(0.0, 0.0),
      prev_gps_enu_(0.0, 0.0),
      prev_gps_time_sec_(0.0)
{
    // Set reference GPS position
    if (params_->use_user_defined_ref_wgs84_position_ == true) {
        ref_wgs84_position_ << params_->ref_wgs84_latitude_, params_->ref_wgs84_longitude_;
    }
    else {
        ref_wgs84_position_ << 0.0, 0.0;
    }
    
    // Initialize state vector [x, y, yaw, vx, vy, yawrate, ax, ay]
    state_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

/**
 * @brief Update with IMU data (acceleration, yaw rate and IMU orientation)
 * @param imu_input [ax, ay, yaw_rate]
 * @param q_imu IMU orientation
 * @param curr_time_sec Current time in seconds
 */
void Localization::updateImu(const Eigen::Vector3d& imu_input, Eigen::Quaterniond& q_imu, double curr_time_sec) {
    
    // Update state
    state_[5] = imu_input[2]; // yaw rate
    state_[6] = imu_input[0]; // acceleration
    state_[7] = imu_input[1]; // lateral acceleration

    // Update yaw angle
    double yaw_angle = atan2(2.0 * (q_imu.w() * q_imu.z() + q_imu.x() * q_imu.y()), 1.0 - 2.0 * (q_imu.y() * q_imu.y() + q_imu.z() * q_imu.z()));
    state_[2] = yaw_angle; // yaw angle
    
    // Update timing
    double dt = curr_time_sec - last_time_sec_;
    if (dt > 0.0 && last_time_sec_ > std::numeric_limits<double>::epsilon()) {
        // Predict if time is not updated
        state_ = predictState(state_, dt);
    }
    last_time_sec_ = curr_time_sec;
}

void Localization::updateGps(const Eigen::Vector2d& gps_wgs84, double curr_time_sec) {
    if (params_->use_user_defined_ref_wgs84_position_ == false) {
        if (ref_wgs84_position_.x() == 0.0 && ref_wgs84_position_.y() == 0.0) {
            ref_wgs84_position_ << gps_wgs84[0], gps_wgs84[1];
        }
    }

    // Update timing
    double dt = curr_time_sec - last_time_sec_;
    if (dt > 0.0 && last_time_sec_ > std::numeric_limits<double>::epsilon()) {
        // Predict if time is not updated
        state_ = predictState(state_, dt);
        last_time_sec_ = curr_time_sec;
    }
    
    // Convert GPS to ENU coordinates
    Eigen::Vector2d gps_enu = wgs84ToEnu(gps_wgs84);
    
    // Update step with GPS measurement
    state_[0] = gps_enu[0]; // x
    state_[1] = gps_enu[1]; // y

    // Correct velocity with complementary filter(GPS + IMU)
    double dt_gps = curr_time_sec - prev_gps_time_sec_;
    if (dt_gps > 0.0 && curr_time_sec > std::numeric_limits<double>::epsilon()) {
        double dx = state_[0] - prev_gps_enu_[0];
        double dy = state_[1] - prev_gps_enu_[1];
        prev_gps_enu_ << state_[0], state_[1];

        // Calculate velocity in ENU frame
        double vx_enu = dx / dt_gps;
        double vy_enu = dy / dt_gps;

        // Transform velocity to vehicle frame
        double vx_vehicle =  vx_enu * cos(-state_[2]) - vy_enu * sin(-state_[2]);
        double vy_vehicle =  vx_enu * sin(-state_[2]) + vy_enu * cos(-state_[2]);

        // Recall IMU velocity
        double predicted_vx = state_[3];
        double predicted_vy = state_[4];

        // Apply complementary filter to fuse GPS and IMU velocities
        state_[3] = (1.0 - params_->gps_correction_gain_) * predicted_vx + params_->gps_correction_gain_ * vx_vehicle; // longitudinal velocity
        state_[4] = (1.0 - params_->gps_correction_gain_) * predicted_vy + params_->gps_correction_gain_ * vy_vehicle; // lateral velocity

        // Update previous GPS position
        prev_gps_enu_ << state_[0], state_[1];
    }
    prev_gps_time_sec_ = curr_time_sec;
}

std::vector<double> Localization::predictState(const std::vector<double>& state, double dt) {
    double x = state[0];
    double y = state[1];
    double yaw = state[2];
    double vx = state[3];
    double vy = state[4];
    double yaw_rate = state[5];
    double ax = state[6];
    double ay = state[7];

    double yaw_middle = yaw + yaw_rate * dt * 0.5;

    double new_x = x + vx * cos(yaw_middle) * dt + 0.5 * ax * cos(yaw_middle) * dt * dt;
    double new_y = y + vx * sin(yaw_middle) * dt + 0.5 * ax * sin(yaw_middle) * dt * dt;
    double new_yaw = yaw + yaw_rate * dt;
    double new_vx = vx + ax * dt;
    double new_vy = vy + ay * dt;

    std::vector<double> new_state = {new_x, new_y, new_yaw, new_vx, new_vy, yaw_rate, ax, ay};

    return new_state;
}

Eigen::Vector2d Localization::wgs84ToEnu(const Eigen::Vector2d& wgs84_pos) const {
    // Convert WGS84 (lat, lon) to ENU coordinates
    double lat_rad = wgs84_pos[0] * DEG_TO_RAD;
    double lon_rad = wgs84_pos[1] * DEG_TO_RAD;
    double ref_lat_rad = ref_wgs84_position_[0] * DEG_TO_RAD;
    double ref_lon_rad = ref_wgs84_position_[1] * DEG_TO_RAD;
    
    double dlat = lat_rad - ref_lat_rad;
    double dlon = lon_rad - ref_lon_rad;
    
    // Convert WGS84 to ENU coordinates: x = (lon - ref_lon) * EARTH_RADIUS * cos(ref_lat), y = (lat - ref_lat) * EARTH_RADIUS
    double x = EARTH_RADIUS * dlon * cos(ref_lat_rad); // cos(ref_lat_rad) to account for latitude scaling
    double y = EARTH_RADIUS * dlat;
    
    return Eigen::Vector2d(x, y);
}

// ================================================================================================
// ========================================== 3. Mapping ==========================================
// ================================================================================================

// =================== Mapping Implementation ====================

MapManager::MapManager(const std::shared_ptr<MappingParams>& params) : params_(params){}

std::vector<Cone> MapManager::updateAndGetPlannedCones(const VehicleState& current_state, const std::vector<Cone>& real_time_cones) {
    {
        // mutex lock to protect cone memory
        std::lock_guard<std::mutex> lock(cone_memory_mutex_);
        
        // Current vehicle position and orientation (global frame)
        double vehicle_x = current_state.position.x();
        double vehicle_y = current_state.position.y();
        double vehicle_yaw = current_state.yaw;

        for (const auto& local_cone : real_time_cones) {
            Cone global_cone = local_cone;
            global_cone.center.x = vehicle_x + local_cone.center.x * std::cos(vehicle_yaw) - local_cone.center.y * std::sin(vehicle_yaw);
            global_cone.center.y = vehicle_y + local_cone.center.x * std::sin(vehicle_yaw) + local_cone.center.y * std::cos(vehicle_yaw);

            bool found_in_memory = false;

            for (auto& mem_cone : cone_memory_) {
                double dist = std::hypot(global_cone.center.x - mem_cone.center.x, global_cone.center.y - mem_cone.center.y);

                if (dist < params_->cone_memory_association_threshold_) { // Flag to check if cone is already in memory
                    mem_cone.center.x = (mem_cone.center.x * 0.9) + (global_cone.center.x * 0.1);
                    mem_cone.center.y = (mem_cone.center.y * 0.9) + (global_cone.center.y * 0.1);
                    found_in_memory = true;
                    break;
                }
            }

            // Update cone memory only if not found
            if (!found_in_memory) {
                cone_memory_.push_back(global_cone);
            }
        }
    }

    //
    std::vector<Cone> cones_for_planning = real_time_cones;
    {
        // mutex lock to protect cone memory
        std::lock_guard<std::mutex> lock(cone_memory_mutex_);

        // Vehicle position and orientation (global frame)
        double vehicle_x = current_state.position.x();
        double vehicle_y = current_state.position.y();
        double vehicle_yaw = current_state.yaw;

        for (const auto& global_cone : cone_memory_) {
            double dist_to_car = std::hypot(global_cone.center.x - vehicle_x, global_cone.center.y - vehicle_y);

            // Check if the cone is within the search radius and in front of the vehicle
            if (dist_to_car < params_->cone_memory_search_radius_) {
                Cone local_cone = global_cone;
                double dx = global_cone.center.x - vehicle_x;
                double dy = global_cone.center.y - vehicle_y;
                local_cone.center.x = dx * std::cos(-vehicle_yaw) - dy * std::sin(-vehicle_yaw);
                local_cone.center.y = dx * std::sin(-vehicle_yaw) + dy * std::cos(-vehicle_yaw);
                
                // Only consider cones in front of the vehicle
                if (local_cone.center.x > 0) {
                    cones_for_planning.push_back(local_cone);
                }
            }
        }
    }
    return cones_for_planning;
}

std::vector<Cone> MapManager::getGlobalConeMap() const {
    std::lock_guard<std::mutex> lock(cone_memory_mutex_);
    return cone_memory_; // Return a copy for thread safety
}

std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>> MapManager::getTrackLanes() {
    std::lock_guard<std::mutex> lock(cone_memory_mutex_);
    return {left_lane_points_, right_lane_points_};
}

std::vector<Eigen::Vector2d> MapManager::sortConesByProximity(const std::vector<Eigen::Vector2d>& cones) {
    if (cones.size() < 2) {
        return cones; // No sorting needed
    }

    std::vector<Eigen::Vector2d> sorted_cones;
    std::vector<Eigen::Vector2d> remaining_cones = cones;

    const double max_connection_distance = params_->max_connection_distance_;
    const double direction_weight = params_->direction_weight_;

    // Start from the cone with the smallest x coordinate (closest to the vehicle)
    auto start_it = std::min_element(remaining_cones.begin(), remaining_cones.end(),
        [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
            return a.x() < b.x();
        });

    sorted_cones.push_back(*start_it);
    remaining_cones.erase(start_it);

    // Vehicle heading(x) as initial direction
    Eigen::Vector2d current_direction(1.0, 0.0);
    
    // Repeat until all cones are sorted
    while (!remaining_cones.empty()) {

        Eigen::Vector2d current_point = sorted_cones.back();
        
        // Update direction vector
        if (sorted_cones.size() > 1) {
            current_direction = (sorted_cones.back() - sorted_cones[sorted_cones.size() - 2]).normalized();
        }

        double best_score = std::numeric_limits<double>::max();
        auto best_it = remaining_cones.begin();

        for (auto it = remaining_cones.begin(); it != remaining_cones.end(); ++it) {
            double dist = (it->x() - current_point.x()) * (it->x() - current_point.x()) +
                             (it->y() - current_point.y()) * (it->y() - current_point.y());
            dist = std::sqrt(dist);

            // noise filter
            if (dist > max_connection_distance) {
                continue;
            }

            // Cost function based on direction vectors
            Eigen::Vector2d candidate_direction = (*it - current_point).normalized();
            double dot_product = current_direction.dot(candidate_direction); // scalar product of two direction vector (correlation)

            // Cost function(lower the better)
            double score = dist * (1.0 + direction_weight * (1.0 - dot_product));

            if (score < best_score) {
                best_score = score;
                best_it = it;
            }
        }


        if (best_score == std::numeric_limits<double>::max()) {
            break;
        }

        sorted_cones.push_back(*best_it);
        remaining_cones.erase(best_it);
    }

    return sorted_cones;
}

void MapManager::generateLanesFromMemory() {

    std::lock_guard<std::mutex> lock(cone_memory_mutex_);
    std::vector<Eigen::Vector2d> blue_cones, yellow_cones;
    
    for (const auto& cone : cone_memory_) {
        if (cone.color == "blue") {
            blue_cones.emplace_back(cone.center.x, cone.center.y);
        } else if (cone.color == "yellow") {
            yellow_cones.emplace_back(cone.center.x, cone.center.y);
        }
    }

    // Sort cones by proximity
    auto sorted_blue = sortConesByProximity(blue_cones);
    auto sorted_yellow = sortConesByProximity(yellow_cones);

    // Generate spline for left lane (blue cones)
    left_lane_points_.clear();
    if (sorted_blue.size() >= 3) {
        std::vector<double> s_pts, x_pts, y_pts;
        s_pts.push_back(0.0);
        x_pts.push_back(sorted_blue[0].x());
        y_pts.push_back(sorted_blue[0].y());

        for (size_t i = 1; i < sorted_blue.size(); ++i) {
            double dist = std::hypot(sorted_blue[i].x() - sorted_blue[i-1].x(), sorted_blue[i].y() - sorted_blue[i-1].y());
            double new_s = s_pts.back() + dist;
            if (new_s <= s_pts.back()) {
                new_s = s_pts.back() + 0.01; // Ensure strictly increasing s values
            }
            s_pts.push_back(new_s);
            x_pts.push_back(sorted_blue[i].x());
            y_pts.push_back(sorted_blue[i].y());
        }

        tk::spline spline_x, spline_y;
        spline_x.set_points(s_pts, x_pts);
        spline_y.set_points(s_pts, y_pts);

        double total_length = s_pts.back();
        for (double s = 0; s < total_length; s += 0.5) {
            left_lane_points_.emplace_back(spline_x(s), spline_y(s));
        }
        left_lane_points_.push_back(sorted_blue.back());
    } else {
        left_lane_points_ = sorted_blue; // Use raw points if not enough for spline
    }

    // Generate spline for right lane (yellow cones)
    right_lane_points_.clear();
    if (sorted_yellow.size() >= 3) {
        std::vector<double> s_pts, x_pts, y_pts;
        s_pts.push_back(0.0);
        x_pts.push_back(sorted_yellow[0].x());
        y_pts.push_back(sorted_yellow[0].y());

        for (size_t i = 1; i < sorted_yellow.size(); ++i) {
            double dist = std::hypot(sorted_yellow[i].x() - sorted_yellow[i-1].x(), sorted_yellow[i].y() - sorted_yellow[i-1].y());
            double new_s = s_pts.back() + dist;
            if (new_s <= s_pts.back()) {
                new_s = s_pts.back() + 0.01; // Ensure strictly increasing s values
            }
            s_pts.push_back(new_s);
            x_pts.push_back(sorted_yellow[i].x());
            y_pts.push_back(sorted_yellow[i].y());
        }

        tk::spline spline_x, spline_y;
        spline_x.set_points(s_pts, x_pts);
        spline_y.set_points(s_pts, y_pts);

        double total_length = s_pts.back();
        for (double s = 0; s < total_length; s += 0.5) {
            right_lane_points_.emplace_back(spline_x(s), spline_y(s));
        }
        right_lane_points_.push_back(sorted_yellow.back());
    } else {
        right_lane_points_ = sorted_yellow; // Use raw points if not enough for spline
    }
}

/**
 * @brief Iterates through the cone memory to infer the color of 'unknown' cones
 * and removes cones that are far from the generated lanes, considering them as noise.
 * @details 1. Generates temporary lanes from the current map.
 * 2. For each 'unknown' cone, calculates its distance to the lanes.
 * 3. If close, assigns the color of the nearest lane; if far, discards it as noise.
 * 4. Replaces the old map with the refined map.
 */

void MapManager::refineConeMap() {
    std::lock_guard<std::mutex> lock(cone_memory_mutex_);

    // 1. Generate temporary lanes from current map (guideline)
    generateLanesFromMemory();

    // Check sufficiency
    if (left_lane_points_.size() < 2 && right_lane_points_.size() < 2) {
        ROS_WARN("MapManager: Not enough lane points to refine cone map. Skipping.");
        return;
    }

    // New memory vector for refined cone data
    std::vector<Cone> refined_cone_memory;
    const double max_dist_threshold = params_->max_dist_from_lane_; 

    // 2. Classification of every cone data
    for (const auto& cone : cone_memory_) {
        // Already classified
        if (cone.color != "unknown") {
            refined_cone_memory.push_back(cone);
            continue;
        }

        // Verify and correct the unknowns
        Eigen::Vector2d cone_pos(cone.center.x, cone.center.y);
        double min_dist_to_left_lane = std::numeric_limits<double>::max();
        double min_dist_to_right_lane = std::numeric_limits<double>::max();

        // 3. Calculate minimum distance from left(blue) lane
        if (left_lane_points_.size() >= 2) {
            for (size_t i = 0; i < left_lane_points_.size() - 1; ++i) {
                double dist = pointToLineSegmentDistance(cone_pos, left_lane_points_[i], left_lane_points_[i + 1]);
                min_dist_to_left_lane = std::min(dist, min_dist_to_left_lane);
            }
        }

        // 4. Calculate minimum distance from right(yellow) lane
        if (right_lane_points_.size() >= 2) {
            for (size_t i = 0; i < right_lane_points_.size() - 1; ++i) {
                double dist = pointToLineSegmentDistance(cone_pos, right_lane_points_[i], right_lane_points_[i + 1]);
                min_dist_to_right_lane = std::min(dist, min_dist_to_right_lane);
            }
        }

        // 5. Filter noise and decide color
        double closest_lane_dist = std::min(min_dist_to_left_lane, min_dist_to_right_lane);

        if (closest_lane_dist < max_dist_threshold) {// noise filter
            
            Cone refined_cone = cone;
            if (min_dist_to_left_lane < min_dist_to_right_lane) {
                refined_cone.color = "blue";
            } else {
                refined_cone.color = "yellow";
            }
            refined_cone_memory.push_back(refined_cone);
        } else {
            // 차선과 너무 멀리 떨어져 있다면, 노이즈로 판단하고 맵에서 제거 (새로운 메모리에 추가하지 않음)
            ROS_DEBUG("MapManager: Discarding a noisy object far from lanes at (%.2f, %.2f)", cone.center.x, cone.center.y);
        }
    }

    // 6. Update the global cone map
    cone_memory_ = refined_cone_memory;

    ROS_INFO("MapManager: Cone map refinement complete. Total cones: %zu", cone_memory_.size());
}

// 헬퍼 함수: 점과 선분 사이의 최단 거리 계산
double MapManager::pointToLineSegmentDistance(const Eigen::Vector2d& p, const Eigen::Vector2d& v, const Eigen::Vector2d& w) {
    double l2 = (v - w).squaredNorm();
    if (l2 == 0.0) return (p - v).norm();
    double t = std::max(0.0, std::min(1.0, (p - v).dot(w - v) / l2));
    Eigen::Vector2d projection = v + t * (w - v);
    return (p - projection).norm();
}

// =================================================================================================
// ========================================== 4. Planning ==========================================
// =================================================================================================

// =================== State Machine Implementation ===================

StateMachine::StateMachine()
    : current_state_(ASState::AS_OFF)
    , previous_state_(ASState::AS_OFF)
    , state_entry_time_(std::chrono::steady_clock::now())
    , last_update_time_(std::chrono::steady_clock::now())
    , current_mission_("")
    , mission_track_("")
    , mission_active_(false)
{
    // Initialize valid state transitions
    initializeValidTransitions();
    
    std::cout << "StateMachine: Initialized in AS_OFF state" << std::endl;
}

void StateMachine::initializeValidTransitions() {
    valid_transitions_.clear();
    
    // AS_OFF transitions
    valid_transitions_[{ASState::AS_OFF, ASState::AS_READY}] = true;
    
    // AS_READY transitions
    valid_transitions_[{ASState::AS_READY, ASState::AS_DRIVING}] = true;
    valid_transitions_[{ASState::AS_READY, ASState::AS_OFF}] = true;
    
    // AS_DRIVING transitions
    valid_transitions_[{ASState::AS_DRIVING, ASState::AS_OFF}] = true;
}

bool StateMachine::isValidTransition(ASState from, ASState to) const {
    auto it = valid_transitions_.find({from, to});
    return it != valid_transitions_.end() && it->second;
}

StateTransitionResult StateMachine::processEvent(ASEvent event) {
    ASState target_state = current_state_;
    std::string reason = eventToString(event);
    
    switch (event) {
        case ASEvent::SYSTEM_INIT:
            if (current_state_ == ASState::AS_OFF) {
                target_state = ASState::AS_READY;
            }
            break;
            
        case ASEvent::GO_SIGNAL:
            if (current_state_ == ASState::AS_READY) {
                target_state = ASState::AS_DRIVING;
                mission_active_ = true;
            }
            break;
        default:
            return StateTransitionResult(false, current_state_, current_state_, 
                                       "Unknown event: " + reason);
    }
    
    if (target_state != current_state_) {
        if (performStateTransition(target_state, reason)) {
            return StateTransitionResult(true, previous_state_, current_state_, reason);
        } else {
            return StateTransitionResult(false, current_state_, current_state_, 
                                       "Transition failed: " + reason);
        }
    }
    
    return StateTransitionResult(true, current_state_, current_state_, "No transition needed");
}

bool StateMachine::performStateTransition(ASState new_state, const std::string& reason) {
    if (!isValidTransition(current_state_, new_state)) {
        std::cout << "StateMachine: Invalid transition from " << stateToString(current_state_) 
                  << " to " << stateToString(new_state) << std::endl;
        return false;
    }
    
    // Exit current state
    bool exit_success = true;
    switch (current_state_) {
        case ASState::AS_OFF: exit_success = exitAS_OFF(); break;
        case ASState::AS_READY: exit_success = exitAS_READY(); break;
        case ASState::AS_DRIVING: exit_success = exitAS_DRIVING(); break;
    }
    
    if (!exit_success) {
        std::cout << "StateMachine: Failed to exit state " << stateToString(current_state_) << std::endl;
        return false;
    }
    
    // Update state
    previous_state_ = current_state_;
    current_state_ = new_state;
    state_entry_time_ = std::chrono::steady_clock::now();
    
    // Enter new state
    bool enter_success = true;
    switch (new_state) {
        case ASState::AS_OFF: enter_success = enterAS_OFF(); break;
        case ASState::AS_READY: enter_success = enterAS_READY(); break;
        case ASState::AS_DRIVING: enter_success = enterAS_DRIVING(); break;
    }
    
    logStateTransition(previous_state_, current_state_, reason);
    
    return enter_success;
}

// State entry functions
bool StateMachine::enterAS_OFF() {
    std::cout << "StateMachine: Entering AS_OFF state" << std::endl;
    mission_active_ = false;
    return true;
}

bool StateMachine::enterAS_READY() {
    std::cout << "StateMachine: Entering AS_READY state" << std::endl;
    return true;
}

bool StateMachine::enterAS_DRIVING() {
    std::cout << "StateMachine: Entering AS_DRIVING state" << std::endl;
    mission_active_ = true;
    return true;
}

// State exit functions
bool StateMachine::exitAS_OFF() { return true; }
bool StateMachine::exitAS_READY() { return true; }
bool StateMachine::exitAS_DRIVING() { return true; }

void StateMachine::injectSystemInit() {
    processEvent(ASEvent::SYSTEM_INIT);
}

void StateMachine::injectGoSignal(const std::string& mission, const std::string& track) {
    current_mission_ = mission;
    mission_track_ = track;
    processEvent(ASEvent::GO_SIGNAL);
}

void StateMachine::printStateInfo() const {
    std::cout << "=== State Machine Status ===" << std::endl;
    std::cout << "Current State: " << getCurrentStateString() << std::endl;
    std::cout << "Previous State: " << stateToString(previous_state_) << std::endl;
    std::cout << "Time in State: " << getTimeInCurrentState() << " seconds" << std::endl;
    std::cout << "Mission: " << current_mission_ << " (Active: " << (mission_active_ ? "Yes" : "No") << ")" << std::endl;
    std::cout << "=========================" << std::endl;
}

double StateMachine::getTimeInCurrentState() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - state_entry_time_);
    return elapsed.count() / 1000.0;
}

std::string StateMachine::stateToString(ASState state) const {
    switch (state) {
        case ASState::AS_OFF: return "AS_OFF";
        case ASState::AS_READY: return "AS_READY";
        case ASState::AS_DRIVING: return "AS_DRIVING";
        default: return "UNKNOWN";
    }
}

std::string StateMachine::eventToString(ASEvent event) const {
    switch (event) {
        case ASEvent::SYSTEM_INIT: return "SYSTEM_INIT";
        case ASEvent::SYSTEM_READY: return "SYSTEM_READY";
        case ASEvent::GO_SIGNAL: return "GO_SIGNAL";
        default: return "UNKNOWN_EVENT";
    }
}

void StateMachine::logStateTransition(ASState from, ASState to, const std::string& reason) {
    std::cout << "StateMachine: " << stateToString(from) << " -> " << stateToString(to) 
              << " (Reason: " << reason << ")" << std::endl;
} 

// ================== Trajectory Generator Implementation ===================

TrajectoryGenerator::TrajectoryGenerator(const std::shared_ptr<PlanningParams>& params)
    : params_(params),
      generated_trajectories_(0),
      average_generation_time_(0.0),
      last_generation_time_(0.0)
{
    last_trajectory_.clear();

    std::cout << "TrajectoryGenerator: Initialized with lookahead " 
              << params_->lookahead_distance_ << "m, spacing " 
              << params_->waypoint_spacing_ << "m" << std::endl;
}

std::vector<TrajectoryPoint> TrajectoryGenerator::generateTrajectory(const std::vector<Cone>& cones, ASState planning_state)
{
    last_trajectory_.clear();

    if (planning_state != ASState::AS_DRIVING) {
        return generateStopTrajectory();
    }
    
    std::vector<Eigen::Vector2d> blue_cones_local, yellow_cones_local;

    // Filter cones based on color and position
    for (const auto& cone : cones) {
        if (cone.center.x > 0.1) { // Only consider cones in front of the vehicle
            if (cone.color == "blue") {
                blue_cones_local.push_back(Eigen::Vector2d(cone.center.x, cone.center.y));
            } else if (cone.color == "yellow") {
                yellow_cones_local.push_back(Eigen::Vector2d(cone.center.x, cone.center.y));
            }
        }
    }
    
    // If no cones are detected, generate a default trajectory
    if (blue_cones_local.empty() && yellow_cones_local.empty()) {
        int num_points = static_cast<int>(params_->lookahead_distance_ / params_->waypoint_spacing_);
        for (int i = 0; i <= num_points; ++i) {
            double x = i * params_->waypoint_spacing_;
            last_trajectory_.emplace_back(x, 0.0, 0.0, 0.0, params_->default_speed_, x);
        }
        return last_trajectory_;
    }
    
    std::vector<Eigen::Vector2d> path_points;

    // Start with the vehicle's current position
    path_points.push_back(Eigen::Vector2d(0.0, 0.0));

    int num_points = static_cast<int>(params_->lookahead_distance_ / params_->waypoint_spacing_);
    for (int i = 1; i <= num_points; ++i) {
        double target_x = i * params_->waypoint_spacing_;
        
        Eigen::Vector2d* closest_blue = nullptr;
        Eigen::Vector2d* closest_yellow = nullptr;
        double min_blue_dist = 1.5, min_yellow_dist = 1.5;
        
        // Find the closest blue and yellow cones to the target x position
        for (auto& cone_pos : blue_cones_local) {
            double dist = std::abs(cone_pos.x() - target_x);
            if (dist < min_blue_dist) {
                min_blue_dist = dist;
                closest_blue = &cone_pos;
            }
        }
        for (auto& cone_pos : yellow_cones_local) {
            double dist = std::abs(cone_pos.x() - target_x);
            if (dist < min_yellow_dist) {
                min_yellow_dist = dist;
                closest_yellow = &cone_pos;
            }
        }
        
        Eigen::Vector2d waypoint(target_x, 0.0);

        // Calculate the y position based on the closest cones
        if (closest_blue && closest_yellow) {
            waypoint.y() = (closest_blue->y() + closest_yellow->y()) * 0.5;

        } else if (closest_blue) {
            waypoint.y() = closest_blue->y() - params_->lane_offset_;

        } else if (closest_yellow) {
            waypoint.y() = closest_yellow->y() + params_->lane_offset_;

        // Extrapolation logic
        } else if (path_points.size() >= 2) {

            // Predict future trajectory based on trajectory history
            Eigen::Vector2d last_point = path_points.back();
            Eigen::Vector2d prev_point = path_points[path_points.size() - 2];

            double dx = last_point.x() - prev_point.x();
            double dy = last_point.y() - prev_point.y();
           
            if (std::abs(dx) > 1e-6) { // prevent 0 division
                double slope = dy / dx;
                // 마지막 점에서 동일한 기울기로 연장하여 y좌표 예측
                waypoint.y() = last_point.y() + slope * (target_x - last_point.x());
            } else {
                // 수직에 가까운 경우, 이전 y값을 그대로 사용
                waypoint.y() = last_point.y();
            }

        } else if (!path_points.empty()) {
            // 경로점이 하나만 있을 경우, 이전 y값을 그대로 사용 (직진)
            waypoint.y() = path_points.back().y();
        }

        path_points.push_back(waypoint);
    }

    // Use Parametric Spline
    if (path_points.size() >= 2) {

        std::vector<double> s_pts, x_pts, y_pts;

        s_pts.push_back(0.0); // 0 accumalated distance at the start
        x_pts.push_back(path_points[0].x());
        y_pts.push_back(path_points[0].y());

        // 각 경로점까지의 누적 거리(s)를 계산
        for (size_t i = 1; i < path_points.size(); ++i) {

            double dist = (path_points[i] - path_points[i - 1]).norm();

            s_pts.push_back(s_pts.back() + dist);
            x_pts.push_back(path_points[i].x());
            y_pts.push_back(path_points[i].y());
        }

        // x와 y를 각각의 스플라인으로 생성 (x = fx(s), y = fy(s))
        tk::spline spline_x, spline_y;
        spline_x.set_points(s_pts, x_pts);
        spline_y.set_points(s_pts, y_pts);

        // Generate trajectory points from the path
        double total_length = s_pts.back();
        for (double s = 0; s < total_length; s += 0.5) { // 0.5m 간격으로 점 생성
            double x = spline_x(s);
            double y = spline_y(s);

            // 1차 미분값을 이용해 경로의 접선 각도(yaw) 계산
            double dx = spline_x.deriv(1, s);
            double dy = spline_y.deriv(1, s);
            double yaw = std::atan2(dy, dx);

            // 1차, 2차 미분값을 이용해 곡률(curvature) 계산
            double ddx = spline_x.deriv(2, s);
            double ddy = spline_y.deriv(2, s);
            double curvature = std::abs(dx * ddy - dy * ddx) / std::pow(dx * dx + dy * dy, 1.5);

            // 곡률 기반 속도 계획
            double desired_speed = params_->max_speed_ / (1.0 + params_->curvature_gain_ * std::abs(curvature));
            double target_speed = std::max(params_->min_speed_, std::min(desired_speed, params_->max_speed_));

            last_trajectory_.emplace_back(x, y, yaw, curvature, target_speed, s);
        }
    }
    
    return last_trajectory_;
}

// Utility functions
double TrajectoryGenerator::calculateDistance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) const
{
    return (p1 - p2).norm();
}

double TrajectoryGenerator::calculateAngle(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) const
{
    Eigen::Vector2d diff = p2 - p1;
    return std::atan2(diff.y(), diff.x());
}

double TrajectoryGenerator::calculateCurvature(const tk::spline& s, double x) {
    double dx = s.deriv(1, x);  // first derivative (y')
    double ddx = s.deriv(2, x); // second derivative (y'')
    return std::abs(ddx) / std::pow(1 + dx * dx, 1.5);
}

void TrajectoryGenerator::printTrajectoryStats() const
{
    std::cout << "=== Trajectory Generator Statistics ===" << std::endl;
    std::cout << "Generated trajectories: " << generated_trajectories_ << std::endl;
    std::cout << "Last generation time: " << last_generation_time_ << " ms" << std::endl;
    std::cout << "Average generation time: " << average_generation_time_ << " ms" << std::endl;
    std::cout << "Last trajectory points: " << last_trajectory_.size() << std::endl;
    
    if (!last_trajectory_.empty()) {
        std::cout << "Trajectory length: " << last_trajectory_.back().s << " m" << std::endl;
        
        double avg_speed = 0.0;
        double max_curvature = 0.0;
        for (const auto& point : last_trajectory_) {
            avg_speed += point.speed;
            max_curvature = std::max(max_curvature, point.curvature);
        }
        avg_speed /= last_trajectory_.size();
        
        std::cout << "Average speed: " << avg_speed << " m/s" << std::endl;
        std::cout << "Maximum curvature: " << max_curvature << " (1/m)" << std::endl;
    }
    std::cout << "=====================================" << std::endl;
}

std::vector<TrajectoryPoint> TrajectoryGenerator::generateStopTrajectory()
{
    last_trajectory_.clear();
    // 전방 지점들에 대해 중앙점 계산
    int num_points = static_cast<int>(params_->lookahead_distance_ / params_->waypoint_spacing_);
    for (int i = 0; i < num_points; ++i) {
        double x = i * params_->waypoint_spacing_;
        last_trajectory_.emplace_back(x, 0.0, 0.0, 0.0, 0.0, x);
    }
    return last_trajectory_;
}

// ================================================================================================
// ========================================== 5. Control ==========================================
// ================================================================================================

// ================== PurePursuit Controller Implementation ===================

PurePursuit::PurePursuit(const std::shared_ptr<ControlParams>& params)
    : params_(params) {
}

double PurePursuit::calculateSteeringAngle(const VehicleState& current_state, const std::vector<TrajectoryPoint>& path) const {
    if (path.empty()) {
        return 0.0;
    }

    int target_idx = findTargetPointIndex(path);
    const Eigen::Vector2d& target_point = path[target_idx].position;

    return calculateSteeringAngleInternal(target_point);
}

int PurePursuit::findTargetPointIndex(const std::vector<TrajectoryPoint>& path) const {
    int target_idx = 0;
    for (int i = 0; i < path.size(); ++i) {
        int current_idx = i % path.size();
        double dist_from_car = path[current_idx].position.norm();
        
        if (dist_from_car >= params_->pp_lookahead_distance_) {
            target_idx = current_idx;
            break;
        }
    }
    return target_idx;
}

double PurePursuit::calculateSteeringAngleInternal(const Eigen::Vector2d& target_point) const {
    double alpha = std::atan2(target_point.y(), target_point.x() + params_->vehicle_length_ * 0.5); // local trajectory is defined at the center of the vehicle
    double delta = std::atan2(2.0 * params_->vehicle_length_ * std::sin(alpha), params_->pp_lookahead_distance_);
    
    return std::clamp(delta, -params_->pp_max_steer_angle_, params_->pp_max_steer_angle_);
}

// ================== Stanley Controller Implementation ===================

Stanley::Stanley(const std::shared_ptr<ControlParams>& params) : params_(params) {}

double Stanley::calculateSteeringAngle(const VehicleState& current_state, const std::vector<TrajectoryPoint>& path) const
{
    if (path.empty()) return 0.0;

    // Get the point of front axle
    double min_dist = std::numeric_limits<double>::max();
    int closest_idx = 0;
    Eigen::Vector2d front_axle_pos = path[0].position + 
        Eigen::Vector2d(params_->vehicle_length_ * cos(path[0].yaw) * 0.5, 
                        params_->vehicle_length_ * sin(path[0].yaw) * 0.5); // local trajectory is defined at the center of the vehicle
    
    // Find the closest point on the path to the front axle
    for (int i = 0; i < path.size(); ++i) {
        double dist = (path[i].position - front_axle_pos).norm();
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }

    // Get the target point
    const TrajectoryPoint& target_point = path[closest_idx];
    double path_yaw = target_point.yaw;

    // Calculate the cross track error
    Eigen::Vector2d error_vec = front_axle_pos - target_point.position;
    double cross_track_error = error_vec.norm();
    
    // Calculate the heading error
    double path_dx = cos(path_yaw + M_PI_2);
    double path_dy = sin(path_yaw + M_PI_2);
    if (error_vec.dot(Eigen::Vector2d(path_dx, path_dy)) < 0) {
        cross_track_error *= -1.0;
    }

    double heading_error = path_yaw;
    while (heading_error > M_PI) heading_error -= 2 * M_PI;
    while (heading_error < -M_PI) heading_error += 2 * M_PI;

    // Calculate the cross track steering, 0.1 is added to the speed to avoid division by zero
    double cross_track_steering = atan2(params_->stanley_k_gain_ * cross_track_error, current_state.speed + 0.1);

    // Calculate the steering angle
    double steering_angle = heading_error + cross_track_steering;

    // Clamp the steering angle: -max_steer_angle_ <= steering_angle <= max_steer_angle_
    return std::clamp(steering_angle, -params_->pp_max_steer_angle_, params_->pp_max_steer_angle_);
}

// ==================== PID Controller Implementation ====================

PIDController::PIDController(const std::shared_ptr<ControlParams>& params)
    : params_(params),
      kp_(params_->pid_kp_), ki_(params_->pid_ki_), kd_(params_->pid_kd_), 
      min_output_(0.0), max_output_(params_->max_throttle_),
      integral_error_(0.0), previous_error_(0.0), first_run_(true) {}

double PIDController::calculate(double setpoint, double measured_value) {
    auto current_time = std::chrono::steady_clock::now();
    
    // 첫 실행 시 dt가 비정상적으로 커지는 것을 방지
    if (first_run_) {
        last_time_ = current_time;
        first_run_ = false;
        previous_error_ = setpoint - measured_value;
        return 0.0;
    }

    // 시간 변화량(dt) 계산
    std::chrono::duration<double> delta_time = current_time - last_time_;
    double dt = delta_time.count();
    
    // dt가 0이거나 너무 작은 경우, 계산 오류를 방지
    if (dt <= 1e-6) {
        // 이전 제어값을 그대로 사용하거나 0을 반환할 수 있습니다.
        // 여기서는 P, I, D 중 P항만 계산하여 반환합니다.
        return std::clamp(kp_ * (setpoint - measured_value), min_output_, max_output_);
    }

    // 1. 비례(Proportional) 항 계산
    double error = setpoint - measured_value;
    double p_term = kp_ * error;

    // 2. 적분(Integral) 항 계산
    integral_error_ += error * dt;
    // Integral Wind-up 방지를 위해 적분항도 제한할 수 있습니다. (선택적)
    // integral_error_ = std::clamp(integral_error_, min_integral, max_integral);
    double i_term = ki_ * integral_error_;

    // 3. 미분(Derivative) 항 계산
    double derivative_error = (error - previous_error_) / dt;
    double d_term = kd_ * derivative_error;

    // 최종 제어 출력값 계산
    double output = p_term + i_term + d_term;

    // 다음 계산을 위해 현재 상태 저장
    previous_error_ = error;
    last_time_ = current_time;

    // 출력값을 지정된 범위 내로 제한(clamping)
    return std::clamp(output, min_output_, max_output_);
}

void PIDController::reset() {
    integral_error_ = 0.0;
    previous_error_ = 0.0;
    first_run_ = true;
}

// ===============================================================================================
// =================================== FormulaAutonomousSystem ===================================
// ===============================================================================================

FormulaAutonomousSystem::FormulaAutonomousSystem():
    is_initialized_(false),
    pnh_(ros::NodeHandle()),
    ground_removal_(nullptr),
    clustering_(nullptr),
    color_detection_(nullptr),
    localization_(nullptr),
    state_machine_(nullptr),
    lateral_controller_(nullptr),
    longitudinal_controller_(nullptr),
    is_start_finish_line_defined_(false),
    just_crossed_line_(false),
    current_mode_(DrivingMode::MAPPING),
    is_global_path_generated_(false),
    current_lap_(1) {
}

FormulaAutonomousSystem::~FormulaAutonomousSystem(){
    ROS_INFO("FormulaAutonomousSystem: Destructor called");
    return;
}

// =================== Initialization ====================
bool FormulaAutonomousSystem::init(ros::NodeHandle& pnh){
    pnh_ = pnh;

    // Allocate memory for parameters
    perception_params_ = std::make_shared<PerceptionParams>();
    localization_params_ = std::make_shared<LocalizationParams>();
    mapping_params_ = std::make_shared<MappingParams>();
    planning_params_ = std::make_shared<PlanningParams>();
    control_params_ = std::make_shared<ControlParams>();

    // Load parameters from Yaml file
    if (!getParameters()) {
        ROS_ERROR("FormulaAutonomousSystem: Failed to get parameters");
        return false;
    }

    // Initialize algorithm modules
    // Perception
    // LiDAR perception
    roi_extractor_ = std::make_unique<RoiExtractor>(perception_params_);
    roi_point_cloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
    ground_removal_ = std::make_unique<GroundRemoval>(perception_params_);    
    ground_point_cloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
    non_ground_point_cloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
    clustering_ = std::make_unique<Clustering>(perception_params_);

    // Camera perception
    color_detection_ = std::make_unique<ColorDetection>(perception_params_);

    // Localization
    localization_ = std::make_unique<Localization>(localization_params_);

    // Mapping
    map_manager_ = std::make_unique<MapManager>(mapping_params_);

    // Planning
    state_machine_ = std::make_unique<StateMachine>();
    trajectory_generator_ = std::make_unique<TrajectoryGenerator>(planning_params_);

    // Control
    if (control_params_->lateral_controller_type_ == "Stanley") { 
        lateral_controller_ = std::make_unique<Stanley>(control_params_);
        ROS_INFO("Lateral Controller: Stanley selected");
    } else { // Default to Pure Pursuit
        lateral_controller_ = std::make_unique<PurePursuit>(control_params_);
        ROS_INFO("Lateral Controller: PurePursuit selected");
    }
    longitudinal_controller_ = std::make_unique<PIDController>(control_params_);

    is_initialized_ = true;
    return true;
}

bool FormulaAutonomousSystem::getParameters(){
    // Get parameters
    if(perception_params_->getParameters(pnh_) == false){
        ROS_ERROR("FormulaAutonomousSystem: Failed to get perception parameters");
        return false;
    }
    if(localization_params_->getParameters(pnh_) == false){
        ROS_ERROR("FormulaAutonomousSystem: Failed to get localization parameters");
        return false;
    }
    if(mapping_params_->getParameters(pnh_) == false){
        ROS_ERROR("FormulaAutonomousSystem: Failed to get mapping parameters");
        return false;
    }
    if(planning_params_->getParameters(pnh_) == false){
        ROS_ERROR("FormulaAutonomousSystem: Failed to get planning parameters");
        return false;
    }
    if(control_params_->getParameters(pnh_) == false){
        ROS_ERROR("FormulaAutonomousSystem: Failed to get control parameters");
        return false;
    }
    return true;
}

/**
 * @brief Run the formula autonomous system
 * @param lidar_msg: (input) Lidar point cloud
 * @param camera1_msg: (input) Camera1 image
 * @param camera2_msg: (input) Camera2 image
 * @param imu_msg: (input) IMU data
 * @param gps_msg: (input) GPS data
 * @param go_signal_msg: (input) Go signal
 * @param control_command_msg: (output) Control command
 * @return true if the formula autonomous system is running, false otherwise
 */
bool FormulaAutonomousSystem::run(sensor_msgs::PointCloud2& lidar_msg,
                                    sensor_msgs::Image& camera1_msg,
                                    sensor_msgs::Image& camera2_msg,
                                    sensor_msgs::Imu& imu_msg,
                                    sensor_msgs::NavSatFix& gps_msg,
                                    fs_msgs::GoSignal& go_signal_msg,
                                    fs_msgs::ControlCommand& control_command_msg,
                                    std_msgs::String& autonomous_mode_msg){

    if(is_initialized_ == false){
        static int non_init_count = 0;
        non_init_count++;
        if(non_init_count % 1000 == 0){
            ROS_WARN("FormulaAutonomousSystem: Not initialized");
        }
        return false;
    }

    // =================================================================
    // STEP 1: UPDATE STATE - "Who am I and what should I do?"
    // =================================================================

    // Update current vehicle state with IMU and GPS data
    Eigen::Vector3d acc;    
    Eigen::Vector3d gyro;
    Eigen::Quaterniond orientation;
    getImuData(imu_msg, acc, gyro, orientation);
    localization_->updateImu(Eigen::Vector3d(acc.x(), acc.y(), gyro.z()), orientation, imu_msg.header.stamp.toSec());
    localization_->updateGps(Eigen::Vector2d(gps_msg.latitude, gps_msg.longitude), gps_msg.header.stamp.toSec());

    auto current_pose = localization_->getCurrentPose(); 
    auto current_velocity = localization_->getCurrentVelocity();
    VehicleState vehicle_state(current_pose.x(), current_pose.y(), current_pose.z(), current_velocity);

    // Update state machine with go signal
    state_machine_->injectSystemInit();
    if(go_signal_msg.mission != "None" && go_signal_msg.mission != ""){
        state_machine_->injectGoSignal(go_signal_msg.mission, go_signal_msg.track);
    }

    // Get current planning state
    planning_state_ = state_machine_->getCurrentState();

    // Update autonomous mode message
    std::string autonomous_mode = state_machine_->getCurrentStateString();
    autonomous_mode_msg.data = autonomous_mode;

    // =================================================================
    // STEP 2: CHECK DRIVING CONDITIONS - "Can I drive?"
    // =================================================================

    if (planning_state_ != ASState::AS_DRIVING) {
        control_command_msg.steering = 0.0;
        control_command_msg.throttle = 0.0;
        control_command_msg.brake = 1.0;
        return true;
    }

    // =================================================================
    // STEP 3: PERCEPTION - ""What is around me?"
    // =================================================================

    // Point cloud Update
    bool cone_updated = false;

    // Convert LiDAR point cloud from ROS message to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    getLidarPointCloud(lidar_msg, lidar_point_cloud);

    // Extract ROI from point cloud
    roi_extractor_->extractRoi(lidar_point_cloud, roi_point_cloud_);

    // Remove ground points from point cloud
    ground_removal_->removeGround(roi_point_cloud_, ground_point_cloud_, non_ground_point_cloud_);

    // Cluster the remaining points to detect cones
    clustering_->extractCones(non_ground_point_cloud_, cones_);

    // Detect cone colors using camera images
    projected_cones_image_ = color_detection_->ConesColor(cones_, camera1_msg, camera2_msg);
    
    // =================================================================
    // STEP 4: MAPPING & PLANNING - "Which way to go?"
    // =================================================================

    // Behavior planning
    setRacingStrategy(vehicle_state, cones_);

    // Behavior planning
    // setRacingStrategy(vehicle_state, cones_for_planning);

    // =================================================================
    // STEP 5: CONTROL - "How do I get there?"
    // =================================================================
    
    // 1. 횡방향 제어: 경로와 현재 상태를 기반으로 조향각 계산
    double steering_angle = lateral_controller_->calculateSteeringAngle(vehicle_state, trajectory_points_);

    // 2. 종방향 제어: 목표 속도와 현재 속도를 기반으로 스로틀 계산
    // 2-1. (기본 속도) 곡률 기반으로 계산된 경로의 목표 속도를 가져옴
    double base_target_speed = trajectory_points_[0].speed;

    // 2-2. (실시간 보정) 계산된 스티어링 각도에 비례하여 목표 속도를 추가로 감속
    double steering_dampening = std::abs(steering_angle) * control_params_->steering_based_speed_gain_;
    double final_target_speed = base_target_speed - steering_dampening;

    // 2-3. 최종 목표 속도가 planning_params_의 최소 속도보다 낮아지지 않도록 제한
    final_target_speed = std::max(planning_params_->min_speed_, final_target_speed);

    // 2-4. 최종 목표 속도를 바탕으로 PID 제어기를 통해 스로틀 계산
    double throttle = longitudinal_controller_->calculate(final_target_speed, vehicle_state.speed);

    // 3. 계산된 제어 명령을 멤버 변수에 저장
    control_command_msg.steering = -steering_angle; // FSDS 좌표계에 맞게 음수(-) 적용
    if (throttle > 0.0){
        control_command_msg.throttle = throttle;
        control_command_msg.brake = 0.0;
    }
    else{
        control_command_msg.throttle = 0.0;
        control_command_msg.brake = -throttle;
    }

    // Debug
    static int count = 0;
    count++;
    if(count % 10 == 0){
        static std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = current_time - last_time;
        double fps = (duration.count() > 0.0) ? (10.0 / duration.count()) : 0.0;
        std::cout << "count: " << count << ", \tavg time: " << duration.count()*1000 / 10 << "ms, \tavg FPS: " << fps << " Hz" << std::endl;
        last_time = current_time;
    }
    if(count > 1000){
        count = 0;
    }

    return true;
}

// =================== Getters ====================

void FormulaAutonomousSystem::getLidarPointCloud(sensor_msgs::PointCloud2& msg, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud){
    // Convert ROS PointCloud2 message to PCL point cloud
    pcl::fromROSMsg(msg, *point_cloud);
    return;
}

void FormulaAutonomousSystem::getImuData(sensor_msgs::Imu& msg, Eigen::Vector3d& acc, Eigen::Vector3d& gyro, Eigen::Quaterniond& orientation){
    // Extract IMU orientation data from ROS message
    orientation.w() = msg.orientation.w;
    orientation.x() = msg.orientation.x;
    orientation.y() = msg.orientation.y;
    orientation.z() = msg.orientation.z;
    
    // Extract raw acceleration data
    Eigen::Vector3d raw_acc;
    raw_acc.x() = msg.linear_acceleration.x;
    raw_acc.y() = msg.linear_acceleration.y;
    raw_acc.z() = msg.linear_acceleration.z;
    
    // Extract roll and pitch from quaternion orientation
    Eigen::Matrix3d rotation_matrix = orientation.toRotationMatrix();
    double roll = atan2(rotation_matrix(2,1), rotation_matrix(2,2));
    double pitch = asin(-rotation_matrix(2,0));
    
    // Create rotation matrix for roll and pitch only (yaw removed)
    Eigen::Matrix3d roll_pitch_rotation;
    roll_pitch_rotation = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    
    // Remove gravity using roll and pitch compensation
    // Gravity vector in world frame (0, 0, -9.81)
    Eigen::Vector3d gravity_world(0.0, 0.0, -9.81);
    
    // Remove gravity from raw acceleration to get linear acceleration in vehicle-aligned global frame
    acc = roll_pitch_rotation.transpose() * raw_acc - gravity_world;
    
    // Extract angular velocity data
    gyro.x() = msg.angular_velocity.x;
    gyro.y() = msg.angular_velocity.y;
    gyro.z() = msg.angular_velocity.z;
    return;
}

std::vector<Cone> FormulaAutonomousSystem::getGlobalConeMap() const {
    if (map_manager_) {
        return map_manager_->getGlobalConeMap();
    }
    return {}; // Return empty vector if map_manager_ is not initialized
}
void FormulaAutonomousSystem::setRacingStrategy(const VehicleState& vehicle_state, const std::vector<Cone>& real_time_cones) {
    // 1. 맵 매니저 업데이트: 실시간 콘으로 전역 콘 맵 업데이트
    std::vector<Cone> cones_for_planning_local = map_manager_->updateAndGetPlannedCones(vehicle_state, real_time_cones);
    
    // 2. 시작/결승선 정의 및 랩 카운트 업데이트
    if (!is_start_finish_line_defined_) {
        // 충분한 콘이 모였을 때 시작/결승선을 정의
        // 이 부분은 config.yaml의 planning/behavior/min_y_separation, max_x_separation 파라미터를 활용하여
        // 트랙 폭을 고려한 시작/결승선 로직으로 대체하는 것을 고려할 수 있습니다.
        if (map_manager_->getGlobalConeMap().size() > 50) { // 예시: 충분한 콘이 있을 때 (기준은 상황에 따라 조정)
             defineStartFinishLine(map_manager_->getGlobalConeMap());
        }
    }
    if (is_start_finish_line_defined_) {
        updateLapCount(vehicle_state); // current_lap_이 여기서 업데이트됩니다.
    }

    // 3. 모드 전환 로직 (MAPPING -> RACING)
    if (current_mode_ == DrivingMode::MAPPING) {
        // 맵핑 모드에서는 로컬 콘 맵과 차선 정보로 경로 생성
        map_manager_->generateLanesFromMemory(); // 로컬 플래닝을 위해 매번 차선 생성
        trajectory_points_ = trajectory_generator_->generateTrajectory(cones_for_planning_local, planning_state_);

        // 1랩 완료 (current_lap_이 2가 되었을 때) 시 글로벌 경로 생성 및 RACING 모드 전환
        if (current_lap_ > 1 && !is_global_path_generated_) {
            ROS_INFO("Lap 1 finished. Refining cone map and generating Global Path...");
            ROS_INFO("Starting map refinement...");
            map_manager_->refineConeMap(); // 1랩 완료 시 한 번만 맵 정제
            ROS_INFO("Map refinement completed.");
            ROS_INFO("Starting lane generation from memory...");
            map_manager_->generateLanesFromMemory(); // 정제된 맵으로 차선 다시 생성 (글로벌 경로 생성의 기반)
            ROS_INFO("Lane generation from memory completed.");
            ROS_INFO("Calling generateGlobalPath()...");     
            generateGlobalPath();           // 글로벌 경로 생성
            ROS_INFO("generateGlobalPath() completed."); 
            is_global_path_generated_ = true;
            current_mode_ = DrivingMode::RACING; // RACING 모드로 전환
            ROS_INFO("Switched to RACING mode!");
        }
    } 
    
    // 4. RACING 모드 경로 추종 로직
    else if (current_mode_ == DrivingMode::RACING) { // else if를 사용하여 MAPPING과 RACING 모드 중 하나만 실행되도록
        if (!global_path_.empty()) {
            // 차량의 현재 위치에서 글로벌 경로의 일부를 잘라내어 로컬 궤적으로 사용
            // --- 이 부분이 핵심: global_path_를 기반으로 trajectory_points_를 채움 ---
            // 가장 가까운 global_path_ 포인트 찾기
            double min_dist_sq = std::numeric_limits<double>::max();
            int closest_idx = 0;
            Eigen::Vector2d current_vehicle_pos(vehicle_state.position.x(), vehicle_state.position.y());

            for (size_t i = 0; i < global_path_.size(); ++i) {
                double dist_sq = (global_path_[i].position - current_vehicle_pos).squaredNorm();
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_idx = i;
                }
            }

            // Lookahead Distance를 고려하여 글로벌 경로에서 일정 길이의 로컬 궤적 추출
            trajectory_points_.clear();
            double accumulated_s = 0.0;
            // global_path_가 순환 경로라고 가정하고, closest_idx부터 시작하여 lookahead_distance_ 만큼의 경로를 추출
            // 트랙이 닫힌 루프 형태임을 감안하여 순환적으로 접근 (modulo 연산 사용)
            for (size_t i = 0; i < global_path_.size(); ++i) {
                // closest_idx부터 시작하여 global_path_를 순환적으로 탐색
                size_t current_path_idx = (closest_idx + i) % global_path_.size();
                trajectory_points_.push_back(global_path_[current_path_idx]);

                // 첫 번째 포인트 이후부터 거리 누적
                if (i > 0) {
                    accumulated_s += (trajectory_points_[i].position - trajectory_points_[i-1].position).norm();
                }

                // 누적 거리가 lookahead_distance_를 넘으면 추출 중단
                if (accumulated_s >= planning_params_->lookahead_distance_) {
                    break;
                }
                // (선택적) trajectory_points_에 충분한 점이 없다면 global_path_를 더 길게 탐색하도록 할 수 있으나
                // 여기서는 lookahead_distance_ 내의 점들만 추출
            }
            
            if (trajectory_points_.empty()) {
                ROS_WARN("FormulaAutonomousSystem: Global path is too short or empty for RACING mode. Using stop trajectory.");
                trajectory_points_ = trajectory_generator_->generateStopTrajectory();
            }

            // 추출된 로컬 궤적은 이미 글로벌 경로에서 계산된 속도 프로파일을 가지고 있음.
            // 필요하다면 여기서 추가적인 속도 조정 (예: 현재 차량 속도에 따른 가감속 여유) 가능.

        } else {
            ROS_WARN("RACING mode is active, but global path is not ready. Using stop trajectory as fallback.");
            trajectory_points_ = trajectory_generator_->generateStopTrajectory(); // 폴백: 정지 궤적
        }
    }
}

/**
 * @brief 1랩 주행이 완료되면 MapManager의 정보를 바탕으로 글로벌 경로를 생성합니다.
 * 생성된 경로는 `global_path_`에 저장됩니다.
 */
void FormulaAutonomousSystem::generateGlobalPath() {
    global_path_.clear();

    // MapManager에서 정제된 좌/우 차선 포인트를 가져옵니다.
    auto lanes = map_manager_->getTrackLanes();
    const auto& left_lane = lanes.first;
    const auto& right_lane = lanes.second;

    // 양쪽 차선이 충분한 포인트 수를 가지고 있는지 확인합니다.
    if (left_lane.empty() || right_lane.empty() || left_lane.size() < 2 || right_lane.size() < 2) {
        ROS_WARN("FormulaAutonomousSystem: Cannot generate global path. Not enough lane points from MapManager.");
        return;
    }

    // 1. 좌/우 차선을 기반으로 트랙의 중심선을 생성합니다.
    std::vector<Eigen::Vector2d> center_points;
    // ✨ 핵심 개선: 좌우 차선점 재매칭 로직
    // 짧은 차선 기준으로 긴 차선의 가장 가까운 점을 찾아서 매칭
    const auto* shorter_lane = &left_lane;
    const auto* longer_lane = &right_lane;
    if (right_lane.size() < left_lane.size()) {
        shorter_lane = &right_lane;
        longer_lane = &left_lane;
    }

    for (const auto& shorter_point : *shorter_lane) {
        double min_dist_sq = std::numeric_limits<double>::max();
        Eigen::Vector2d matched_longer_point;
        bool found_match = false;

        for (const auto& longer_point : *longer_lane) {
            double dist_sq = (shorter_point - longer_point).squaredNorm();
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                matched_longer_point = longer_point;
                found_match = true;
            }
        }
        
        // 매칭되는 점이 있으면 중심점 생성
        if (found_match) {
            Eigen::Vector2d mid_point = (shorter_point + matched_longer_point) * 0.5;
            center_points.push_back(mid_point);
        }
    }
    
    // 이 로그를 추가하여 중심선 포인트의 개수를 확인합니다.
    ROS_INFO("Generated center points: %zu", center_points.size());


    // 2. 생성된 중심선을 파라메트릭 스플라인으로 부드럽게 만듭니다.
    // (TrajectoryGenerator::generateTrajectory와 유사한 스플라인 로직)
    if (center_points.size() >= 2) { // 2점 이상이어야 스플라인이 의미 있음
        std::vector<double> s_pts, x_pts, y_pts;
        s_pts.push_back(0.0);
        x_pts.push_back(center_points[0].x());
        y_pts.push_back(center_points[0].y());

        for (size_t i = 1; i < center_points.size(); ++i) {
            double dist = (center_points[i] - center_points[i - 1]).norm();
            double new_s = s_pts.back() + dist;
            // s 값은 엄격하게 증가해야 스플라인이 안정적임. 같은 위치의 점 방지
            if (new_s <= s_pts.back()) {
                new_s = s_pts.back() + 0.01; 
            }
            s_pts.push_back(new_s);
            x_pts.push_back(center_points[i].x());
            y_pts.push_back(center_points[i].y());
        }
        
        // 스플라인 생성 시 최소 3개의 포인트가 필요함을 고려
        if (s_pts.size() < 3) {
            ROS_WARN("FormulaAutonomousSystem: Not enough points for spline generation. Global path will be linear.");
            // 2점 이하일 경우 스플라인 대신 선형 보간 로직 추가 가능 (선택적)
            for (size_t i = 0; i < center_points.size(); ++i) {
                global_path_.emplace_back(center_points[i].x(), center_points[i].y(), 
                                           0.0, 0.0, planning_params_->default_speed_, s_pts[i]);
            }
            return;
        }

        tk::spline spline_x, spline_y;
        spline_x.set_points(s_pts, x_pts);
        spline_y.set_points(s_pts, y_pts);

        // 3. 스플라인에서 최종 글로벌 경로(`global_path_`)를 샘플링하고 속도 프로파일을 할당합니다.
        double total_length = s_pts.back();
        double sampling_interval = 0.5; // 0.5m 간격으로 샘플링

        for (double s = 0; s < total_length; s += sampling_interval) {
            double x = spline_x(s);
            double y = spline_y(s);

            double dx = spline_x.deriv(1, s);
            double dy = spline_y.deriv(1, s);
            double yaw = std::atan2(dy, dx);

            double ddx = spline_x.deriv(2, s);
            double ddy = spline_y.deriv(2, s);
            // 곡률 공식: |dx * ddy - dy * ddx| / (dx^2 + dy^2)^(3/2)
            double curvature_denominator = std::pow(dx * dx + dy * dy, 1.5);
            double curvature = (curvature_denominator < 1e-6) ? 0.0 : std::abs(dx * ddy - dy * ddx) / curvature_denominator;

            // 곡률 기반 속도 계획 (planning_params_ 사용)
            double desired_speed = planning_params_->max_speed_ / (1.0 + planning_params_->curvature_gain_ * std::abs(curvature));
            double target_speed = std::max(planning_params_->min_speed_, std::min(desired_speed, planning_params_->max_speed_));

            global_path_.emplace_back(x, y, yaw, curvature, target_speed, s);
        }
        // 마지막 포인트 추가 (샘플링 간격으로 인해 끝점이 잘릴 수 있으므로, 마지막 중심점 추가)
        if (!center_points.empty() && (global_path_.empty() || (global_path_.back().position - center_points.back()).norm() > sampling_interval * 0.5 )) {
             // 마지막 중심점의 yaw와 curvature를 대략적으로 계산하여 추가
             double last_yaw = 0.0, last_curvature = 0.0;
             if (total_length > 0) { // total_length가 0이 아닐 때만 미분값 계산
                 double final_dx = spline_x.deriv(1, total_length);
                 double final_dy = spline_y.deriv(1, total_length);
                 last_yaw = std::atan2(final_dy, final_dx);

                 double final_ddx = spline_x.deriv(2, total_length);
                 double final_ddy = spline_y.deriv(2, total_length);
                 double final_curvature_denominator = std::pow(final_dx * final_dx + final_dy * final_dy, 1.5);
                 last_curvature = (final_curvature_denominator < 1e-6) ? 0.0 : std::abs(final_dx * final_ddy - final_ddy * final_ddx) / final_curvature_denominator;
             }
             double final_speed = planning_params_->max_speed_ / (1.0 + planning_params_->curvature_gain_ * std::abs(last_curvature));
             final_speed = std::max(planning_params_->min_speed_, std::min(final_speed, planning_params_->max_speed_));

             global_path_.emplace_back(center_points.back().x(), center_points.back().y(), 
                                       last_yaw, last_curvature, final_speed, total_length);
        }

    } else {
        ROS_WARN("FormulaAutonomousSystem: Not enough center points to generate smooth global path.");
    }

    if (!global_path_.empty()) {
        ROS_INFO("FormulaAutonomousSystem: Global Path generated successfully with %zu points.", global_path_.size());
    } else {
        ROS_ERROR("FormulaAutonomousSystem: Failed to generate Global Path.");
    }
}

/**
 * @brief 시작/결승선을 정의합니다. 첫 랩 동안 맵 매니저가 축적한 콘들을 기반으로 합니다.
 * FSDS 규정에 따라 Start/Finish Line은 보통 주황색 콘으로 정의됩니다.
 * 가장 가까운 두 개의 주황색 콘을 찾아 시작/결승선으로 정의합니다.
 * @param global_cones 전역 맵에 있는 모든 콘
 */
void FormulaAutonomousSystem::defineStartFinishLine(const std::vector<Cone>& global_cones) {
    if (global_cones.empty()) return;

    std::vector<Eigen::Vector2d> orange_cones;
    for (const auto& cone : global_cones) {
        if (cone.color == "orange") {
            orange_cones.emplace_back(cone.center.x, cone.center.y);
        }
    }

    // 최소 2개의 주황색 콘이 있어야 시작/결승선 정의 가능
    if (orange_cones.size() >= 2) {
        // --- 아이디어: 트랙 폭(planning_params_->min_y_separation)을 고려하여
        //              x좌표가 어느 정도 비슷하면서 y좌표가 크게 차이나는 두 콘을 찾는 것이 더 정확
        //              여기서는 가장 가까운 두 콘을 찾는 것으로 단순화되어 있음 ---

        // 두 콘을 선택하는 기준: x좌표가 비슷하고 y좌표 차이가 큰 (트랙 폭에 가까운) 두 콘
        // 또는 특정 ROI 내의 주황색 콘 쌍을 찾는 로직
        Eigen::Vector2d p1_sf = orange_cones[0];
        Eigen::Vector2d p2_sf = orange_cones[1];
        double max_y_diff = -1.0; // y축 차이가 가장 큰 두 콘을 찾기 위함

        for (size_t i = 0; i < orange_cones.size(); ++i) {
            for (size_t j = i + 1; j < orange_cones.size(); ++j) {
                // x좌표가 크게 다르지 않으면서 y좌표 차이가 큰 두 콘을 찾음
                // planning_params_->min_y_separation_은 트랙 폭으로 사용될 수 있음
                // planning_params_->max_x_separation_은 시작/결승선 콘의 x축 오차 허용 범위
                if (std::abs(orange_cones[i].x() - orange_cones[j].x()) < planning_params_->max_x_separation_ &&
                    std::abs(orange_cones[i].y() - orange_cones[j].y()) > max_y_diff) {
                    
                    max_y_diff = std::abs(orange_cones[i].y() - orange_cones[j].y());
                    p1_sf = orange_cones[i];
                    p2_sf = orange_cones[j];
                }
            }
        }
        
        // 두 콘의 중간 지점을 시작/결승선 중심으로 설정
        start_finish_line_center_ = (p1_sf + p2_sf) * 0.5;
        // 두 콘을 잇는 선의 수직 방향을 시작/결승선 yaw로 설정
        // 트랙이 x축 방향으로 진행한다고 가정하고, y축 방향 벡터의 역방향이 라인 노멀이 됨
        start_finish_line_yaw_ = std::atan2(p2_sf.x() - p1_sf.x(), -(p2_sf.y() - p1_sf.y())); // p1->p2 벡터의 수직 방향
        
        is_start_finish_line_defined_ = true;
        ROS_INFO("FormulaAutonomousSystem: Start/Finish Line defined at (%.2f, %.2f) with yaw %.2f rad.", 
                 start_finish_line_center_.x(), start_finish_line_center_.y(), start_finish_line_yaw_);
    } else {
        ROS_WARN("FormulaAutonomousSystem: Not enough orange cones to define Start/Finish Line. Need at least 2.");
        is_start_finish_line_defined_ = false;
    }
}

/**
 * @brief 차량이 시작/결승선을 통과했는지 확인하고 랩 카운트를 업데이트합니다.
 * @param current_state 현재 차량의 상태 (전역 좌표계)
 */
// formula_autonomous_system.cpp
// formula_autonomous_system.cpp
void FormulaAutonomousSystem::updateLapCount(const VehicleState& current_state) {
    if (!is_start_finish_line_defined_) {
        ROS_INFO_THROTTLE(5.0, "Start/Finish line is not yet defined. Lap counting is on hold.");
        return;
    }

    // [수정된 부분] previous_position_가 (0,0)이면 첫 실행으로 간주하고 현재 위치로 초기화
    // .squaredNorm()은 벡터의 길이 제곱을 계산하며, (0,0)일 때만 0이 됩니다.
    if (previous_position_.squaredNorm() == 0.0) {
        previous_position_ = current_state.position;
        last_lap_crossing_time_ = std::chrono::steady_clock::now(); // 타이머도 함께 초기화
        ROS_INFO("Lap counter initialized. previous_position set to (%.2f, %.2f)", previous_position_.x(), previous_position_.y());
        return;
    }

    // 통과 간 최소 시간 설정 (예: 5초, 동일 랩에서 중복 카운팅 방지)
    const double min_crossing_interval_sec = 5.0;
    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_since_last_crossing = current_time - last_lap_crossing_time_;

    // 결승선 통과 감지 거리 설정 (결승선 중심으로부터의 최대 거리)
    const double line_crossing_threshold_dist = planning_params_->max_x_separation_ * 2.0;

    // 1. 결승선 라인에 수직인 벡터(노멀 벡터) 계산
    double line_normal_x = std::cos(start_finish_line_yaw_);
    double line_normal_y = std::sin(start_finish_line_yaw_);
    Eigen::Vector2d line_normal(line_normal_x, line_normal_y);

    // 2. 현재 위치와 이전 위치의 결승선 라인에 대한 상대적 위치 계산
    Eigen::Vector2d current_vec = current_state.position - start_finish_line_center_;
    Eigen::Vector2d prev_vec = previous_position_ - start_finish_line_center_;

    // 내적(dot product)을 통해 차량이 라인 앞/뒤 중 어디에 있는지 판별 (+는 앞, -는 뒤)
    double current_dot_product = current_vec.dot(line_normal);
    double prev_dot_product = prev_vec.dot(line_normal);

    // 차량의 주행 방향이 결승선 통과 방향과 일치하는지 확인
    Eigen::Vector2d velocity_vector(std::cos(current_state.yaw), std::sin(current_state.yaw));
    double velocity_dot_product = velocity_vector.dot(line_normal);

    // [디버깅 로그 추가]
    ROS_INFO_THROTTLE(1.0, "[Lap Check] Dist: %.2f | PrevDot: %.2f | CurrDot: %.2f | TimeSince: %.2f | VelDot: %.2f",
        current_vec.norm(), prev_dot_product, current_dot_product, time_since_last_crossing.count(), velocity_dot_product);

    // 3. 랩 통과 조건 검사
    if (current_vec.norm() < line_crossing_threshold_dist) { // 조건 1: 결승선 근처에 있는가?
        if (current_dot_product > 0 && prev_dot_product <= 0) { // 조건 2: 라인을 뒤에서 앞으로 통과했는가?
            if (time_since_last_crossing.count() > min_crossing_interval_sec) { // 조건 3: 마지막 통과 후 충분한 시간이 지났는가?
                if (velocity_dot_product > 0.1) { // 조건 4: 정방향으로 주행 중인가?
                    current_lap_++;
                    last_lap_crossing_time_ = current_time;
                    ROS_INFO("==============================================");
                    ROS_INFO("LAP %d PASSED!", current_lap_ -1);
                    ROS_INFO("LAP %d STARTED!", current_lap_);
                    ROS_INFO("==============================================");
                }
            }
        }
    }

    // 다음 프레임 계산을 위해 현재 위치를 이전 위치로 업데이트
    previous_position_ = current_state.position;
}
/*void FormulaAutonomousSystem::setRacingStrategy(const VehicleState& vehicle_state, const std::vector<Cone>& cones_for_planning) {
    // 1. Lap Counting & Mode Switching Logic
    if (!is_start_finish_line_defined_) {
        defineStartFinishLine(cones_for_planning);
    }
    if (is_start_finish_line_defined_) {
        updateLapCount(vehicle_state); // current_lap_ is updated inside
    }
    // 2. Trajectory Planning based on Driving Mode
    if (current_mode_ == DrivingMode::MAPPING) {
        // In mapping mode, we do not generate a global path
        trajectory_points_ = trajectory_generator_->generateTrajectory(cones_for_planning, planning_state_);

        // Switch to RACING mode after completing the first lap
        if (current_lap_ > 1 && !is_global_path_generated_) {
            ROS_INFO("Lap 1 finished. Generating Global Path and switching to RACING mode...");
            generateGlobalPath();
            is_global_path_generated_ = true;
            current_mode_ = DrivingMode::RACING;
            ROS_INFO("Switched to RACING mode!");
        }

    } else {
        // In racing mode, follow the global path
        if (!global_path_.empty()) {
            // TODO: (다음 단계에서 구현) 전역 경로에서 현재 주행에 필요한 부분을 추출
            // trajectory_points_ = trajectory_generator_->getTrajectoryFromGlobalPath(vehicle_state, global_path_);
            trajectory_points_ = trajectory_generator_->generateTrajectory(cones_for_planning, planning_state_); // 임시
        } else {
            ROS_WARN("RACING mode is active, but global path is not ready. Using local planner as fallback.");
            trajectory_points_ = trajectory_generator_->generateTrajectory(cones_for_planning, planning_state_);
        }
    }
}
*/
