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

std::vector<Eigen::Vector2d> MapManager::sortConesByProximity(const std::vector<Eigen::Vector2d>& cones, const Eigen::Vector2d& start_pos) {
    if (cones.size() < 2) {
        return cones; // No sorting needed
    }

    std::vector<Eigen::Vector2d> sorted_cones;
    std::vector<Eigen::Vector2d> remaining_cones = cones;

    const double max_connection_distance = params_->max_connection_distance_;
    const double direction_weight = params_->direction_weight_;

    // Start sorting from the cone closest to the vehicle's origin (0,0),
    // which represents the "first" cone encountered.
    auto start_it = std::min_element(remaining_cones.begin(), remaining_cones.end(),
        [&start_pos](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
            // squaredNorm() calculates the squared distance from the start_pos
            return (a - start_pos).squaredNorm() < (b - start_pos).squaredNorm();
        });

    sorted_cones.push_back(*start_it);
    remaining_cones.erase(start_it);

    // Initial direction is from the origin to the first cone.
    Eigen::Vector2d current_direction = sorted_cones.front().normalized();
    
    // Repeat until all cones are sorted
    while (!remaining_cones.empty()) {

        Eigen::Vector2d current_point = sorted_cones.back();
        
        // Update direction vector based on the last two sorted cones
        if (sorted_cones.size() > 1) {
            current_direction = (sorted_cones.back() - sorted_cones[sorted_cones.size() - 2]).normalized();
        }

        double best_score = std::numeric_limits<double>::max();
        auto best_it = remaining_cones.begin();

        for (auto it = remaining_cones.begin(); it != remaining_cones.end(); ++it) {
            double dist = (*it - current_point).norm();

            // Skip cones that are too far away
            if (dist > max_connection_distance) {
                continue;
            }

            // Cost function based on distance and direction consistency
            Eigen::Vector2d candidate_direction = (*it - current_point).normalized();
            double dot_product = current_direction.dot(candidate_direction);

            // Lower score is better. Penalize directions that deviate from the current path.
            double score = dist * (1.0 + direction_weight * (1.0 - dot_product));

            if (score < best_score) {
                best_score = score;
                best_it = it;
            }
        }

        // If a large gap is detected (no cone found within max_connection_distance)
        if (best_score == std::numeric_limits<double>::max()) {
            break;

            if (remaining_cones.empty()) {
                break; // No more cones left to process.
            }

            ROS_WARN("MapManager: Large gap detected. Jumping to the next spatially closest cone.");

            // Find the absolute closest cone to bridge the gap
            auto closest_after_gap_it = remaining_cones.begin();
            double min_dist_after_gap = std::numeric_limits<double>::max();

            for (auto it = remaining_cones.begin(); it != remaining_cones.end(); ++it) {
                double dist = (*it - current_point).norm();

                if (dist < min_dist_after_gap) {
                    min_dist_after_gap = dist;
                    closest_after_gap_it = it;
                }
            }

            best_it = closest_after_gap_it; // Force the connection
        }

        if (best_it == remaining_cones.end()) { // Safety check
             break;
        }

        sorted_cones.push_back(*best_it);
        remaining_cones.erase(best_it);
    }

    return sorted_cones;
}

void MapManager::generateLanesFromMemory(const Eigen::Vector2d& vehicle_pos) {

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
    auto sorted_blue = sortConesByProximity(blue_cones, vehicle_pos);
    auto sorted_yellow = sortConesByProximity(yellow_cones, vehicle_pos);

    // To create a closed loop, check if the last and first cones are close enough to connect.
    if (sorted_blue.size() > 2) {

        double dist = (sorted_blue.back() - sorted_blue.front()).norm();

        if (dist < params_->max_connection_distance_) {
            sorted_blue.push_back(sorted_blue.front()); // Add the first point to the end
            ROS_INFO_ONCE("MapManager: Blue lane is now a closed loop.");
        }
    }

    if (sorted_yellow.size() > 2) {

        double dist = (sorted_yellow.back() - sorted_yellow.front()).norm();

        if (dist < params_->max_connection_distance_) {
            sorted_yellow.push_back(sorted_yellow.front()); // Add the first point to the end
            ROS_INFO_ONCE("MapManager: Yellow lane is now a closed loop.");
        }
    }

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

void MapManager::refineConeMap(const Eigen::Vector2d& vehicle_pos) {
    std::lock_guard<std::mutex> lock(cone_memory_mutex_);

    // 1. Generate temporary lanes from current map (guideline)
    generateLanesFromMemory(vehicle_pos);

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
    valid_transitions_[{ASState::AS_DRIVING, ASState::AS_FINISHED}] = true;

    // AS_FINISHED transitions
    valid_transitions_[{ASState::AS_FINISHED, ASState::AS_OFF}] = true;
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

        // Finishing event
        case ASEvent::RACE_FINISHED:
            if (current_state_ == ASState::AS_DRIVING) {
                target_state = ASState::AS_FINISHED;
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

bool StateMachine::enterAS_FINISHED() {
    std::cout << "StateMachine: Entering AS_FINISHED state" << std::endl;
    mission_active_ = false; // Race finished, deactivate mission
    return true;
}

// State exit functions
bool StateMachine::exitAS_OFF() { return true; }
bool StateMachine::exitAS_READY() { return true; }
bool StateMachine::exitAS_DRIVING() { return true; }
bool StateMachine::exitAS_FINISHED() { return true; }

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
        case ASState::AS_FINISHED: return "AS_FINISHED";
        default: return "UNKNOWN";
    }
}

std::string StateMachine::eventToString(ASEvent event) const {
    switch (event) {
        case ASEvent::SYSTEM_INIT: return "SYSTEM_INIT";
        case ASEvent::SYSTEM_READY: return "SYSTEM_READY";
        case ASEvent::GO_SIGNAL: return "GO_SIGNAL";
        case ASEvent::RACE_FINISHED: return "RACE_FINISHED";
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

    std::cout << "TrajectoryGenerator: Initialized with default mapping lookahead " 
              << params_->trajectory_generation.mapping_mode.lookahead_distance_ << "m, spacing " 
              << params_->trajectory_generation.mapping_mode.waypoint_spacing_ << "m" << std::endl;
}

std::vector<TrajectoryPoint> TrajectoryGenerator::generatePathFromClosestCones(const std::vector<Cone>& cones, const PlanningParams::TrajectoryModeParams& params)
{
    last_trajectory_.clear();

    // 1. Filter blue and yellow cones
    std::vector<Eigen::Vector2d> blue_cones_local, yellow_cones_local;
    for (const auto& cone : cones) {
        if (cone.center.x > 0.1) { // Use only cones in front of the vehicle
            if (cone.color == "blue") {
                blue_cones_local.push_back(Eigen::Vector2d(cone.center.x, cone.center.y));
            } else if (cone.color == "yellow") {
                yellow_cones_local.push_back(Eigen::Vector2d(cone.center.x, cone.center.y));
            }
        }
    }

    // 2. Find the closest blue and yellow cone from the vehicle's origin (0,0)
    auto find_closest_cone = [](const std::vector<Eigen::Vector2d>& cone_list) {
        if (cone_list.empty()) return Eigen::Vector2d(0.0, 0.0);
        return *std::min_element(cone_list.begin(), cone_list.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.squaredNorm() < b.squaredNorm();
            });
    };

    Eigen::Vector2d closest_blue = find_closest_cone(blue_cones_local);
    Eigen::Vector2d closest_yellow = find_closest_cone(yellow_cones_local);

    // 3. Calculate the target y-coordinate (target_y)
    double target_y = 0.0;
    bool target_found = false;

    if (!closest_blue.isZero() && !closest_yellow.isZero()) {
        // If both cones are found, set the target to their midpoint
        target_y = (closest_blue.y() + closest_yellow.y()) * 0.5;
        target_found = true;
    } else if (!closest_blue.isZero()) {
        // If only a blue cone is found, offset to the right by a constant distance
        target_y = closest_blue.y() - params.lane_offset_;
        target_found = true;
    } else if (!closest_yellow.isZero()) {
        // If only a yellow cone is found, offset to the left by a constant distance
        target_y = closest_yellow.y() + params.lane_offset_;
        target_found = true;
    }

    // 4. Generate a simple straight path towards the target point
    int num_points = static_cast<int>(params.lookahead_distance_ / params.waypoint_spacing_);
    for (int i = 0; i <= num_points; ++i) {
        double x = i * params.waypoint_spacing_;
        // If a cone is found, aim for the target_y; otherwise, go straight (y=0)
        double y = target_found ? (x / params.lookahead_distance_) * target_y : 0.0;
        
        last_trajectory_.emplace_back(x, y, 0.0, 0.0, params.max_speed_, x, 0.0);
    }
    
    // Calculate the yaw for the path (simply the angle between two points)
    if (last_trajectory_.size() >= 2) {
        for(size_t i = 0; i < last_trajectory_.size() - 1; ++i) {
            double dx = last_trajectory_[i+1].position.x() - last_trajectory_[i].position.x();
            double dy = last_trajectory_[i+1].position.y() - last_trajectory_[i].position.y();
            last_trajectory_[i].yaw = std::atan2(dy, dx);
        }
        last_trajectory_.back().yaw = last_trajectory_[last_trajectory_.size() - 2].yaw;
    }

    return last_trajectory_;
}

/**
 * @brief Generates a local trajectory for the controller by extracting and transforming a segment from the global path.
 * @param vehicle_state The current state of the vehicle in the global frame.
 * @param global_path The pre-computed global path.
 * @return A local trajectory in the vehicle's coordinate frame.
 */

std::vector<TrajectoryPoint> TrajectoryGenerator::getTrajectoryFromGlobalPath(const VehicleState& vehicle_state, const std::vector<TrajectoryPoint>& global_path, const PlanningParams::TrajectoryModeParams& params)
{   
    last_trajectory_.clear();
    last_local_path_points_.clear();
    if (global_path.size() < 2) return last_trajectory_;

    double vehicle_yaw = vehicle_state.yaw;
    Eigen::Vector2d vehicle_pos = vehicle_state.position;

    // 1. Find the closest point on the global path to the vehicle.
    auto closest_it = std::min_element(global_path.begin(), global_path.end(),
        [&](const TrajectoryPoint& a, const TrajectoryPoint& b) {
            return (a.position - vehicle_pos).squaredNorm() < (b.position - vehicle_pos).squaredNorm();
        });
    size_t start_idx = std::distance(global_path.begin(), closest_it);

    // 2. Extract a segment of the path from the starting point.
    double traversed_s = 0.0;
    for (size_t i = 0; i < global_path.size(); ++i) {
        size_t current_idx = (start_idx + i) % global_path.size();
        const auto& global_point = global_path[current_idx];

        if (i > 0) {
            size_t prev_idx = (start_idx + i - 1) % global_path.size();
            traversed_s += (global_point.position - global_path[prev_idx].position).norm();
        }

        if (traversed_s > params.lookahead_distance_) {
            break;
        }

        // 3. Transform the global point to the vehicle's local coordinate frame.
        Eigen::Vector2d relative_pos = global_point.position - vehicle_pos;
        double x_local = relative_pos.x() * cos(-vehicle_yaw) - relative_pos.y() * sin(-vehicle_yaw);
        double y_local = relative_pos.x() * sin(-vehicle_yaw) + relative_pos.y() * cos(-vehicle_yaw);
        
        // Normalize angle to be within [-PI, PI]
        double yaw_local = global_point.yaw - vehicle_yaw;
        while (yaw_local > M_PI) yaw_local -= 2.0 * M_PI;
        while (yaw_local < -M_PI) yaw_local += 2.0 * M_PI;

        // Add the transformed point with pre-calculated data to the local trajectory
        last_trajectory_.emplace_back(x_local, y_local, yaw_local, global_point.curvature, global_point.speed, global_point.s, global_point.complexity_score);
        last_local_path_points_.emplace_back(x_local, y_local); // For visualization
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
    // Use mapping_mode parameters for the stop trajectory as a default
    const auto& params = params_->trajectory_generation.mapping_mode;
    int num_points = static_cast<int>(params.lookahead_distance_ / params.waypoint_spacing_);
    for (int i = 0; i < num_points; ++i) {
        double x = i * params.waypoint_spacing_;
        last_trajectory_.emplace_back(x, 0.0, 0.0, 0.0, 0.0, x, 0.0);
    }
    return last_trajectory_;
}

void FormulaAutonomousSystem::generateGlobalPath() {
    global_path_.clear();

    // vehicle_state_는 run() 함수에서 지속적으로 업데이트됩니다.
    Eigen::Vector2d current_vehicle_pos(vehicle_state_[0], vehicle_state_[1]);
    // getTrackLanes() 전에 generateLanesFromMemory를 먼저 호출해야 합니다.
    // getTrackLanes()는 이미 생성된 lane point를 반환하기만 합니다.

    map_manager_->generateLanesFromMemory(current_vehicle_pos);

    auto track_lanes = map_manager_->getTrackLanes();
    const auto& left_lane = track_lanes.first;
    const auto& right_lane = track_lanes.second;

    if (left_lane.size() < 2 || right_lane.size() < 2) {
        ROS_ERROR("MapManager: Not enough lane points to generate a global path.");
        return;
    }

    // 1. Calculate the center points of the track.
    std::vector<Eigen::Vector2d> center_points;
    for (const auto& left_point : left_lane) {
        auto closest_right_it = std::min_element(right_lane.begin(), right_lane.end(),
            [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return (a - left_point).squaredNorm() < (b - left_point).squaredNorm();
            });
        Eigen::Vector2d center_point = (*closest_right_it + left_point) / 2.0;
        if (center_points.empty() || (center_points.back() - center_point).norm() > 0.1) {
            center_points.push_back(center_point);
        }
    }
    if (center_points.size() < 3) {
        ROS_ERROR("MapManager: Failed to generate sufficient center points for the global path.");
        return;
    }

    // 2. Connect the center points with a smooth spline curve.
    std::vector<double> s_pts, x_pts, y_pts;
    s_pts.push_back(0.0);
    x_pts.push_back(center_points[0].x());
    y_pts.push_back(center_points[0].y());
    for (size_t i = 1; i < center_points.size(); ++i) {
        double dist = (center_points[i] - center_points[i - 1]).norm();
        s_pts.push_back(s_pts.back() + dist);
        x_pts.push_back(center_points[i].x());
        y_pts.push_back(center_points[i].y());
    }

    tk::spline spline_x, spline_y;
    spline_x.set_points(s_pts, x_pts);
    spline_y.set_points(s_pts, y_pts);
    // 3. [개선된 로직] Sample points and calculate speed with dynamic S-curve detection
    double total_length = s_pts.back();

    const auto& params = planning_params_->trajectory_generation.racing_mode; // Use RACING parameters
    std::vector<TrajectoryPoint> temp_path;

    for (double s = 0; s < total_length; s += 0.5) { // Sample every 0.5m
        double x = spline_x(s);
        double y = spline_y(s);
        double dx = spline_x.deriv(1, s);
        double dy = spline_y.deriv(1, s);
        double ddx = spline_x.deriv(2, s);
        double ddy = spline_y.deriv(2, s);
        double yaw = std::atan2(dy, dx);

        // S-curve detection(retain curvature sign)
        double curvature = (dx * ddy - dy * ddx) / std::pow(dx * dx + dy * dy, 1.5);
        temp_path.emplace_back(x, y, yaw, curvature, 0.0, s); // 속도는 나중에 계산
    }

    if (temp_path.size() < 5) {
        ROS_WARN("Not enough points in temp_path to generate a reliable global path.");
        return;
    }

    // Speed profiling based on complexity
    // 1. Calculate 'Complexity Score'
    std::vector<double> raw_complexity_scores(temp_path.size(), 0.0);
    if (params.complexity_enable_) {
        for (size_t i = 0; i < temp_path.size(); ++i) {
            size_t target_idx = i;
            double target_s = temp_path[i].s + params.complexity_check_distance_;
            for (size_t j = i; j < temp_path.size(); ++j) {
                if (temp_path[j].s >= target_s) { target_idx = j; break; }
                if (j == temp_path.size() - 1) target_idx = j;
            }
            if (target_idx <= i) continue;

            double total_curvature_change = 0.0;
            for (size_t j = i + 1; j <= target_idx; ++j) {
                total_curvature_change += std::abs(temp_path[j].curvature - temp_path[j-1].curvature);
            }
            double score = total_curvature_change / params.complexity_vibration_max_;
            raw_complexity_scores[i] = std::max(0.0, std::min(1.0, score));
        }
    }

    // 2. Smoothing score(Moving Average)
    std::vector<double> smoothed_complexity_scores = raw_complexity_scores;
    if (params.complexity_enable_ && params.complexity_smoothing_window_ > 1) {
        int half_window = params.complexity_smoothing_window_ / 2;
        if (temp_path.size() > params.complexity_smoothing_window_) {
            for (size_t i = half_window; i < temp_path.size() - half_window; ++i) {
                double sum = 0;
                for (int j = -half_window; j <= half_window; ++j) {
                    sum += raw_complexity_scores[i + j];
                }
                smoothed_complexity_scores[i] = sum / params.complexity_smoothing_window_;
            }
        }
    }

    // 3. Apply Relaxation Filter
    std::vector<double> final_scores = smoothed_complexity_scores;
    if (params.complexity_enable_) {
        for (size_t i = 1; i < final_scores.size(); ++i) {
            double max_decrease = params.complexity_score_decay_rate_;
            final_scores[i] = std::max(final_scores[i], final_scores[i-1] - max_decrease);
        }
    }

    // 4. Speed Blending
    for (size_t i = 0; i < temp_path.size(); ++i) {
        double high_speed = params.max_speed_ / (1.0 + params.curvature_gain_ * std::abs(temp_path[i].curvature));
        double low_speed = params.complexity_low_speed_;
        
        double complexity_score = final_scores[i];

        temp_path[i].complexity_score = complexity_score;
        
        temp_path[i].speed = high_speed * (1.0 - complexity_score) + low_speed * complexity_score;
        
        temp_path[i].speed = std::max(params.min_speed_, std::min(temp_path[i].speed, params.max_speed_));
    }

    // 5. Final Moving Average Filter
    std::vector<TrajectoryPoint> smoothed_path = temp_path;
    int final_smoothing_window = 5;
    if (temp_path.size() > final_smoothing_window) {
        for (size_t i = final_smoothing_window / 2; i < temp_path.size() - final_smoothing_window / 2; ++i) {
            double sum_speed = 0;
            for (int j = -final_smoothing_window / 2; j <= final_smoothing_window / 2; ++j) {
                sum_speed += temp_path[i + j].speed;
            }
            smoothed_path[i].speed = sum_speed / final_smoothing_window;
        }
    }

    // 6. Update global_path
    global_path_ = smoothed_path;
    ROS_INFO("Complexity-based Global Path generated with %zu points.", global_path_.size());
}

// ================================================================================================
// ========================================== 5. Control ==========================================
// ================================================================================================

// ================== PurePursuit Controller Implementation ===================
PurePursuit::PurePursuit() {}

double PurePursuit::calculateSteeringAngle(const VehicleState& current_state,
                                          const std::vector<TrajectoryPoint>& path,
                                          const ControlParams::ControllerModeParams& params,
                                          const double& vehicle_length) const {

    if (path.empty()) {
        return 0.0;
    }

    int target_idx = findTargetPointIndex(path, params);
    const Eigen::Vector2d& target_point = path[target_idx].position;

    return calculateSteeringAngleInternal(target_point, params, vehicle_length);
}

int PurePursuit::findTargetPointIndex(const std::vector<TrajectoryPoint>& path, const ControlParams::ControllerModeParams& params) const {

    int target_idx = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        double dist_from_car = path[i].position.norm();

        if (dist_from_car >= params.pp_lookahead_distance_) {
            target_idx = i;
            break;
        }
    }
    return target_idx;
}

double PurePursuit::calculateSteeringAngleInternal(const Eigen::Vector2d& target_point,
                                                   const ControlParams::ControllerModeParams& params,
                                                   const double& vehicle_length) const {

    double alpha = std::atan2(target_point.y(), target_point.x() + vehicle_length * 0.5); // local trajectory is defined at the center of the vehicle
    double delta = std::atan2(2.0 * vehicle_length * std::sin(alpha), params.pp_lookahead_distance_);

    return std::clamp(delta, -params.pp_max_steer_angle_, params.pp_max_steer_angle_);
}

// ================== Stanley Controller Implementation ===================
Stanley::Stanley() {}

double Stanley::calculateSteeringAngle(const VehicleState& current_state,
                                       const std::vector<TrajectoryPoint>& path,
                                       const ControlParams::ControllerModeParams& params,
                                       const double& vehicle_length) const
{
    if (path.empty()) return 0.0;

    // 1. Get the point of front axle(vehicle frame)
    const Eigen::Vector2d front_axle_pos(vehicle_length * 0.5, 0.0); // local trajectory is defined at the center of the vehicle
    
    // 2. Find the closest point on the path to the front axle
    double min_dist = std::numeric_limits<double>::max();
    int closest_segment_idx = 0;
    Eigen::Vector2d closest_proj_point;
    
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        const Eigen::Vector2d& p1 = path[i].position;
        const Eigen::Vector2d& p2 = path[i+1].position;
        const Eigen::Vector2d segment = p2 - p1;

        if (segment.squaredNorm() < 1e-8) continue; // skip if too short

        double t = (front_axle_pos - p1).dot(segment) / segment.squaredNorm();
        t = std::max(0.0, std::min(1.0, t)); // t (0.0 ~ 1.0)

        Eigen::Vector2d projection = p1 + t * segment;
        double dist_sq = (front_axle_pos - projection).squaredNorm();

        if (dist_sq < min_dist) {
            min_dist = dist_sq;
            closest_segment_idx = i;
            closest_proj_point = projection;
        }
    }

    // 3. Calculate the cross track error(with sign)
    double path_yaw = path[closest_segment_idx].yaw;
    Eigen::Vector2d error_vec = front_axle_pos - closest_proj_point;

    // Determine left or right direction error
    double cross_track_error = error_vec.y() * std::cos(path_yaw) - error_vec.x() * std::sin(path_yaw);

    // 4. Calculate the heading error
    double heading_error = path_yaw;

    // 5. Calculate dynamic k gain based on curvature
    const TrajectoryPoint& target_point = path[closest_segment_idx];
    double target_curvature = std::abs(target_point.curvature);

    // k_gain_curvature_boost_ has value only if RACING MODE
    double curvature_boost_factor = 1.0 + params.k_gain_curvature_boost_ * target_curvature;
    double dynamic_k = params.k_gain_ * curvature_boost_factor;

    // 6. Calculate the cross track steering, 0.1 is added to the speed to avoid division by zero
    double cross_track_steering = atan2(dynamic_k * -cross_track_error, current_state.speed + 0.1);

    // 7. Calculate the steering angle
    double steering_angle = heading_error + cross_track_steering;

    // 8. Low-Pass Filter
    const double alpha = params.stanley_alpha_; // stability(0.0) <---> response(1.0)
    double filtered_steering_angle = alpha * steering_angle + (1.0 - alpha) * last_filtered_steering_angle_;
    last_filtered_steering_angle_ = filtered_steering_angle;

    // Clamp the steering angle: -max_steer_angle_ <= steering_angle <= max_steer_angle_
    return std::clamp(filtered_steering_angle, -params.pp_max_steer_angle_, params.pp_max_steer_angle_);
}

// ==================== PID Controller Implementation ====================

// formula_autonomous_system.cpp

// ==================== PID Controller Implementation ====================

// 생성자에서 멤버 변수 초기화 부분을 제거하고, 상태 변수만 초기화
PIDController::PIDController()
    : integral_error_(0.0), previous_error_(0.0), first_run_(true) {}

// 함수 시그니처 변경 및 게인 값들을 인자로 직접 받아서 사용
double PIDController::calculate(double setpoint, double measured_value, double kp, double ki, double kd, double min_output, double max_output) {
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
    
    if (dt <= 1e-6) {
        return std::clamp(kp * (setpoint - measured_value), min_output, max_output);
    }

    // 1. 비례(Proportional) 항 계산
    double error = setpoint - measured_value;
    double p_term = kp * error;

    // 2. 적분(Integral) 항 계산
    integral_error_ += error * dt;
    double i_term = ki * integral_error_;

    // 3. 미분(Derivative) 항 계산
    double derivative_error = (error - previous_error_) / dt;
    double d_term = kd * derivative_error;

    // 최종 제어 출력값 계산
    double output = p_term + i_term + d_term;

    // 다음 계산을 위해 현재 상태 저장
    previous_error_ = error;
    last_time_ = current_time;

    // 출력값을 지정된 범위 내로 제한(clamping)
    return std::clamp(output, min_output, max_output);
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
    current_lap_(0),
    is_race_finished_(false),
    vehicle_position_relative_to_line_(0.0)
    {
        last_lap_time_ = ros::Time(0);
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
    vehicle_state_ = std::vector<double>(8, 0.0);

    // Mapping
    map_manager_ = std::make_unique<MapManager>(mapping_params_);

    // Planning
    state_machine_ = std::make_unique<StateMachine>();
    trajectory_generator_ = std::make_unique<TrajectoryGenerator>(planning_params_);

    // Control
    if (control_params_->lateral_controller_type_ == "Stanley") { 
        lateral_controller_ = std::make_unique<Stanley>();
        ROS_INFO("Lateral Controller: Stanley selected");

    } else { // Default to Pure Pursuit
        lateral_controller_ = std::make_unique<PurePursuit>();
        ROS_INFO("Lateral Controller: PurePursuit selected");
    }

    longitudinal_controller_ = std::make_unique<PIDController>();

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

    vehicle_state_ = localization_->getCurrentState();
    VehicleState vehicle_state(vehicle_state_[0], vehicle_state_[1], vehicle_state_[2], vehicle_state_[3]);

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

    if (planning_state_ != ASState::AS_DRIVING && planning_state_ != ASState::AS_FINISHED) {
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

    // Update global cone map
    std::vector<Cone> cones_for_planning = map_manager_->updateAndGetPlannedCones(vehicle_state, cones_);
    map_manager_->generateLanesFromMemory(vehicle_state.position);

    // Behavior planning
    setRacingStrategy(vehicle_state, cones_for_planning);

    // Trajectory generation (Only if it's AS_DRIVING state)
    if (planning_state_ == ASState::AS_DRIVING) {

        // Select parameters based on the current driving mode
        const auto& current_planning_params = (current_mode_ == DrivingMode::RACING) 
                                            ? planning_params_->trajectory_generation.racing_mode 
                                            : planning_params_->trajectory_generation.mapping_mode;

        // Generate trajectory based on the current driving mode
        if (current_mode_ == DrivingMode::RACING && is_global_path_generated_) {
            // In RACING mode, follow the pre-calculated global path
            trajectory_points_ = trajectory_generator_->getTrajectoryFromGlobalPath(vehicle_state, global_path_, current_planning_params);

        } else {
            // In MAPPING mode, generate a simple and stable path for mapping
            trajectory_points_ = trajectory_generator_->generatePathFromClosestCones(cones_for_planning, current_planning_params);
        }
    }


    // =================================================================
    // STEP 5: CONTROL - "How do I get there?"
    // =================================================================
    // Default control command(STOP)
    double final_steering = 0.0;
    double final_throttle = 0.0;
    double final_brake = 1.0;

    // Choose control method depending on the vehicle's state
    if (planning_state_ == ASState::AS_FINISHED) {
        double distance_past_line = std::abs(vehicle_position_relative_to_line_);

        // slow speed mode in finish stop distance
        if (distance_past_line < planning_params_->behavioral_logic.finish_stop_distance_) {
            trajectory_points_ = trajectory_generator_->generateStopTrajectory();
            for(auto& point : trajectory_points_) { point.speed = 1.0; }

            const auto& control_params = control_params_->mapping_mode;
            final_steering = lateral_controller_->calculateSteeringAngle(vehicle_state, trajectory_points_, control_params, control_params_->vehicle_length_);
            
            // Brake logic for FINISHED state
            double control_effort = longitudinal_controller_->calculate(1.0, vehicle_state.speed, control_params.pid_kp_, control_params.pid_ki_, control_params.pid_kd_, -control_params.max_brake_, control_params.max_throttle_);
            if (control_effort > 0.0) {
                final_throttle = control_effort;
                final_brake = 0.0;

            } else {
                final_throttle = 0.0;
                final_brake = -control_effort; // control_effort is negative, so - makes it positive for brake command
            }

        // Absolute STOP
        } else {
            final_steering = 0.0;
            final_throttle = 0.0;
            final_brake = 1.0;
        }

    } else if (planning_state_ == ASState::AS_DRIVING) {
        const auto& planning_params = (current_mode_ == DrivingMode::RACING)
                                        ? planning_params_->trajectory_generation.racing_mode
                                        : planning_params_->trajectory_generation.mapping_mode;

        const auto& control_params = (current_mode_ == DrivingMode::RACING)
                                         ? control_params_->racing_mode
                                         : control_params_->mapping_mode;

        final_steering = lateral_controller_->calculateSteeringAngle(vehicle_state, trajectory_points_, control_params, control_params_->vehicle_length_);
        
        double base_target_speed = trajectory_points_.empty() ? 0.0 : trajectory_points_[0].speed;        
        double steering_dampening = std::abs(final_steering) * control_params.steering_based_speed_gain_;
        double final_target_speed = base_target_speed - steering_dampening;
        final_target_speed = std::max(planning_params.min_speed_, final_target_speed);
        
        double control_effort = longitudinal_controller_->calculate(
            final_target_speed, 
            vehicle_state.speed, 
            control_params.pid_kp_, 
            control_params.pid_ki_, 
            control_params.pid_kd_, 
            -control_params.max_brake_,    // min_output for PID
            control_params.max_throttle_   // max_output for PID
        );

        double current_complexity = trajectory_points_.empty() ? 0.0 : trajectory_points_[0].complexity_score;

        if (control_effort > 0.0) {
            final_throttle = control_effort;
            final_brake = 0.0;
            
        } else {
            // check complexity score to decide braking strategy
            if (current_complexity > control_params.brake_activation_complexity_threshold_) { // brake required
                final_throttle = 0.0;
                final_brake = -control_effort; // control_effort is negative, so - makes it positive
            } else { // break not required
                final_throttle = 0.0;
                final_brake = 0.0;
            }
        }

        ROS_INFO_THROTTLE(1.0, "[DEBUG] Complexity: %.2f | Brake Threshold: %.2f | Final Brake: %.2f",
                          current_complexity,
                          control_params.brake_activation_complexity_threshold_,
                          final_brake);
    }

    // 계산된 값을 최종적으로 제어 명령에 '할당' (한 곳에서만 처리)
    control_command_msg.steering = -final_steering; // FSDS 좌표계에 맞게 음수(-) 적용
    control_command_msg.throttle = final_throttle;
    control_command_msg.brake = final_brake;

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

/**
 * @brief Manages the overall racing strategy, including lap counting and mode switching.
 */

void FormulaAutonomousSystem::setRacingStrategy(const VehicleState& vehicle_state, const std::vector<Cone>& cones_for_planning) {
    // 1. Define the start/finish line if it hasn't been defined yet.
    if (!is_start_finish_line_defined_) {
        defineStartFinishLine(vehicle_state, cones_for_planning);
    }

    // 2. Update the lap count if the line has been defined.
    if (is_start_finish_line_defined_) {
        updateVehiclePositionRelativeToLine(vehicle_state);

        if (!is_race_finished_) {
            updateLapCount(vehicle_state);
        }
    }
    
    // 3. Switch driving mode and generate global path after lap 1.
    if (current_lap_ > 1 && current_mode_ == DrivingMode::MAPPING && !is_global_path_generated_) {

        ROS_INFO("FormulaAutonomousSystem: Lap 1 finished. Generating Global Path...");
        generateGlobalPath(); // Call the path generation function

        if (!global_path_.empty()) {

            is_global_path_generated_ = true;
            current_mode_ = DrivingMode::RACING;
            ROS_INFO("FormulaAutonomousSystem: Global Path generated with %zu points. Switched to RACING mode!", global_path_.size());

        } else {
            ROS_ERROR("FormulaAutonomousSystem: Global Path generation failed. Staying in MAPPING mode.");
        }
    }
}

/**
 * @brief Defines the start/finish line using four orange cones.
 */
void FormulaAutonomousSystem::defineStartFinishLine(const VehicleState& vehicle_state, const std::vector<Cone>& cones) {

    std::vector<Eigen::Vector2d> orange_cones;

    for (const auto& cone : cones) {
        if (cone.color == "orange" && cone.center.x > 0 && cone.center.x < 15.0) { // 전방 15m 이내 주황 콘
            orange_cones.emplace_back(cone.center.x, cone.center.y);
        }
    }

    if (orange_cones.size() < 4) {
        return; // Not enough orange cones detected yet.
    }

    // Sort cones by y-coordinate to separate left and right pairs
    std::sort(orange_cones.begin(), orange_cones.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return a.y() > b.y(); // y가 큰 쪽이 왼쪽
    });

    // The two cones with the largest y are left, the two with the smallest y are right.
    Eigen::Vector2d left_midpoint = (orange_cones[0] + orange_cones[1]) / 2.0;
    Eigen::Vector2d right_midpoint = (orange_cones[orange_cones.size()-1] + orange_cones[orange_cones.size()-2]) / 2.0;
    Eigen::Vector2d local_center = (left_midpoint + right_midpoint) / 2.0;
    Eigen::Vector2d local_direction = (left_midpoint - right_midpoint).normalized(); // The direction vector of the line goes from right to left.

    // Vehicle frame -> Global frame
    double vehicle_x = vehicle_state.position.x();
    double vehicle_y = vehicle_state.position.y();
    double vehicle_yaw = vehicle_state.yaw;

    start_finish_line_center_.x() = vehicle_x + local_center.x() * std::cos(vehicle_yaw) - local_center.y() * std::sin(vehicle_yaw);
    start_finish_line_center_.y() = vehicle_y + local_center.x() * std::sin(vehicle_yaw) + local_center.y() * std::cos(vehicle_yaw);

    start_finish_line_direction_.x() = local_direction.x() * std::cos(vehicle_yaw) - local_direction.y() * std::sin(vehicle_yaw);
    start_finish_line_direction_.y() = local_direction.x() * std::sin(vehicle_yaw) + local_direction.y() * std::cos(vehicle_yaw);

    // The yaw of the line is perpendicular to its direction.
    start_finish_line_yaw_ = std::atan2(start_finish_line_direction_.y(), start_finish_line_direction_.x()) - M_PI / 2.0;

    is_start_finish_line_defined_ = true;
    just_crossed_line_ = true; // IMPORTANT: Prevent counting the first pass right after starting.
    current_lap_ = 1;
    last_lap_time_ = ros::Time::now();

    ROS_INFO("FormulaAutonomousSystem: Start/Finish line defined at GLOBAL (%.2f, %.2f) with direction (%.2f, %.2f)",
             start_finish_line_center_.x(), start_finish_line_center_.y(),
             start_finish_line_direction_.x(), start_finish_line_direction_.y());
}

/**
 * @brief Updates the lap count when the vehicle crosses the start/finish line.
 */
void FormulaAutonomousSystem::updateLapCount(const VehicleState& current_state) {

    Eigen::Vector2d car_position_vec(current_state.position.x(), current_state.position.y());

    // 1. Proximity Gate
    double dist_from_line_center = (car_position_vec - start_finish_line_center_).norm();
    const double PROXIMITY_GATE_RADIUS = 10.0;

    // Reset the 'just_crossed_line_' flag only when the car is far away from the line on the "after" side.
    // This prevents re-triggering if the car wiggles across the line.
    if (dist_from_line_center > PROXIMITY_GATE_RADIUS) {
        if (just_crossed_line_) {
            just_crossed_line_ = false;
        }
        return; // Quit function
    }

    // Check for sign change (crossing the line) using the continuously updated member variables
    if (previous_position_relative_to_line_ < 0 && vehicle_position_relative_to_line_ >= 0) {

        // Time filter
        const double MINIMUM_LAP_TIME_SEC = 15.0; // Prevent double counting on wiggles
        ros::Duration lap_duration = ros::Time::now() - last_lap_time_;

        if (!just_crossed_line_ && lap_duration.toSec() > MINIMUM_LAP_TIME_SEC) {
            current_lap_++;
            just_crossed_line_ = true; // Set flag to prevent double counting
            last_lap_time_ = ros::Time::now();

            ROS_INFO("================================================");
            ROS_INFO("FormulaAutonomousSystem: Crossed line! New Lap: %d", current_lap_);
            ROS_INFO("================================================");

            if (current_lap_ > planning_params_->behavioral_logic.total_laps_) {
                ROS_INFO("================================================");
                ROS_INFO("FormulaAutonomousSystem: Final lap complete! Finishing race...");
                ROS_INFO("================================================");
                is_race_finished_ = true;
                state_machine_->processEvent(ASEvent::RACE_FINISHED);
            }
        }
    }
}

void FormulaAutonomousSystem::updateVehiclePositionRelativeToLine(const VehicleState& current_state) {
    Eigen::Vector2d car_position_vec(current_state.position.x(), current_state.position.y());
    Eigen::Vector2d vec_to_car = car_position_vec - start_finish_line_center_;

    // Dynamic Normal Vector for pass test
    Eigen::Vector2d line_normal_candidate1(-start_finish_line_direction_.y(), start_finish_line_direction_.x());
    Eigen::Vector2d line_normal_candidate2(start_finish_line_direction_.y(), -start_finish_line_direction_.x());

    Eigen::Vector2d vehicle_heading_vec(std::cos(current_state.yaw), std::sin(current_state.yaw));
    Eigen::Vector2d line_normal = (line_normal_candidate1.dot(vehicle_heading_vec) > line_normal_candidate2.dot(vehicle_heading_vec))
                                    ? line_normal_candidate1 : line_normal_candidate2;

    // Update previous and current relative positions
    previous_position_relative_to_line_ = vehicle_position_relative_to_line_;
    vehicle_position_relative_to_line_ = vec_to_car.dot(line_normal);
}