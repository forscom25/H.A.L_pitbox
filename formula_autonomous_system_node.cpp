/**
 * @file formula_autonomous_system.cpp
 * @author Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * @brief 
 * @version 0.1
 * @date 2025-07-21
 * 
 * @copyright Copyright (c) 2025
 */

#include "formula_autonomous_system_node.hpp"

FormulaAutonomousSystemNode::FormulaAutonomousSystemNode():
    is_initialized_(false),
    main_loop_rate_(1000.0),
    nh_("~"),
    pnh_("~"),
    lidar_msg_(),
    camera1_msg_(),
    camera2_msg_(),
    imu_msg_(),
    gps_msg_(),
    go_signal_msg_(),
    is_lidar_msg_init_(false),
    is_camera1_msg_init_(false),
    is_camera2_msg_init_(false),
    is_imu_msg_init_(false),
    is_gps_msg_init_(false),
    is_go_signal_msg_init_(false),
    lidar_msg_mutex_(),
    camera1_msg_mutex_(),
    camera2_msg_mutex_(),
    imu_msg_mutex_(),
    gps_msg_mutex_(),
    go_signal_msg_mutex_(),
    formula_autonomous_system_(nullptr) {   
    init();
}

FormulaAutonomousSystemNode::~FormulaAutonomousSystemNode(){
    ROS_INFO("FormulaAutonomousSystemNode: Destructor called");

    // Thread join
    main_thread_.join();

    return;
}

bool FormulaAutonomousSystemNode::init(){

    // Initialize tf listener (constructor initialization)
    // tf_listener_ is initialized in constructor with tf_buffer_

    // Initialize subscribers
    lidar_sub_ = nh_.subscribe("/fsds/lidar/Lidar1", 1, &FormulaAutonomousSystemNode::lidarCallback, this);
    camera1_sub_ = nh_.subscribe("/fsds/cameracam1", 1, &FormulaAutonomousSystemNode::camera1Callback, this);
    camera2_sub_ = nh_.subscribe("/fsds/cameracam2", 1, &FormulaAutonomousSystemNode::camera2Callback, this);
    imu_sub_ = nh_.subscribe("/fsds/imu", 1, &FormulaAutonomousSystemNode::imuCallback, this);
    gps_sub_ = nh_.subscribe("/fsds/gps", 1, &FormulaAutonomousSystemNode::gpsCallback, this);
    go_signal_sub_ = nh_.subscribe("/fsds/signal/go", 1, &FormulaAutonomousSystemNode::goSignalCallback, this);

    // Initialize publishers
    control_pub_ = nh_.advertise<fs_msgs::ControlCommand>("/fsds/control_command", 1);
    autonomous_mode_pub_ = nh_.advertise<std_msgs::String>("/fsds/AS_status", 1);
    vehicle_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/fsds/vehicle_odom", 1);
    non_ground_point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/fsds/non_ground_point_cloud", 1);
    ground_point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/fsds/ground_point_cloud", 1);
    detected_cones_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/detected_cones_marker", 1);
    projected_cones_image_pub_ = nh_.advertise<sensor_msgs::Image>("/fsds/projected_cones_image", 1);
    center_line_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/center_line_marker", 1);
    lap_count_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/lap_count_marker", 1);
    global_cones_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/global_cones_marker", 1);
    lane_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/lane_marker", 1);
    global_path_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/fsds/global_path", 1);                                // Globalpath
    trajectory_from_global_path_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/fsds/trajectory_from_global_path", 1); // TrajectoryFromGlobalpath
    critical_sections_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/fsds/critical_sections_marker", 1); // critical section debugging
    
    // Get parameters
    pnh_.getParam("/system/main_loop_rate", main_loop_rate_);

    // Algorithms
    formula_autonomous_system_ = std::make_unique<FormulaAutonomousSystem>();
    formula_autonomous_system_->init(pnh_);

    is_initialized_ = true;
    return true;
}

bool FormulaAutonomousSystemNode::getParameters(){
    // Get parameters
    return true;
}

void FormulaAutonomousSystemNode::run(){

    if(checkEssentialMessages() == false){
        return;
    }
    
    // Get lidar point cloud
    lidar_msg_mutex_.lock();
    auto lidar_msg = lidar_msg_;
    lidar_msg_mutex_.unlock();

    // Get camera image
    camera1_msg_mutex_.lock();
    auto camera1_msg = camera1_msg_;
    camera1_msg_mutex_.unlock();

    // Get camera image
    camera2_msg_mutex_.lock();
    auto camera2_msg = camera2_msg_;
    camera2_msg_mutex_.unlock();

    // Get imu data
    imu_msg_mutex_.lock();
    auto imu_msg = imu_msg_;
    imu_msg_mutex_.unlock();

    // Get gps data
    gps_msg_mutex_.lock();  
    auto gps_msg = gps_msg_;
    gps_msg_mutex_.unlock();

    // Get go signal
    go_signal_msg_mutex_.lock();
    auto go_signal_msg = go_signal_msg_;
    go_signal_msg_mutex_.unlock();

    // Get control command
    fs_msgs::ControlCommand control_command;
    std_msgs::String autonomous_mode;
    formula_autonomous_system_->run(lidar_msg, camera1_msg, camera2_msg, imu_msg, gps_msg, go_signal_msg, control_command, autonomous_mode);
    control_command_msg_ = control_command;
    autonomous_mode_msg_ = autonomous_mode;

    return;
}

void FormulaAutonomousSystemNode::publish(){

    // Publish
    publishControl();
    publishAutonomousMode();
    publishVehicleOdom();
    publishNonGroundPointCloud();
    publishGroundPointCloud();
    publishDetectedConesMarker();
    publishProjectedConesImage();
    publishCenterLineMarker();
    publishGlobalConesMarker();
    publishLaneMarker();
    publishStartFinishLineMarker();
    publishGlobalPathMarker();                  // Globalpath
    publishTrajectoryFromGlobalPathMarker();    // TrajectoryFromGlobalpath
    publishCriticalSectionsMarker();            // CriticalSection
    return;
}

bool FormulaAutonomousSystemNode::checkEssentialMessages(){
    // Check essential messages
    std::stringstream ss;
    bool is_essential_message_updated = true;
    if(is_lidar_msg_init_ == false){
        ss << "Wait for lidar message ";
        is_essential_message_updated = false;
    }
    if(is_camera1_msg_init_ == false){
        ss << "Wait for camera1 message ";
        is_essential_message_updated = false;
    }
    if(is_camera2_msg_init_ == false){
        ss << "Wait for camera2 message ";
        is_essential_message_updated = false;
    }
    if(is_imu_msg_init_ == false){
        ss << "Wait for imu message ";
        is_essential_message_updated = false;
    }
    if(is_gps_msg_init_ == false){
        ss << "Wait for gps message ";
        is_essential_message_updated = false;
    }

    if(is_essential_message_updated == false){
        ROS_WARN_STREAM(ss.str());
    }

    return true;
}

void FormulaAutonomousSystemNode::publishControl(){
    // Publish
    control_pub_.publish(control_command_msg_);
    return;
}

void FormulaAutonomousSystemNode::publishAutonomousMode(){
    // Publish
    autonomous_mode_pub_.publish(autonomous_mode_msg_);
    return;
}

void FormulaAutonomousSystemNode::publishVehicleOdom(){
    // Publish
    std_msgs::Header header;
    imu_msg_mutex_.lock();
    header = imu_msg_.header;
    double yaw_rate = imu_msg_.angular_velocity.z;
    imu_msg_mutex_.unlock();
    
    nav_msgs::Odometry odom;
    odom.header = header;
    odom.header.frame_id = "map";
    odom.child_frame_id = header.frame_id;

    auto state = formula_autonomous_system_->localization_->getCurrentState();
    odom.pose.pose.position.x = state[0];
    odom.pose.pose.position.y = state[1];
    odom.pose.pose.position.z = 0.0;
    double yaw = state[2];
    
    // Set rotation (orientation from yaw angle)
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);  // roll=0, pitch=0, yaw=pose[2]
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom.twist.twist.linear.x = state[3];
    odom.twist.twist.linear.y = state[4];
    odom.twist.twist.linear.z = 0.0;
    odom.twist.twist.angular.x = 0.0;
    odom.twist.twist.angular.y = 0.0;
    odom.twist.twist.angular.z = yaw_rate;

    vehicle_odom_pub_.publish(odom);

    // Publish tf using tf2
    geometry_msgs::TransformStamped transform_stamped;
    transform_stamped.header.stamp = header.stamp;
    transform_stamped.header.frame_id = "map";
    transform_stamped.child_frame_id = header.frame_id;
    
    transform_stamped.transform.translation.x = odom.pose.pose.position.x;
    transform_stamped.transform.translation.y = odom.pose.pose.position.y;
    transform_stamped.transform.translation.z = odom.pose.pose.position.z;
    
    transform_stamped.transform.rotation = odom.pose.pose.orientation;
    
    tf_broadcaster_.sendTransform(transform_stamped);

    return;
}

void FormulaAutonomousSystemNode::publishNonGroundPointCloud(){
    // Publish
    // Get header
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();

    // Publish non-ground point cloud
    sensor_msgs::PointCloud2 non_ground_point_cloud_msg;
    pcl::toROSMsg(*formula_autonomous_system_->non_ground_point_cloud_, non_ground_point_cloud_msg);
    non_ground_point_cloud_msg.header = header;
    non_ground_point_cloud_pub_.publish(non_ground_point_cloud_msg);

    return;
}

void FormulaAutonomousSystemNode::publishGroundPointCloud(){
    // Publish
    // Get header
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();

    // Publish ground point cloud
    sensor_msgs::PointCloud2 ground_point_cloud_msg;
    pcl::toROSMsg(*formula_autonomous_system_->ground_point_cloud_, ground_point_cloud_msg);
    ground_point_cloud_msg.header = header;
    ground_point_cloud_pub_.publish(ground_point_cloud_msg);

    return;
}

void FormulaAutonomousSystemNode::publishDetectedConesMarker(){
    // Publish
    // Get header
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();
    header.frame_id = "fsds/FSCar";

    // Publish detected cones marker
    visualization_msgs::MarkerArray marker_array;
    int id = 0;
    for(const auto& cone : formula_autonomous_system_->cones_){
        visualization_msgs::Marker marker;
        marker.header = header;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(0.1);  // Show for 0.1 seconds
        marker.id = id++;
        marker.pose.position.x = cone.center.x;
        marker.pose.position.y = cone.center.y;
        marker.pose.position.z = cone.center.z;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        // Set marker color based on cone color
        if (cone.color == "yellow") {
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        } else if (cone.color == "blue") {
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
        } else if (cone.color == "orange") {
            marker.color.r = 1.0;
            marker.color.g = 0.5;
            marker.color.b = 0.0;
        } else if (cone.color == "out_of_image") {
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
        } else { // unknown or any other color
            marker.color.r = 0.5;
            marker.color.g = 0.5;
            marker.color.b = 0.5;
        }
        marker.color.a = 1.0;
        marker_array.markers.push_back(marker);
    }
    detected_cones_marker_pub_.publish(marker_array);
    return;
}

void FormulaAutonomousSystemNode::publishProjectedConesImage(){
    // Publish
    sensor_msgs::Image projected_cones_image_msg;
    cv_bridge::CvImage cv_image;
    cv_image.header = camera1_msg_.header;
    cv_image.encoding = sensor_msgs::image_encodings::BGR8;
    cv_image.image = formula_autonomous_system_->projected_cones_image_;
    cv_image.toImageMsg(projected_cones_image_msg);  
    projected_cones_image_pub_.publish(projected_cones_image_msg);
    return;
}

void FormulaAutonomousSystemNode::publishGlobalConesMarker() {
    std_msgs::Header header;
    imu_msg_mutex_.lock();
    header.stamp = imu_msg_.header.stamp;
    imu_msg_mutex_.unlock();
    header.frame_id = "map"; // 전역 좌표계인 "map" 프레임 기준

    visualization_msgs::MarkerArray marker_array;

    // MapManager에 저장된 전역 콘 데이터를 가져옵니다.
    std::vector<Cone> global_cones = formula_autonomous_system_->getGlobalConeMap();

    int id = 0;
    for (const auto& cone : global_cones) {
        visualization_msgs::Marker marker;
        marker.header = header;
        marker.ns = "global_cones";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(0.2); // 0.2초 동안 유지

        // Use global coordinate
        marker.pose.position.x = cone.center.x;
        marker.pose.position.y = cone.center.y;
        marker.pose.position.z = cone.center.z;
        marker.pose.orientation.w = 1.0;

        // Cone size
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;

        // Cone color
        if (cone.color == "yellow") {
            marker.color.r = 1.0;
            marker.color.g = 1.0;
        } else if (cone.color == "blue") {
            marker.color.b = 1.0;
        } else if (cone.color == "orange") {
            marker.color.r = 1.0;
            marker.color.g = 0.5;
        } else { // unknown
            marker.color.r = 0.5;
            marker.color.g = 0.5;
            marker.color.b = 0.5;
        }
        marker.color.a = 1.0; // 불투명

        marker_array.markers.push_back(marker);
    }

    global_cones_marker_pub_.publish(marker_array);
}

void FormulaAutonomousSystemNode::publishCenterLineMarker(){
    // Publish
    // Get header
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();
    header.frame_id = "fsds/FSCar";

    // Publish center line marker
    visualization_msgs::MarkerArray center_line_marker;
    visualization_msgs::Marker marker_line;
    marker_line.header = header;
    marker_line.type = visualization_msgs::Marker::LINE_STRIP;
    marker_line.action = visualization_msgs::Marker::ADD;
    marker_line.lifetime = ros::Duration(0.1);  // Show for 0.1 seconds
    marker_line.id = -1;  // Single line strip marker
    marker_line.pose.orientation.x = 0.0;
    marker_line.pose.orientation.y = 0.0;
    marker_line.pose.orientation.z = 0.0;
    marker_line.pose.orientation.w = 1.0;
    marker_line.scale.x = 0.1;  // Line width
    marker_line.color.r = 0.0;
    marker_line.color.g = 1.0;
    marker_line.color.b = 0.0;
    marker_line.color.a = 1.0;
    for(int i = 0; i < formula_autonomous_system_->trajectory_points_.size(); i++){
        visualization_msgs::Marker marker_speed;
        marker_speed.header = header;
        marker_speed.type = visualization_msgs::Marker::LINE_STRIP;
        marker_speed.action = visualization_msgs::Marker::ADD;
        marker_speed.lifetime = ros::Duration(0.1);  // Show for 0.1 seconds
        marker_speed.id = i;  // Single line strip marker
        marker_speed.pose.orientation.x = 0.0;
        marker_speed.pose.orientation.y = 0.0;
        marker_speed.pose.orientation.z = 0.0;
        marker_speed.pose.orientation.w = 1.0;
        marker_speed.scale.x = 0.1;  // Line width
        marker_speed.color.r = 0.0;
        marker_speed.color.g = 1.0;
        marker_speed.color.b = 0.0;
        marker_speed.color.a = 1.0;

        // Add first point
        geometry_msgs::Point first_point;
        first_point.x = formula_autonomous_system_->trajectory_points_[i].position.x();
        first_point.y = formula_autonomous_system_->trajectory_points_[i].position.y();
        first_point.z = 0.0;
        marker_speed.points.push_back(first_point);

        // Add last point
        geometry_msgs::Point last_point;
        last_point.x = formula_autonomous_system_->trajectory_points_[i].position.x();
        last_point.y = formula_autonomous_system_->trajectory_points_[i].position.y();
        last_point.z = formula_autonomous_system_->trajectory_points_[i].speed * 0.3; // 0.3 is a scale factor
        marker_speed.points.push_back(last_point);
        
        // Add marker to array
        center_line_marker.markers.push_back(marker_speed);

        // Add line marker points
        geometry_msgs::Point traj_point;
        traj_point.x = formula_autonomous_system_->trajectory_points_[i].position.x();
        traj_point.y = formula_autonomous_system_->trajectory_points_[i].position.y();
        traj_point.z = 0.0;
        marker_line.points.push_back(traj_point);

    }
    center_line_marker.markers.push_back(marker_line);
    center_line_marker_pub_.publish(center_line_marker);
    return;
}

void FormulaAutonomousSystemNode::publishLaneMarker() {
    // Get header from a reliable source like lidar message
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();
    header.frame_id = "map"; // The frame should be the vehicle's base frame

    visualization_msgs::MarkerArray marker_array;
    
    // Get the generated lane points from the core system
    auto lanes = formula_autonomous_system_->getGlobalTrackLanes();
    auto left_lane_points = lanes.first;
    auto right_lane_points = lanes.second;

    // --- Left Lane Marker (Blue) ---
    visualization_msgs::Marker left_lane_marker;
    left_lane_marker.header = header;
    left_lane_marker.ns = "lanes";
    left_lane_marker.id = 0;
    left_lane_marker.type = visualization_msgs::Marker::LINE_STRIP;
    left_lane_marker.action = visualization_msgs::Marker::ADD;
    left_lane_marker.lifetime = ros::Duration(0.1);
    left_lane_marker.pose.orientation.w = 1.0;
    left_lane_marker.scale.x = 0.1; // Line width
    left_lane_marker.color.b = 1.0; // Blue color
    left_lane_marker.color.a = 1.0; // Alpha (opacity)

    for (const auto& point : left_lane_points) {
        geometry_msgs::Point p;
        p.x = point.x();
        p.y = point.y();
        p.z = 0.0;
        left_lane_marker.points.push_back(p);
    }
    marker_array.markers.push_back(left_lane_marker);

    // --- Right Lane Marker (Yellow) ---
    visualization_msgs::Marker right_lane_marker = left_lane_marker; // Copy properties
    right_lane_marker.id = 1;
    right_lane_marker.color.b = 0.0; // Reset blue
    right_lane_marker.color.r = 1.0; // Set red
    right_lane_marker.color.g = 1.0; // Set green (R+G = Yellow)
    right_lane_marker.points.clear(); // Clear points from the copied marker

    for (const auto& point : right_lane_points) {
        geometry_msgs::Point p;
        p.x = point.x();
        p.y = point.y();
        p.z = 0.0;
        right_lane_marker.points.push_back(p);
    }
    marker_array.markers.push_back(right_lane_marker);

    // Publish the marker array
    lane_marker_pub_.publish(marker_array);
}

// For lap counting



void FormulaAutonomousSystemNode::publishStartFinishLineMarker() {

    if (!formula_autonomous_system_->isStartFinishLineDefined()) {
        return;
    }
    
    // Get the current timestamp from a reliable source
    imu_msg_mutex_.lock();
    ros::Time current_stamp = imu_msg_.header.stamp;
    imu_msg_mutex_.unlock();

    visualization_msgs::MarkerArray marker_array;

    // --- 1. Line Marker (in "map" frame) ---
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = current_stamp;
    line_marker.ns = "start_finish_line";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;
    line_marker.lifetime = ros::Duration(0.2);
    line_marker.pose.orientation.w = 1.0;

    // MODIFIED: Make the line thicker and orange
    line_marker.scale.x = 0.5; // <-- CHANGE: Increased thickness
    line_marker.color.r = 1.0; // Orange
    line_marker.color.g = 0.5; // Orange
    line_marker.color.b = 0.0; // Orange
    line_marker.color.a = 1.0;

    Eigen::Vector2d center = formula_autonomous_system_->getStartFinishLineCenter();
    Eigen::Vector2d dir = formula_autonomous_system_->getStartFinishLineDirection();

    double track_width = 7.0;

    geometry_msgs::Point p1, p2;
    p1.x = center.x() - dir.x() * track_width / 2.0;
    p1.y = center.y() - dir.y() * track_width / 2.0;
    p1.z = 0.1;

    p2.x = center.x() + dir.x() * track_width / 2.0;
    p2.y = center.y() + dir.y() * track_width / 2.0;
    p2.z = 0.1;

    line_marker.points.push_back(p1);
    line_marker.points.push_back(p2);
    marker_array.markers.push_back(line_marker);

    // --- 2. Text Marker for Lap Count (in "fsds/FSCar" vehicle frame) ---
    visualization_msgs::Marker text_marker;
    text_marker.header.frame_id = "fsds/FSCar";
    text_marker.header.stamp = current_stamp;
    text_marker.ns = "lap_count_text";
    text_marker.id = 1;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.lifetime = ros::Duration(0.2);
    text_marker.pose.position.x = 4.0;
    text_marker.pose.position.y = 4.0;
    text_marker.pose.position.z = 4.0;
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.z = 0.8;

    // MODIFIED: Make the text green
    text_marker.color.r = 0.0; // Green
    text_marker.color.g = 1.0; // Green
    text_marker.color.b = 0.0; // Green
    text_marker.color.a = 1.0;

    int total_laps = 2;
    std::stringstream ss;
    ss << "Lap: " << formula_autonomous_system_->getCurrentLap() << " / " << total_laps;
    text_marker.text = ss.str();
    marker_array.markers.push_back(text_marker);
    lap_count_marker_pub_.publish(marker_array);
}

// Globalpath
void FormulaAutonomousSystemNode::publishGlobalPathMarker() {

    if (!formula_autonomous_system_->isGlobalPathGenerated()) {
        return; // Don't publish if the path doesn't exist yet
    }

    auto global_path = formula_autonomous_system_->getGlobalPath();

    if (global_path.empty()) {
        return;
    }

    // Get header from a reliable source
    imu_msg_mutex_.lock();
    std_msgs::Header header = imu_msg_.header;
    imu_msg_mutex_.unlock();
    header.frame_id = "map"; // The global path is in the "map" frame
    visualization_msgs::Marker path_marker;
    path_marker.header = header;
    path_marker.ns = "global_path";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::Marker::ADD;
    path_marker.lifetime = ros::Duration(); // A duration of 0 means it persists forever
    path_marker.pose.orientation.w = 1.0;
    path_marker.scale.x = 0.15; // Line width
    path_marker.color.r = 0.0;  // Cyan color for distinction
    path_marker.color.g = 1.0;
    path_marker.color.b = 1.0;
    path_marker.color.a = 0.8;  // Slightly transparent

    for (const auto& point : global_path) {

        geometry_msgs::Point p;
        p.x = point.position.x();
        p.y = point.position.y();
        p.z = 0.1; // Draw it slightly above the ground to avoid z-fighting
        path_marker.points.push_back(p);
    }

    global_path_marker_pub_.publish(path_marker);
}

// GlobalpathTrajectory
void FormulaAutonomousSystemNode::publishTrajectoryFromGlobalPathMarker() { 

    if (!formula_autonomous_system_->isGlobalPathGenerated()) {
        return;
    }

    auto local_points = formula_autonomous_system_->getTrajectoryGenerator()->getLastLocalPathPoints();

    if (local_points.empty()) {
        return;
    }

    imu_msg_mutex_.lock();
    std_msgs::Header header = imu_msg_.header;
    imu_msg_mutex_.unlock();
    header.frame_id = "fsds/FSCar";
    visualization_msgs::Marker points_marker;
    points_marker.header = header;
    points_marker.ns = "trajectory_from_global_path"; 
    points_marker.id = 0;
    points_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    points_marker.action = visualization_msgs::Marker::ADD;
    points_marker.lifetime = ros::Duration(0.1);
    points_marker.pose.orientation.w = 1.0;
    points_marker.scale.x = 0.2;
    points_marker.scale.y = 0.2;
    points_marker.scale.z = 0.2;
    points_marker.color.r = 1.0;
    points_marker.color.g = 0.0;
    points_marker.color.b = 0.0;
    points_marker.color.a = 1.0;

    for (const auto& point : local_points) {

        geometry_msgs::Point p;
        p.x = point.x();
        p.y = point.y();
        p.z = 0.2;
        points_marker.points.push_back(p);
    }

    trajectory_from_global_path_marker_pub_.publish(points_marker); 
}

void FormulaAutonomousSystemNode::publishCriticalSectionsMarker() {
    if (!formula_autonomous_system_->isGlobalPathGenerated()) {
        return;
    }

    auto global_path = formula_autonomous_system_->getGlobalPath();
    auto critical_sections = formula_autonomous_system_->getCriticalSections();

    if (global_path.empty() || critical_sections.empty()) {
        return;
    }

    imu_msg_mutex_.lock();
    std_msgs::Header header = imu_msg_.header;
    imu_msg_mutex_.unlock();
    header.frame_id = "map";

    visualization_msgs::Marker sections_marker;
    sections_marker.header = header;
    sections_marker.ns = "critical_sections";
    sections_marker.id = 0;
    sections_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    sections_marker.action = visualization_msgs::Marker::ADD;
    sections_marker.lifetime = ros::Duration(0.2);
    sections_marker.pose.orientation.w = 1.0;
    sections_marker.scale.x = 0.4; // Sphere diameter
    sections_marker.scale.y = 0.4;
    sections_marker.scale.z = 0.4;
    sections_marker.color.r = 1.0; // Bright red
    sections_marker.color.g = 0.0;
    sections_marker.color.b = 0.0;
    sections_marker.color.a = 1.0;

    for (const auto& section : critical_sections) {
        for (size_t i = section.first; i <= section.second; ++i) {
            if (i < global_path.size()) {
                geometry_msgs::Point p;
                p.x = global_path[i].position.x();
                p.y = global_path[i].position.y();
                p.z = 0.2; // Elevate slightly to be visible over the path
                sections_marker.points.push_back(p);
            }
        }
    }
    critical_sections_marker_pub_.publish(sections_marker);
}

int main(int argc, char** argv){
    // Initialize ROS
    ros::init(argc, argv, "formula_autonomous_system");

    // Create FormulaAutonomousSystemNode object
    FormulaAutonomousSystemNode formula_autonomous_system;
    formula_autonomous_system.excute();

    ros::spin();

    return 0;
}