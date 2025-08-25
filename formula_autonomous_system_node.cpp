/**
 * @file formula_autonomous_system.cpp
 * @author Jiwon Seok (jiwonseok@hanyang.ac.kr)
 * @brief 
 * @version 0.1
 * @date 2025-07-21
 * @copyright Copyright (c) 2025
 */

#include "formula_autonomous_system_node.hpp"

FormulaAutonomousSystemNode::FormulaAutonomousSystemNode():
    is_initialized_(false),
    main_loop_rate_(1000.0),
    nh_("~"),
    pnh_("~"),
    is_ready_to_publish_(false),
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
    lane_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/lane_marker", 1);
    finish_line_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/finish_line_marker", 1);
    global_path_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fsds/global_path_marker", 1);
    
    // Get parameters
    pnh_.getParam("/system/main_loop_rate", main_loop_rate_);

    // Algorithms
    formula_autonomous_system_ = std::make_unique<FormulaAutonomousSystem>();
    formula_autonomous_system_->init(pnh_);

    is_ready_to_publish_ = true;
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
    if (!is_ready_to_publish_){
        return;
    }

    // Publish
    publishControl();
    publishAutonomousMode();
    publishVehicleOdom();
    publishNonGroundPointCloud();
    publishGroundPointCloud();
    publishDetectedConesMarker();
    publishProjectedConesImage();
    publishCenterLineMarker();
    publishLaneMarker();
    publishLapCountMarker();
    publishGlobalPathMarker();
    publishFinishLineMarker();

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
//  =================================================================
// 값을 6단계 색상 스펙트럼으로 변환하는 헬퍼 함수
// value: 변환할 값 (예: 속도), min_value/max_value: 값의 범위
std_msgs::ColorRGBA getColorByValue(double value, double min_value, double max_value) {
    std_msgs::ColorRGBA color;
    color.a = 1.0;

    // 값을 0.0 ~ 1.0 사이로 정규화
    double ratio = (value - min_value) / (max_value - min_value);
    ratio = std::max(0.0, std::min(1.0, ratio)); // 0과 1 사이로 값 고정

    // 6단계 색상 (빨강-주황-노랑-연두-하늘-파랑)
    int stage = static_cast<int>(ratio / 0.2);
    double local_ratio = (ratio - stage * 0.2) / 0.2;

    switch (stage) {
        case 0: // 빨강 -> 주황
            color.r = 1.0;
            color.g = local_ratio * 0.5;
            color.b = 0.0;
            break;
        case 1: // 주황 -> 노랑
            color.r = 1.0;
            color.g = 0.5 + local_ratio * 0.5;
            color.b = 0.0;
            break;
        case 2: // 노랑 -> 연두
            color.r = 1.0 - local_ratio;
            color.g = 1.0;
            color.b = 0.0;
            break;
        case 3: // 연두 -> 하늘
            color.r = 0.0;
            color.g = 1.0;
            color.b = local_ratio;
            break;
        case 4: // 하늘 -> 파랑
        case 5: // ratio가 1.0일 때 포함
            color.r = 0.0;
            color.g = 1.0 - local_ratio;
            color.b = 1.0;
            break;
    }
    return color;
}

void FormulaAutonomousSystemNode::publishCenterLineMarker(){
    // 헤더 정보 가져오기
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();
    header.frame_id = "fsds/FSCar";

    visualization_msgs::MarkerArray marker_array;
    const auto& trajectory = formula_autonomous_system_->trajectory_points_;

    if (trajectory.empty()) {
        // 오래된 마커를 지우기 위해 빈 배열을 발행
        center_line_marker_pub_.publish(marker_array);
        return;
    }

    // --- 1. 경로 라인 (항상 회색으로 고정하여 속도 프로파일 강조) ---
    visualization_msgs::Marker path_marker;
    path_marker.header = header;
    path_marker.ns = "path_line";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::Marker::ADD;
    path_marker.lifetime = ros::Duration(0.1);
    path_marker.pose.orientation.w = 1.0;
    path_marker.scale.x = 0.15; // 라인 두께
    path_marker.color.r = 0.5;  // 회색
    path_marker.color.g = 0.5;
    path_marker.color.b = 0.5;
    path_marker.color.a = 0.6;  // 약간 투명하게

    for(const auto& pt : trajectory){
        geometry_msgs::Point p;
        p.x = pt.position.x();
        p.y = pt.position.y();
        p.z = 0.05; // 지면보다 살짝 위에 표시
        path_marker.points.push_back(p);
    }
    marker_array.markers.push_back(path_marker);


    // --- 2. 속도 프로파일 (속도에 따른 높이 및 6단계 색상 변경) ---
    visualization_msgs::Marker speed_marker;
    speed_marker.header = header;
    speed_marker.ns = "speed_profile";
    speed_marker.id = 1;
    speed_marker.type = visualization_msgs::Marker::LINE_LIST;
    speed_marker.action = visualization_msgs::Marker::ADD;
    speed_marker.lifetime = ros::Duration(0.1);
    speed_marker.pose.orientation.w = 1.0;
    speed_marker.scale.x = 0.1; // 라인 두께

    // ✨ 핵심 변경: 차량의 '현재 실제 속도'를 기준으로 색상 범위를 동적으로 설정 (범위 확대)
    double current_vehicle_speed = formula_autonomous_system_->localization_->getCurrentVelocity();
    double speed_range = 2.5; // 현재 속도 기준 +- 2.5 m/s 범위를 색상으로 표현
    double min_speed_for_viz = current_vehicle_speed - speed_range;
    double max_speed_for_viz = current_vehicle_speed + speed_range;


    for(const auto& pt : trajectory){
        // 수직선의 시작점 (경로 위)
        geometry_msgs::Point p_start;
        p_start.x = pt.position.x();
        p_start.y = pt.position.y();
        p_start.z = 0.05;
        
        // 수직선의 끝점 (높이는 '계획된' 속도에 비례)
        geometry_msgs::Point p_end = p_start;
        p_end.z += pt.speed * 0.5; // 시각화를 위한 스케일 팩터

        speed_marker.points.push_back(p_start);
        speed_marker.points.push_back(p_end);

        // '계획된' 속도를 '현재' 속도 기준으로 6단계 색상 계산
        std_msgs::ColorRGBA speed_color = getColorByValue(pt.speed, min_speed_for_viz, max_speed_for_viz);
        speed_marker.colors.push_back(speed_color);
        speed_marker.colors.push_back(speed_color);
    }
    marker_array.markers.push_back(speed_marker);

    // 최종 마커 배열 발행
    center_line_marker_pub_.publish(marker_array);
    return;
}
//  =================================================================

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

int main(int argc, char** argv){
    // Initialize ROS
    ros::init(argc, argv, "formula_autonomous_system");

    // Create FormulaAutonomousSystemNode object
    FormulaAutonomousSystemNode formula_autonomous_system;
    formula_autonomous_system.excute();

    ros::spin();

    return 0;
}
void FormulaAutonomousSystemNode::publishLapCountMarker(){
    // 헤더 정보 가져오기
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();
    header.frame_id = "fsds/FSCar";

    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "lap_count";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.lifetime = ros::Duration(0.1);
    marker.scale.z = 1.0; // 텍스트 크기

    // 텍스트 위치 (예: 차량 전방 상단)
    marker.pose.position.x = 2.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 2.0;
    marker.pose.orientation.w = 1.0;

    // 텍스트 내용: 현재 랩 수
    int current_lap = formula_autonomous_system_->getCurrentLap();
    marker.text = "Lap: " + std::to_string(current_lap);

    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(marker);
    lap_count_marker_pub_.publish(marker_array);
}
void FormulaAutonomousSystemNode::publishGlobalPathMarker(){
    // 헤더 정보 가져오기
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();
    header.frame_id = "map"; // 글로벌 맵은 "map" 프레임에 기반

    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "global_path";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.lifetime = ros::Duration(0.1);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2; // 라인 두께

    // 글로벌 경로 데이터 가져오기
    const auto& global_path = formula_autonomous_system_->getGlobalPath();

    for(const auto& pt : global_path){
        geometry_msgs::Point p;
        p.x = pt.position.x();
        p.y = pt.position.y();
        p.z = 0.0;
        marker.points.push_back(p);
    }
    
    // 라인 색상 (예: 녹색)
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    // 경로가 비어있으면 마커를 제거
    if (marker.points.empty()) {
        marker.action = visualization_msgs::Marker::DELETE;
    }

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(marker);
    global_path_marker_pub_.publish(marker_array);
}
void FormulaAutonomousSystemNode::publishFinishLineMarker(){
    // 헤더 정보 가져오기
    lidar_msg_mutex_.lock();
    std_msgs::Header header = lidar_msg_.header;
    lidar_msg_mutex_.unlock();
    header.frame_id = "map"; // map 프레임 기반

    // 피니시 라인 정보 가져오기
    const Eigen::Vector2d& line_center = formula_autonomous_system_->getStartFinishLineCenter();
    double line_yaw = formula_autonomous_system_->getStartFinishLineYaw();

    // 라인 시각화 마커 생성
    visualization_msgs::Marker line_marker;
    line_marker.header = header;
    line_marker.ns = "finish_line";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;
    line_marker.lifetime = ros::Duration(0.1);
    line_marker.pose.orientation.w = 1.0;
    line_marker.scale.x = 0.5; // 라인 두께 (더 두껍게 하여 잘 보이게)

    // 라인의 시작점과 끝점 계산 (예: 트랙 폭을 고려하여)
    double line_half_length = 2.0; // 트랙 폭의 절반
    geometry_msgs::Point start_point, end_point;

    start_point.x = line_center.x() - line_half_length * sin(line_yaw);
    start_point.y = line_center.y() + line_half_length * cos(line_yaw);
    start_point.z = 0.0;
    
    end_point.x = line_center.x() + line_half_length * sin(line_yaw);
    end_point.y = line_center.y() - line_half_length * cos(line_yaw);
    end_point.z = 0.0;

    line_marker.points.push_back(start_point);
    line_marker.points.push_back(end_point);

    // 라인 색상 (예: 빨간색)
    line_marker.color.r = 1.0;
    line_marker.color.g = 0.0;
    line_marker.color.b = 0.0;
    line_marker.color.a = 1.0;

    // 마커 배열에 담아 발행
    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(line_marker);
    finish_line_marker_pub_.publish(marker_array);
}
