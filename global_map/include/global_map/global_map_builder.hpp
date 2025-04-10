#ifndef GLOBAL_MAP_GLOBAL_MAP_BUILDER_CLASS_H
#define GLOBAL_MAP_GLOBAL_MAP_BUILDER_CLASS_H

// Standard Libraries
#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <stdlib.h>

// ROS 2 Headers
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/float32.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

// Custom Libraries and Messages
// #include "path_tools.hpp"
#include "voxel_grid.hpp"
#include <env_builder_msgs/msg/voxel_grid_stamped.hpp>
#include <env_builder_msgs/srv/get_voxel_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>

namespace global_map {

// GlobalMapBuilder Class Declaration
class GlobalMapBuilder : public rclcpp::Node {
public:
    // Constructor
    GlobalMapBuilder();

private:
    // Methods
    void DeclareRosParameters(); // Declare ROS parameters
    void InitializeRosParameters(); // Initialize ROS parameters
    void DisplayCompTime(std::vector<double> &comp_time); // Display computation time statistics
    void LocalVoxelGridCallback(const env_builder_msgs::msg::VoxelGridStamped::SharedPtr vg_msg_stamped); // Callback for local voxel grids
    void OnShutdown(); // Execute on node shutdown
    void CreateEmptyGlobalMap();
    void CreateEnvironmentPointCloud();

    void StoreVoxelGrids(::voxel_grid_util::VoxelGrid &vg_global, ::voxel_grid_util::VoxelGrid &vg_last_global);// copy vg
    void MergeVoxelGrids(::voxel_grid_util::VoxelGrid &vg_global,
                                                 const ::voxel_grid_util::VoxelGrid &vg_local); // Merge global and local voxel grids
    void TimerCallbackEnvironmentPC();
    void TimerCallbackEnvironmentVG();

    double ComputeRewards(::voxel_grid_util::VoxelGrid &vg_global, ::voxel_grid_util::VoxelGrid &vg_last_global);
    double ComputeMapCompleteness(::voxel_grid_util::VoxelGrid &vg_global);

    // Publishers and Subscribers
    rclcpp::Publisher<env_builder_msgs::msg::VoxelGridStamped>::SharedPtr global_vg_pub_; // Global voxel grid publisher
    std::vector<rclcpp::Subscription<env_builder_msgs::msg::VoxelGridStamped>::SharedPtr> local_grid_subs_; // Local voxel grid subscriptions

    // Parameters
    std::string env_vg_topic_;
    std::string global_pc_topic_;
    std::string local_grid_topic_;  // Topic name for subscribing to local voxel grids
    std::string agent_frame_;       // Frame ID representing the agent in the global map
    std::string world_frame_;       // Frame ID representing the global world frame

    int n_rob_;             // Number of robots or agents in the system
    int id_;                // Unique identifier for this specific agent or robot
    double voxel_size_;
    std::vector<double> pos_curr_;  // Current position of the agent [x, y, z]
    std::vector<double> dimension_global_grid_; // Dimensions of the global voxel grid [x, y, z]
    std::vector<double> voxel_grid_range_;      // Range of the voxel grid around the agent [x, y, z]
    std::vector<double> origin_global_grid_;    // Origin coordinates of the global voxel grid [x, y, z]

    // variables to publish the environment pointcloud at constant frequency
    ::rclcpp::TimerBase::SharedPtr global_vg_timer_;
    ::rclcpp::TimerBase::SharedPtr  gpc_pub_timer_;
    ::rclcpp::Publisher<::sensor_msgs::msg::PointCloud2>::SharedPtr gpc_pub_;

    ::rclcpp::Publisher<::std_msgs::msg::Float32>::SharedPtr reward_pub_;
    
    ::std::shared_ptr<::sensor_msgs::msg::PointCloud2> gvg_pc_msg_;
    double publish_period_;
    // member variables
    ::voxel_grid_util::VoxelGrid::Ptr global_vg_shared_ptr_; // Current voxel grid being processed for the agent(t)
    ::voxel_grid_util::VoxelGrid::Ptr global_vg_last_shared_ptr_; // Last voxel grid being processed for the agent(t-1)
    ::std::vector<double> merge_comp_time_;
    ::std::vector<double> tot_comp_time_;
    bool free_grid_;                // Indicates if the grid is free space (`true`) or occupied space (`false`)
    
    double reward;
    double reward_total_;// reward for the global voxel grid;

};

// Utility Functions
::voxel_grid_util::VoxelGrid ConvertVGMsgToVGUtil(env_builder_msgs::msg::VoxelGrid &vg_msg); // Convert VoxelGrid message to utility format
env_builder_msgs::msg::VoxelGrid ConvertVGUtilToVGMsg(::voxel_grid_util::VoxelGrid &vg); // Convert utility format to VoxelGrid message

} // namespace global_map

#endif // GLOBAL_MAP_BUILDER_HPP
