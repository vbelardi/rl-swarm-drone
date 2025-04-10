#include "global_map_builder.hpp"

namespace global_map {
GlobalMapBuilder::GlobalMapBuilder() : ::rclcpp::Node("global_map_builder") {
  // declare environment parameters
  DeclareRosParameters();

  // initialize parameters
  InitializeRosParameters();
  CreateEmptyGlobalMap();
  // set up a callback to execute code on shutdown
  on_shutdown(::std::bind(&GlobalMapBuilder::OnShutdown, this));

  // create pointcloud publisher and publish at constant frequency
  gpc_pub_ = create_publisher<::sensor_msgs::msg::PointCloud2>(
      "~/" + global_pc_topic_, 10);
  gpc_pub_timer_ = create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&GlobalMapBuilder::TimerCallbackEnvironmentPC, this));

  //create global grid publisher
  ::std::string global_vg_pub_topic = "global_voxel_grid";
  global_vg_pub_ = create_publisher<::env_builder_msgs::msg::VoxelGridStamped>(
      "~/" + global_vg_pub_topic, 10);
  global_vg_timer_ = create_wall_timer(
      std::chrono::milliseconds(int(publish_period_ * 1e3)),
      std::bind(&GlobalMapBuilder::TimerCallbackEnvironmentVG, this));
  
  
  reward_pub_ = create_publisher<std_msgs::msg::Float32>("~/map_reward", 10);

  //create local grid subscriber
  for(int i = 0; i < n_rob_ ; i++){
    ::std::string local_grid_topic_ = "agent_" + ::std::to_string(i) + "/voxel_grid";
    // Create the subscription and store it in the vector
    local_grid_subs_.push_back(
        create_subscription<::env_builder_msgs::msg::VoxelGridStamped>(
            local_grid_topic_, 10,
            std::bind(&GlobalMapBuilder::LocalVoxelGridCallback, this, std::placeholders::_1)
        )
    );
  }

  reward_total_ = 0.0;
}

void GlobalMapBuilder::DeclareRosParameters() {
  declare_parameter("env_vg_topic", "/env_builder_node/environment_voxel_grid");
  declare_parameter("local_grid_topic","agent_0/voxel_grid");
  declare_parameter("dimension_global_grid",::std::vector<double>(3, 200.0));
  declare_parameter("voxel_grid_range", ::std::vector<double>(3, 10.0));
  declare_parameter("world_frame", "world");
  declare_parameter("agent_frame","agent");
  declare_parameter("free_grid", true);
  declare_parameter("id", 0);
  declare_parameter("n_rob",0);
  declare_parameter("origin_global_grid",::std::vector<double>(3, 0.0));
  declare_parameter("publish_period",0.2);
  declare_parameter("voxel_size", 0.3);
  declare_parameter("global_pc_topic", "global_pointcloud");

}

void GlobalMapBuilder::InitializeRosParameters() {

  env_vg_topic_ = get_parameter("env_vg_topic").as_string();
  local_grid_topic_ = get_parameter("local_grid_topic").as_string();
  dimension_global_grid_ = get_parameter("dimension_global_grid").as_double_array();
  voxel_grid_range_ = get_parameter("voxel_grid_range").as_double_array();
  world_frame_ = get_parameter("world_frame").as_string();
  agent_frame_ = get_parameter("agent_frame").as_string();
  free_grid_ = get_parameter("free_grid").as_bool();
  id_ = get_parameter("id").as_int();
  n_rob_ = get_parameter("n_rob").as_int();
  origin_global_grid_ = get_parameter("origin_global_grid").as_double_array();
  publish_period_ = get_parameter("publish_period").as_double();
  voxel_size_ = get_parameter("voxel_size").as_double();
  global_pc_topic_ = get_parameter("global_pc_topic").as_string();
}
void GlobalMapBuilder::LocalVoxelGridCallback(const ::env_builder_msgs::msg::VoxelGridStamped::SharedPtr vg_msg_stamped){
  // start global timer
  auto t_start_wall_global = ::std::chrono::high_resolution_clock::now();
  // get voxel size
  //double voxel_size = vg_msg_stamped->voxel_grid.voxel_size;
  // find the origin of the local grid
  //::std::array<double, 3> local_origin = vg_msg_stamped->voxel_grid.origin;
  // if haven't created the global map, then create a empty initial one.
  // if (global_vg_shared_ptr_) {
  //   auto data = global_vg_shared_ptr_->GetData();
  // } else {
  //     std::cerr << "Error: global_vg_shared_ptr_ is not initialized." << std::endl;
  // }
  ::voxel_grid_util::VoxelGrid local_vg = ConvertVGMsgToVGUtil(vg_msg_stamped->voxel_grid);
  auto t_start_wall = ::std::chrono::high_resolution_clock::now();
  //copy the current gvg before update
  StoreVoxelGrids(*global_vg_shared_ptr_, *global_vg_last_shared_ptr_);
  MergeVoxelGrids(*global_vg_shared_ptr_, local_vg);
  reward = ComputeRewards(*global_vg_shared_ptr_, *global_vg_last_shared_ptr_);
  reward_total_ += reward;
  ::std::cout<<"reward_total_:"<<reward_total_<<std::endl;
  std_msgs::msg::Float32 reward_msg;
  reward_msg.data = reward;
  reward_pub_->publish(reward_msg);
  auto t_end_wall = ::std::chrono::high_resolution_clock::now();
  double merging_time_wall_ms =
    ::std::chrono::duration_cast<::std::chrono::nanoseconds>(t_end_wall -
                                                              t_start_wall)
        .count();
  merging_time_wall_ms *= 1e-6;
  // save wall computation time
  merge_comp_time_.push_back(merging_time_wall_ms);
  CreateEnvironmentPointCloud();
  // auto data = global_vg_shared_ptr_->GetData();
  // if (!data.empty()){
  //   int known = std::count_if(data.begin(), data.end(), [](int value) {
  //       return value != -1;
  //   });
  // ::std::cout << "known voxel size" << known << ::std::endl;
  // }

  auto t_end_wall_global = ::std::chrono::high_resolution_clock::now();
  double tot_time_wall_ms =
        ::std::chrono::duration_cast<::std::chrono::nanoseconds>(
            t_end_wall_global - t_start_wall_global)
            .count();
  // convert from nano to milliseconds
  tot_time_wall_ms *= 1e-6;
  // save wall computation time
  tot_comp_time_.push_back(tot_time_wall_ms);
}
void GlobalMapBuilder::CreateEmptyGlobalMap(){
  //build an empty global voxel grid
  ::Eigen::Vector3d origin_tmp(origin_global_grid_.data());
  // origin << origin_global_grid_[0], origin_global_grid_[1], origin_global_grid_[2]; 
  ::Eigen::Vector3d dimension_tmp(dimension_global_grid_.data());
  //dim <<  round(dimension_global_grid_[0]/ voxel_size), round(dimension_global_grid_[1]/ voxel_size), round(dimension_global_grid_[2]/ voxel_size); 
  global_vg_shared_ptr_  = ::std::make_shared<::voxel_grid_util::VoxelGrid>(
      origin_tmp, dimension_tmp, voxel_size_, false);
  
  global_vg_last_shared_ptr_  = ::std::make_shared<::voxel_grid_util::VoxelGrid>(
      origin_tmp, dimension_tmp, voxel_size_, false);
}

void GlobalMapBuilder::CreateEnvironmentPointCloud() {
  ::Eigen::Vector3d origin_vg = global_vg_shared_ptr_->GetOrigin();
  ::Eigen::Vector3i dim_vg = global_vg_shared_ptr_->GetDim();
  double vox_size = global_vg_shared_ptr_->GetVoxSize();


  ::pcl::PointCloud<::pcl::PointXYZ> cloud_env;

  // add obstacles points to point cloud
  for (int i = 0; i < dim_vg(0); i++) {
    for (int j = 0; j < dim_vg(1); j++) {
      for (int k = 0; k < dim_vg(2); k++) {
        if (global_vg_shared_ptr_->IsOccupied(Eigen::Vector3i(i, j, k))) {
          ::pcl::PointXYZ pt;
          pt.x = i * vox_size + vox_size / 2 + origin_vg[0];
          pt.y = j * vox_size + vox_size / 2 + origin_vg[1];
          pt.z = k * vox_size + vox_size / 2 + origin_vg[2];
          cloud_env.points.push_back(pt);
        }
      }
    }
  }
  // create pc message
  gvg_pc_msg_ = ::std::make_shared<::sensor_msgs::msg::PointCloud2>();
  ::pcl::toROSMsg(cloud_env, *gvg_pc_msg_);
  gvg_pc_msg_->header.frame_id = world_frame_;
}

void GlobalMapBuilder::TimerCallbackEnvironmentVG() {
  // voxel grid to publish
  ::env_builder_msgs::msg::VoxelGrid vg_final_msg =
        ConvertVGUtilToVGMsg(*global_vg_shared_ptr_);
  ::env_builder_msgs::msg::VoxelGridStamped vg_final_msg_stamped;
  vg_final_msg_stamped.voxel_grid = vg_final_msg;
  vg_final_msg_stamped.voxel_grid.voxel_size = voxel_size_;
  vg_final_msg_stamped.header.stamp = now();
  vg_final_msg_stamped.header.frame_id = world_frame_;
  global_vg_pub_->publish(vg_final_msg_stamped);
}
void GlobalMapBuilder::TimerCallbackEnvironmentPC() {
  CreateEnvironmentPointCloud();
  gvg_pc_msg_->header.stamp = now();
  gpc_pub_->publish(*gvg_pc_msg_);
}
void GlobalMapBuilder::StoreVoxelGrids(::voxel_grid_util::VoxelGrid &vg_global, ::voxel_grid_util::VoxelGrid &vg_last_global){
  //compute how many unknown cells exist
  ::Eigen::Vector3i dim = vg_global.GetDim();
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
        // the voxels of the new voxel grid stay the same unless they are
        // unknown; in that cast we replace them with the values seen in the old
        // voxel grid
        ::Eigen::Vector3i coord(i, j, k);
        vg_last_global.SetVoxelInt(coord, vg_global.GetVoxelInt(coord));
      }
    }
  }
}

void GlobalMapBuilder::MergeVoxelGrids(
  ::voxel_grid_util::VoxelGrid &vg_global,
  const ::voxel_grid_util::VoxelGrid &vg_local){
  double voxel_size = vg_global.GetVoxSize();
  ::Eigen::Vector3i dim = vg_local.GetDim();
  // std::cout << dim << std::endl;
  ::Eigen::Vector3d offset_double = (vg_local.GetOrigin() - vg_global.GetOrigin());
  //::std::cout << "vg_final origin:" << vg_final.GetOrigin().transpose() << ::std::endl; 
  // ::std::cout << "vg_local origin:" << vg_local.GetOrigin().transpose() << ::std::endl;
  ::Eigen::Vector3i offset_int;
  offset_int[0] = round(offset_double[0] / voxel_size);
  offset_int[1] = round(offset_double[1] / voxel_size);
  offset_int[2] = round(offset_double[2] / voxel_size);
  /* ::std::cout << "offset_int: " << offset_int.transpose() << ::std::endl; */
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
        // the voxels of the new voxel grid stay the same unless they are
        // unknown; in that cast we replace them with the values seen in the old
        // voxel grid
        ::Eigen::Vector3i coord(i, j, k);
        ::Eigen::Vector3i coord_final = coord + offset_int;
        if (vg_global.IsUnknown(coord_final)) {
          int8_t vox_value = vg_local.GetVoxelInt(coord);
          vg_global.SetVoxelInt(coord_final, vox_value);
        }
      }
    }
  }
}

double GlobalMapBuilder::ComputeRewards(::voxel_grid_util::VoxelGrid &vg_global, ::voxel_grid_util::VoxelGrid &vg_last_global){
  double completeness = ComputeMapCompleteness(vg_global);
  double reward = 0;
  if(completeness > 0.999){
    reward = 100.0;//if complete
  }
  else{
    reward = (completeness - ComputeMapCompleteness(vg_last_global));
    // ::std::cout<<"reward:"<<reward<<std::endl;
  }
  return reward;
}

double GlobalMapBuilder::ComputeMapCompleteness(::voxel_grid_util::VoxelGrid &vg_global){
  //compute how many unknown cells exist
  ::Eigen::Vector3i dim = vg_global.GetDim();
  double completeness = 0.0;
  double unknown = 0.0;// unknown cells
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
        // the voxels of the new voxel grid stay the same unless they are
        // unknown; in that cast we replace them with the values seen in the old
        // voxel grid
        ::Eigen::Vector3i coord(i, j, k);
        if (vg_global.IsUnknown(coord)) {
          unknown += 1;
        }
      }
    }
  }
  completeness = (vg_global.GetDataSize() - unknown)/vg_global.GetDataSize();
  return completeness;
}

void GlobalMapBuilder::DisplayCompTime(::std::vector<double> &comp_time) {
  double max_t = 0;
  double min_t = 1e10;
  double sum_t = 0;
  double std_dev_t = 0;
  for (int i = 0; i < int(comp_time.size()); i++) {
    if (comp_time[i] > max_t) {
      max_t = comp_time[i];
    }
    if (comp_time[i] < min_t) {
      min_t = comp_time[i];
    }
    sum_t = sum_t + comp_time[i];
  }
  double mean_t = sum_t / comp_time.size();

  for (int i = 0; i < int(comp_time.size()); i++) {
    std_dev_t += (comp_time[i] - mean_t) * (comp_time[i] - mean_t);
  }
  std_dev_t = std_dev_t / comp_time.size();
  std_dev_t = sqrt(std_dev_t);

  ::std::cout << ::std::endl << "mean: " << mean_t;
  ::std::cout << ::std::endl << "std_dev: " << std_dev_t;
  ::std::cout << ::std::endl << "max: " << max_t;
  ::std::cout << ::std::endl << "min: " << min_t << ::std::endl;
}

void GlobalMapBuilder::OnShutdown() {
  ::std::cout << ::std::endl << "merge: ";
  DisplayCompTime(merge_comp_time_);
  ::std::cout << ::std::endl << "total: ";
  DisplayCompTime(tot_comp_time_);
}
::voxel_grid_util::VoxelGrid
ConvertVGMsgToVGUtil(::env_builder_msgs::msg::VoxelGrid &vg_msg) {
  // get the origin
  ::Eigen::Vector3d origin(vg_msg.origin[0], vg_msg.origin[1],
                           vg_msg.origin[2]);

  // get the dimensions
  ::Eigen::Vector3i dimension(vg_msg.dimension[0], vg_msg.dimension[1],
                              vg_msg.dimension[2]);

  // create voxel grid object
  ::voxel_grid_util::VoxelGrid vg(origin, dimension, vg_msg.voxel_size,
                                  vg_msg.data);

  // return the voxel grid object
  return vg;
}

::env_builder_msgs::msg::VoxelGrid
ConvertVGUtilToVGMsg(::voxel_grid_util::VoxelGrid &vg) {
  ::env_builder_msgs::msg::VoxelGrid vg_msg;

  // get the origin
  ::Eigen::Vector3d origin = vg.GetOrigin();
  vg_msg.origin[0] = origin[0];
  vg_msg.origin[1] = origin[1];
  vg_msg.origin[2] = origin[2];

  // get dimensions
  ::Eigen::Vector3i dim = vg.GetDim();
  vg_msg.dimension[0] = dim[0];
  vg_msg.dimension[1] = dim[1];
  vg_msg.dimension[2] = dim[2];

  // get voxel size
  vg_msg.voxel_size = vg.GetVoxSize();

  // get data
  vg_msg.data = vg.GetData();

  // return voxel grid message
  return vg_msg;
}
}// namespace global_map
