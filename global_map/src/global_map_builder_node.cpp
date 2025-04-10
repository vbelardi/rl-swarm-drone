#include "global_map_builder.hpp"

int main(int argc, char *argv[]){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<global_map::GlobalMapBuilder>());
  rclcpp::shutdown();
}