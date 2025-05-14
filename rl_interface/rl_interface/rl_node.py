import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped
from env_builder_msgs.msg import VoxelGridStamped
import nav_msgs.msg
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box, Dict
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

voxel_size = 0.3
grid_real_dim = [20.0, 20.0, 6.0]  # meters
grid_dim = [67, 67, 20]  # number of voxels
grid_origin = [0.0, 0.0, 0.0]  # meters
num_drones = 1



class Custom3DGridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        D, H, W = observation_space.spaces["observation"].shape
        drone_shape = observation_space.spaces["drone_positions"].shape
        # 3D CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(32, 32, (5,5,3), (2,2,1), (1,1,1)), nn.ReLU(),
            nn.Conv3d(32,64,3,2,1), nn.ReLU(),
            nn.Conv3d(64,64,3,2,1), nn.ReLU(),
            nn.Conv3d(64,128,3,1,1), nn.ReLU(),
            nn.Conv3d(128,128,3,1,1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            d = D//4; h = H//4; w = W//4
            dummy = torch.zeros(1,3,d,h,w)
            flat = self.cnn3d(dummy).shape[1]
        # position MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(drone_shape,32), nn.ReLU(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
        )
        # fusion
        self.fuse = nn.Sequential(
            nn.Linear(flat+64,2048), nn.ReLU(),
            nn.Linear(2048,1024), nn.ReLU(),
            nn.Linear(1024,512), nn.ReLU(),
            nn.Linear(512,features_dim), nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, obs):
        v = obs["observation"].long()
        u = (v==0).unsqueeze(1).float()
        f = (v==1).unsqueeze(1).float()
        o = (v==2).unsqueeze(1).float()
        x = torch.cat([u,f,o],dim=1)
        _, D, H, W = obs["observation"].shape
        x = F.adaptive_avg_pool3d(x, output_size=(D//4, H//4, W//4))
        c = self.cnn3d(x)
        p = self.pos_mlp(obs["drone_positions"])
        return self.fuse(torch.cat([c,p],1))



# ------------------------- Dummy Env for Model Loading -------------------------


policy_kwargs = dict(
    features_extractor_class=Custom3DGridExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

# ------------------------- ROS2 Node -------------------------
class RLSubscriber(Node):
    def __init__(self):
        super().__init__('rl_subscriber')
        self.n_rob_ = num_drones
        self.gvg_dim = [0, 0, 0]
        self.gvg = None
        self.goal_pub_ = {}
        self.pos_sub_ = {}
        self.agent_goal_curr_ = [[] for _ in range(self.n_rob_)]
        self.agent_pos_curr_ = [[1.0, 1.0, 1.0] for _ in range(self.n_rob_)]
        self.time_goal_publish = 0.1
        self.grid_real_dim = grid_real_dim
        self.grid_dim = grid_dim
        self.grid_origin = np.array(grid_origin)
        self.margin = 0.2
        self.sleep_between_obs = 3.0
        self.old_time = time.time()
        self.lstm_states = None
        self.dones = np.array([True] * n_rob_, dtype=bool)

        for i in range(self.n_rob_):
            pub_name = f"/agent_{i}/goal"
            self.goal_pub_[i] = self.create_publisher(PointStamped, pub_name, 10)
            pos_sub_name = f"/agent_{i}/traj"
            self.pos_sub_[i] = self.create_subscription(
                nav_msgs.msg.Path, 
                pos_sub_name, 
                lambda msg, i=i: self.agent_traj_callback(i, msg), 
                10
            )

        vg_sub_name = "global_map_builder_node/global_voxel_grid"
        self.create_subscription(VoxelGridStamped, vg_sub_name, self.global_vg_callback, 10)

        self.timer_ = self.create_timer(self.time_goal_publish, self.timer_callback)

        self.actor_model = RecurrentPPO.load(
            "ppo_finetune_10000000_steps",
            custom_objects={
                "features_extractor_class": Custom3DGridExtractor,
            },
            device=device,
        )


    def global_vg_callback(self, msg):
        self.gvg_dim = msg.voxel_grid.dimension
        full = np.array(msg.voxel_grid.data).reshape(self.gvg_dim)
        new_grid = np.where(full == -1, 0, np.where(full == 0, 1, 2))
        self.gvg = np.array(new_grid, dtype=np.uint8)

    def agent_traj_callback(self, i, msg):
        if msg.poses:
            position = msg.poses[0].pose.position
            self.agent_pos_curr_[i] = [position.x, position.y, position.z]

    def direction_to_goal_point(self, drone_position, direction_vector, voxel_grid_origin, voxel_grid_dims):
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return drone_position
        direction_unit = direction_vector / norm
        grid_min = np.array(voxel_grid_origin)
        grid_max = np.array(voxel_grid_origin) + np.array(voxel_grid_dims)

        travel_distances = []
        for i in range(3):
            if direction_unit[i] > 0:
                travel_distance = (grid_max[i] - drone_position[i]) / direction_unit[i]
            elif direction_unit[i] < 0:
                travel_distance = (grid_min[i] - drone_position[i]) / direction_unit[i]
            else:
                travel_distance = np.inf
            travel_distances.append(travel_distance)

        min_travel = min(travel_distances)
        goal_point = drone_position + direction_unit * min_travel
        goal_point = np.clip(goal_point, self.grid_origin + self.margin, self.grid_origin + self.grid_real_dim - self.margin)
        return goal_point

    def timer_callback(self):
        if self.gvg is None:
            self.get_logger().warn("Global voxel grid not received yet.")
            return
            
        time_callback = time.time()
        should_update = False
        
        # Check if any drone needs a new goal
        for i in range(self.n_rob_):
            if (len(self.agent_goal_curr_[i]) == 0 or 
                np.linalg.norm(np.array(self.agent_pos_curr_[i]) - np.array(self.agent_goal_curr_[i])) < 0.01):
                should_update = True
                break
        
        # Also update if we've exceeded the time between observations
        if time_callback - self.old_time > self.sleep_between_obs:
            should_update = True
        
        if should_update:
            self.old_time = time_callback
            
            # Prepare full drone positions array
            all_drone_positions = np.array([pos for pos in self.agent_pos_curr_])
            normalized_positions = all_drone_positions.flatten() / np.array(self.grid_real_dim).repeat(self.n_rob_)
            
            obs = {
                "observation": self.gvg,
                "drone_positions": normalized_positions.astype(np.float32),
            }

            actions, self.lstm_states = self.actor_model.predict(obs, state=self.lstm_states, episode_start=self.dones, deterministic=True)
            actions = np.array(actions, dtype=np.float32).reshape(self.n_rob_, 3)
            self.dones = np.array([False] * self.n_rob_, dtype=bool)
            
            # Process actions for each drone
            for i in range(self.n_rob_):
                current_position = self.agent_pos_curr_[i]
                goal_point = self.direction_to_goal_point(current_position, actions[i], self.grid_origin, self.grid_real_dim)
                
                goal_msg = PointStamped()
                goal_msg.header.frame_id = "world"
                goal_msg.header.stamp = self.get_clock().now().to_msg()
                goal_msg.point.x, goal_msg.point.y, goal_msg.point.z = goal_point.tolist()
                self.agent_goal_curr_[i] = goal_point.tolist()
                self.goal_pub_[i].publish(goal_msg)
                self.get_logger().info(f"Published goal: {goal_point.tolist()} for agent_{i}")

    def get_current_obs(self):
        """ Capture une observation actuelle sous forme de dict. """
        # Prepare full drone positions array
        all_drone_positions = np.array([pos for pos in self.agent_pos_curr_])
        normalized_positions = all_drone_positions.flatten() / np.array(self.grid_real_dim).repeat(self.n_rob_)
        
        obs = {
            "observation": self.gvg.copy(),  # Important de copier pour ne pas écraser
            "drone_positions": normalized_positions.astype(np.float32),
        }
        return obs

def main(args=None):
    rclpy.init(args=args)
    node = RLSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
