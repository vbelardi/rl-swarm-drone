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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box, Dict

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

voxel_size = 0.3
grid_real_dim = [20.0, 20.0, 6.0]  # meters
grid_dim = [67, 67, 20]  # number of voxels
grid_origin = [0.0, 0.0, 0.0]  # meters
num_drones = 1


def downsample_voxel_grid_onehot(voxel_input, output_size):
    """
    Downsamples a one-hot encoded voxel grid (B, 3, D, H, W) using nearest neighbor,
    preserving one-hot exclusivity.
    """
    # Convert one-hot to label
    labels = torch.argmax(voxel_input, dim=1)  # (B, D, H, W)

    # Downsample using nearest neighbor
    labels = labels.unsqueeze(1).float()
    labels_downsampled = torch.nn.functional.interpolate(labels, size=output_size, mode='nearest')
    labels_downsampled = labels_downsampled.squeeze(1).long()

    # Convert back to one-hot
    voxel_downsampled = torch.nn.functional.one_hot(labels_downsampled, num_classes=3)
    voxel_downsampled = voxel_downsampled.permute(0, 4, 1, 2, 3).float()

    return voxel_downsampled

def downsample_voxel_grid_priority(voxel_input, output_size):
    """
    Downsamples a one-hot encoded voxel grid (B, 3, D, H, W) with priority: obstacle > unknown > free.
    Uses adaptive max-pooling per channel to preserve priority blocks.
    
    Args:
      voxel_input: Tensor of shape (B,3,D,H,W), one-hot encoding of {unknown,free,obstacle}.
      output_size: tuple (d2, h2, w2) target grid size.
    Returns:
      voxel_down: Tensor of shape (B,3,d2,h2,w2), one-hot with priority applied.
    """
    # 1) Extract per-class masks
    unk_mask  = voxel_input[:, 0:1]  # (B,1,D,H,W)
    free_mask = voxel_input[:, 1:2]
    obs_mask  = voxel_input[:, 2:3]
    
    # 2) Adaptive max-pool each mask to output_size
    unk_ds  = F.adaptive_max_pool3d(unk_mask,  output_size).squeeze(1)  # (B, D2,H2,W2)
    free_ds = F.adaptive_max_pool3d(free_mask, output_size).squeeze(1)
    obs_ds  = F.adaptive_max_pool3d(obs_mask,  output_size).squeeze(1)
    
    # 3) Build label grid with priority
    # default = free (1)
    labels = torch.ones_like(obs_ds, dtype=torch.long)
    # override unknown (0) where unk_ds>0
    labels[unk_ds > 0] = 0
    # override obstacle (2) where obs_ds>0  (highest priority)
    labels[obs_ds > 0] = 2
    
    # 4) Convert back to one-hot
    voxel_down = F.one_hot(labels, num_classes=3)        # (B, D2,H2,W2, 3)
    voxel_down = voxel_down.permute(0, 4, 1, 2, 3).float()  # (B,3,D2,H2,W2)
    return voxel_down

'''
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        """
        Custom extractor for observation dict:
          - "observation": a 3D voxel grid of shape (D, H, W)
          - "drone_positions": a low-dimensional vector
        """
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        self.original_voxel_shape = observation_space.spaces["observation"].shape[:3]
        self.downsampled_shape = (40, 40, 12)

        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten()
        )

        dummy_voxel = torch.zeros(1, 3, *self.downsampled_shape)
        cnn_output_dim = self.cnn3d(dummy_voxel).shape[1]

        fusion_input_dim = cnn_output_dim + 3  # 3 for drone_positions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        voxel = observations["observation"]  # (B, D, H, W)

        # One-hot encoding
        channel_unknown  = (voxel == 0).unsqueeze(1).float()
        channel_free     = (voxel == 1).unsqueeze(1).float()
        channel_obstacle = (voxel == 2).unsqueeze(1).float()
        voxel_input = torch.cat([channel_unknown, channel_free, channel_obstacle], dim=1)  # (B, 3, D, H, W)

        # Downsample
        voxel_input_downsampled = downsample_voxel_grid_priority(voxel_input, self.downsampled_shape)

        # CNN
        cnn_features = self.cnn3d(voxel_input_downsampled)

        # Concatenate witorch drone position
        fused = torch.cat([cnn_features, observations["drone_positions"]], dim=1)
        features = self.fusion_mlp(fused)
        return features
'''
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        D,H,W = observation_space.spaces["observation"].shape
        self.downsampled_shape = (40, 40, 12)
        # CNN 3D de base
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8,16,3,padding=1),    nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1),    nn.ReLU(), nn.MaxPool3d(2),
            nn.Flatten()
        )
        # dimension après CNN
        with torch.no_grad():
            dummy = torch.zeros(1,3,*self.downsampled_shape)
            cnn_out = self.cnn3d(dummy).shape[1]
        # projection CNN → 512
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(),
            nn.Dropout(0.1), nn.LayerNorm(512)
        )
        # MLP position → 256
        pos_dim = observation_space.spaces["drone_positions"].shape[0]
        self.position_mlp = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.LayerNorm(256)
        )

        # fusion → features_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 + 256, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

        self._features_dim = features_dim

    def forward(self, obs):
        vox = obs["observation"]             # (B,D,H,W)
        unk  = (vox==0).unsqueeze(1).float()
        free = (vox==1).unsqueeze(1).float()
        obs_ = (vox==2).unsqueeze(1).float()
        x = torch.cat([unk,free,obs_], dim=1) # (B,3,D,H,W)
        x = downsample_voxel_grid_priority(x, self.downsampled_shape)
        c = self.cnn3d(x)
        c = self.cnn_proj(c)
        p = self.position_mlp(obs["drone_positions"])
        return self.fusion_mlp(torch.cat([c,p], dim=1))

# ------------------------- Dummy Env for Model Loading -------------------------
dummy_obs_space = Dict({
    "observation": Box(low=0, high=2, shape=tuple(grid_dim), dtype=np.uint8),
    "drone_positions": Box(low=0, high=1, shape=(3*num_drones,), dtype=np.float32),
})

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
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
        self.time_goal_publish = 3.0
        self.grid_real_dim = grid_real_dim
        self.grid_dim = grid_dim
        self.grid_origin = np.array(grid_origin)
        self.margin = 0.2

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

        self.actor_model = PPO.load(
            "ppo_ckpt_1000000_steps",
            custom_objects={
                "features_extractor_class": CustomCombinedExtractor,
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
        
        if len(self.agent_goal_curr_[0]) == 0 or np.linalg.norm(np.array(self.agent_pos_curr_[0]) - np.array(self.agent_goal_curr_[0])) < 0.1:
            obs = {
                "observation": self.gvg,
                "drone_positions": np.array(self.agent_pos_curr_[0]/np.array(self.grid_real_dim), dtype=np.float32).reshape(self.n_rob_* 3,),
            }

            actions, _ = self.actor_model.predict(obs, deterministic=True)
            actions = np.array(actions, dtype=np.float32).reshape(self.n_rob_, 3)

            goal_points = []
            for i, action in enumerate(actions):
                current_position = self.agent_pos_curr_[0]
                goal_point = self.direction_to_goal_point(current_position, action, self.grid_origin, self.grid_real_dim)
                goal_points.append(goal_point)
            goal_points = np.array(goal_points)
            goal = goal_points[0]

            goal_msg = PointStamped()
            goal_msg.header.frame_id = "world"
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.point.x, goal_msg.point.y, goal_msg.point.z = goal.tolist()
            self.agent_goal_curr_[0] = goal.tolist()
            self.goal_pub_[0].publish(goal_msg)
            self.get_logger().info(f"Published goal: {goal.tolist()} for agent_0")

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
