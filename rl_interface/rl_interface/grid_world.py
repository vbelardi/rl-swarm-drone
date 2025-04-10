'''
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped
from env_builder_msgs.msg import VoxelGridStamped
import nav_msgs.msg
import time
from stable_baselines3 import PPO

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


voxel_size = 0.3



# ------------------------- ROS2 Node -------------------------
class RLSubscriber(Node):
    def __init__(self):
        super().__init__('rl_subscriber')
        self.declare_parameter("n_rob", 10)
        self.n_rob_ = 1  # For now, using a single agent
        self.gvg_dim = [0, 0, 0]  # Dimensions of the full-resolution voxel grid
        self.gvg = None         # Will store the full-resolution voxel grid as a NumPy array
        self.goal_pub_ = {}
        self.pos_sub_ = {}
        self.agent_goal_curr_ = [[] for _ in range(self.n_rob_)]
        # For each agent, keep the latest position (x, y, z)
        self.agent_pos_curr_ = [[1.0, 1.0, 1.0] for _ in range(self.n_rob_)]
        self.time_goal_publish = 2.0  # seconds between goal updates
        #self.reward = 0.0

        # Set up publishers and subscribers for each agent
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

        # Subscribe to the voxel grid topic
        vg_sub_name = "global_map_builder_node/global_voxel_grid"
        self.create_subscription(VoxelGridStamped, vg_sub_name, self.global_vg_callback, 10)

        #self.sub_reward = self.create_subscription(Float32, 'global_map_builder_node/map_reward', self.reward_callback, 10)

        # Timer to periodically compute and publish goals
        self.timer_ = self.create_timer(self.time_goal_publish, self.timer_callback)

        self.actor_model = PPO.load("ppo_drone_exploration_model", device=device)

    #def reward_callback(self, msg):
        #self.reward = msg.data

    def global_vg_callback(self, msg):
        """
        Receives the voxel grid message.
        Assumes:
          - msg.voxel_grid.dimension is a list of 3 integers, e.g. [X, Y, Z]
          - msg.voxel_grid.data is a flat list of voxel values.
        """
        self.gvg_dim = msg.voxel_grid.dimension
        # Set voxel values greater than 0 to 100
        full = np.array(msg.voxel_grid.data).reshape(self.gvg_dim)
        new_grid = np.where( full == -1, 0, np.where(full == 0, 1, 2))
        #full_grid = np.array(msg.voxel_grid.data).reshape(self.gvg_dim)
        # Reshape the voxel grid data to match the dimensions provided in the message
        self.gvg = np.array(new_grid, dtype=np.uint8)


    def agent_traj_callback(self, i, msg):
        if msg.poses:
            position = msg.poses[0].pose.position
            self.agent_pos_curr_[i] = [position.x, position.y, position.z]


    def direction_to_goal_point(self, drone_position, direction_vector, voxel_grid_origin, voxel_grid_dims):
        """
        Converts a normalized direction vector into a target point at the boundary of the voxel grid.
        
        Parameters:
        - drone_position: np.array of shape (3,) representing the current position.
        - direction_vector: np.array of shape (3,) representing the desired direction.
        - voxel_grid_origin: The starting coordinate (np.array of shape (3,)).
        - voxel_grid_dims: real dimensions in meters (np.array of shape (3,)).
        - voxel_size: The size of each voxel.
        
        Returns:
        - goal_point: np.array of shape (3,) representing the point at the limit of the grid in that direction.
        """
        # Normalize the direction vector to ensure it has unit length.
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return drone_position  # No movement if the direction is zero.
        direction_unit = direction_vector / norm

        # Calculate the physical boundaries of the voxel grid.
        grid_min = np.array(voxel_grid_origin)
        grid_max = np.array(voxel_grid_origin) + np.array(voxel_grid_dims)

        # For each axis, compute how far the drone can travel until it reaches the boundary.
        travel_distances = []
        for i in range(3):
            if direction_unit[i] > 0:
                travel_distance = (grid_max[i] - drone_position[i]) / direction_unit[i]
            elif direction_unit[i] < 0:
                travel_distance = (grid_min[i] - drone_position[i]) / direction_unit[i]
            else:
                travel_distance = np.inf  # No movement along this axis.
            travel_distances.append(travel_distance)

        # The maximum distance before hitting a boundary is the minimum across axes.
        min_travel = min(travel_distances)
        goal_point = drone_position + direction_unit * min_travel
        goal_point = np.clip(goal_point, [0.1, 0.1, 0.1], [4.9, 4.9, 4.9])
        return goal_point

    def timer_callback(self):
        # Ensure we have received a voxel grid before proceeding.
        if self.gvg is None:
            self.get_logger().warn("Global voxel grid not received yet.")
            return
        
        print("unique values in gvg: ", np.unique(self.gvg))
        obs = {"observation": self.gvg, "drone_positions": np.array(self.agent_pos_curr_[0], dtype=np.float32).reshape(self.n_rob_* 3,)}
        actions, _ = self.actor_model.predict(obs, deterministic=True)
        
        actions = np.array(actions, dtype=np.float32).reshape(self.n_rob_, 3)
        # Compute goal points for each drone based on its current position and the direction vector.
        goal_points = []
        for i, action in enumerate(actions):
            current_position = self.agent_pos_curr_[0]
            print("current position: ", current_position)
            print("action: ", action)
            goal_point = self.direction_to_goal_point(current_position, action, [0.0, 0.0, 0.0], [5.0, 5.0, 2.0])
            print("goal point: ", goal_point)
            goal_points.append(goal_point)
        goal_points = np.array(goal_points)
        goal = goal_points[0]
        # Publish the goal for the agent.
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
'''



import gymnasium as gym
import torch
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



class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        """
        Custom extractor for an observation dictionary with:
          - "observation": a 3D voxel grid of shape (D, H, W).
          - "drone_positions": a low-dimensional vector.
        
        The pipeline:
          1. Convert voxel grid into 3-channel one-hot encoding.
          2. Apply a 3D CNN on the voxel grid.
          3. Concatenate with drone position vector.
          4. Fuse with an MLP.
        """
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        voxel_shape = observation_space.spaces["observation"].shape[:3]  # (D, H, W)

        # 3-channel CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),  # Input channels = 3
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten()
        )

        # Compute CNN output dimension
        dummy_voxel = torch.zeros(1, 3, *voxel_shape)  # 3 channels now
        cnn_output_dim = self.cnn3d(dummy_voxel).shape[1]

        fusion_input_dim = cnn_output_dim + 3  # +3 for drone position vector
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        voxel = observations["observation"]  # (batch, D, H, W)

        # One-hot encoding to 3 channels
        channel_unknown = (voxel == 0).unsqueeze(1).float()  # (batch, 1, D, H, W)
        channel_free    = (voxel == 1).unsqueeze(1).float()
        channel_obstacle= (voxel == 2).unsqueeze(1).float()
        voxel_input = torch.cat([channel_unknown, channel_free, channel_obstacle], dim=1)

        cnn_features = self.cnn3d(voxel_input)

        # Concatenate with drone position vector
        fused = torch.cat([cnn_features, observations["drone_positions"]], dim=1)
        features = self.fusion_mlp(fused)
        return features


# ------------------------- Dummy Env for Model Loading -------------------------
dummy_obs_space = Dict({
    "observation": Box(low=0, high=2, shape=(17, 17, 17), dtype=np.uint8),
    "drone_positions": Box(low=0, high=1, shape=(3,), dtype=np.float32),
})

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

# ------------------------- ROS2 Node -------------------------
class RLSubscriber(Node):
    def __init__(self):
        super().__init__('rl_subscriber')
        self.n_rob_ = 1
        self.gvg_dim = [0, 0, 0]
        self.gvg = None
        self.goal_pub_ = {}
        self.pos_sub_ = {}
        self.agent_goal_curr_ = [[] for _ in range(self.n_rob_)]
        self.agent_pos_curr_ = [[1.0, 1.0, 1.0] for _ in range(self.n_rob_)]
        self.time_goal_publish = 2.0

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
            "ppo_drone_exploration_model",
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
        goal_point = np.clip(goal_point, [0.1, 0.1, 0.1], [4.9, 4.9, 4.9])
        return goal_point

    def timer_callback(self):
        if self.gvg is None:
            self.get_logger().warn("Global voxel grid not received yet.")
            return

        obs = {
            "observation": self.gvg,
            "drone_positions": np.array(self.agent_pos_curr_[0]/np.array([5.0, 5.0, 5.0]), dtype=np.float32).reshape(self.n_rob_* 3,),
        }

        actions, _ = self.actor_model.predict(obs, deterministic=True)
        actions = np.array(actions, dtype=np.float32).reshape(self.n_rob_, 3)

        goal_points = []
        for i, action in enumerate(actions):
            current_position = self.agent_pos_curr_[0]
            goal_point = self.direction_to_goal_point(current_position, action, [0.0, 0.0, 0.0], [5.0, 5.0, 5.0])
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
