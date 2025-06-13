import numpy as np
from shapes import VoxelGrid, Wall, Cylinder, Loop


def write_config_file(filename, voxel_grid, seed):
    """
    Compute occupancy and dump a YAML config under ../config/<filename>.yaml
    """
    voxel_grid.compute_occupancy()
    config_yaml = """env_builder_node: 
  ros__parameters:
    origin_grid: %s # origin of the voxel grid
    dimension_grid: %s # dimensions in meters
    vox_size: %f # voxel size of the grid
    free_grid: true # if true the grid is initialized to free instead of unknown
  
    multi_obst_size: false # if false, use size_obst as a common size for all obstacles; otherwise use size_obst_multi to define the size of each obstacle
    multi_obst_position: true # if false, generate positions randomly using normal distribution, if true use position_obst_multi to define the position of each obstacle
    range_obst: [15.0, 15.0, 0.0] # area on the map where to generate obstacles, always positive numbers to move it around, use the origin_obst variable
    origin_obst: [2.0, 2.0, 0.0] # origin of the area where we generate the obstacles
    size_obst: %s # height x width x length
    n_obst: 50 # placeholder (not used if position_obst_vec is defined)
    rand_seed: %i # seed for random obstacles
    size_obst_vec: %s # x,y,z size of each obstacle concatenated
    position_obst_vec: %s # x,y,z position of each obstacle concatenated


    env_pc_topic: "environment_poincloud" # topic on which to publish the env pointcloud
    env_vg_topic: "environment_voxel_grid" # topic on which to publish the env voxel grid 
    env_pc_frame: "world" # origin frame of the published poinctloud
    get_grid_service_name: "get_voxel_grid" # name of the service to get the voxel grid of each agent

    save_obstacles: true # if true, save the obstacle position in pos_obs.csv as well as the pointcloud in a file""" % (
        voxel_grid.origin,
        voxel_grid.dimension,
        voxel_grid.voxel_size,
        [0.01, 0.01, 0.01],            # dummy size_obst
        seed,
        [0.01, 0.01, 0.01],            # dummy size_obst_vec
        voxel_grid.occupied_voxels
    )

    config_filename = '../config/%s.yaml' % filename
    with open(config_filename, 'w') as yaml_file:
        yaml_file.write(config_yaml)
    print(f"Configuration saved to '{config_filename}'")



def build_map_walls(seed=1):
    """
    Map 1: Three large walls in different orientations (with small gaps for minimal passage).
    """
    dimension = [20.0, 20.0, 6.0]
    voxel_size = 0.3
    origin = [0.0, 0.0, 0.0]
    vg = VoxelGrid(dimension, voxel_size, origin)

    # Wall 1: horizontal plane at y=5, spanning x=0..20, thickness=0.3 m.
    # Add a 1 m-wide gap centered at x=10.
    wall1 = Wall(
        origin=(15.0, 15.0, 0.0),       # origin (x, y, z)
        direction1=(0.0, -1.0, 0.0),       # direction1 (along +x)
        direction2=(0.0, 0.0, 1.0),       # direction2 (vertical)
        width=0.3                    # thickness
    )
    vg.add_shape(wall1)

    # Wall 2: vertical plane at x=8, spanning y=5..20, thickness=0.3 m.
    # We'll place origin at (8, 0) so it spans entire y; add gap at y=12.
    wall2 = Wall(
        origin=(8.0, 10.0, 0.0),       # origin (x, y, z)
        direction1=(0.0, 1.0, 0.0),       # direction1 (along +y)
        direction2=(0.0, 0.0, 1.0),       # direction2 (vertical)
        width=0.3                    # thickness
    )
    vg.add_shape(wall2)

    # Wall 3: diagonal plane from bottom-left toward top-right.
    # Use direction1=(1,1,0), origin at (0, 10) so it crosses interior.
    # Add a small 1 m gap near the center of that diagonal.
    wall3 = Wall(
        origin=(9.0, 5.0, 0.0),      # origin (x, y, z)
        direction1=(-1.0, -1.0, 0.0),       # direction1 (45° in XY)
        direction2=(0.0, 0.0, 1.0),       # direction2 (vertical)
        width=0.3                    # thickness
    )

    vg.add_shape(wall3)

    wall4 = Wall(
        origin=(5.0, 12.0, 0.0),      # origin (x, y, z)
        direction1=(-1.0, 0.0, 0.0),       # direction1 (45° in XY)
        direction2=(0.0, 0.0, 1.0),       # direction2 (vertical)
        width=0.3                    # thickness
    )

    vg.add_shape(wall4)

    wall5 = Wall(
        origin=(18.0, 6.0, 0.0),      # origin (x, y, z)
        direction1=(1.0, 0.0, 0.0),       # direction1 (45° in XY)
        direction2=(0.0, 0.0, 1.0),       # direction2 (vertical)
        width=0.3                    # thickness
    )

    vg.add_shape(wall5)

    write_config_file("env_map_walls", vg, seed)

def build_map_loops(seed=2):
    """
    Map 2: Five loops, each with an inner radius of 1.0 m and outer radius of 1.5 m.
    No walls or cylinders here.
    """
    dimension = [20.0, 20.0, 6.0]
    voxel_size = 0.3
    origin = [0.0, 0.0, 0.0]
    vg = VoxelGrid(dimension, voxel_size, origin)

    # Define five loop centers + orientations:
    # Note: Loop((x, y, z), yaw, inner_radius, outer_radius).
    # Placing each loop “on the ground” (z=0.0).
    loop_params = [
        ((5.0,  5.0,  3.0),   0.0),         # bottom-left quadrant
        ((5.0, 15.0,  3.0),   np.pi/2),     # top-left quadrant
        ((10.0, 10.0, 3.0),   np.pi/4),     # center
        ((15.0,  5.0,  3.0),   np.pi/3),     # bottom-right quadrant
        ((15.0, 15.0, 3.0),   np.pi/6),     # top-right quadrant
    ]
    for (cx, cy, cz), yaw in loop_params:
        # Correct Loop constructor:
        loop = Loop((cx, cy, cz), yaw, 1.5, 2.5)
        vg.add_shape(loop)

    write_config_file("env_map_loops", vg, seed)



def build_map_mixed(seed=3):
    """
    Map 3: Reduced mixed map with:
      - Three smaller walls
      - Three loops
      - Two cylinders
    """
    dimension = [20.0, 20.0, 6.0]
    voxel_size = 0.3
    origin = [0.0, 0.0, 0.0]
    vg = VoxelGrid(dimension, voxel_size, origin)

    # --- WALLS (reuse the three from build_map_walls) ---
    wall1 = Wall(
        origin=(15.0, 5.0, 0.0),
        direction1=(1.0, 0.0, 0.0),
        direction2=(0.0, 0.0, 1.0),
        width=0.3
    )
    vg.add_shape(wall1)

    wall2 = Wall(
        origin=(10.0, 15.0, 0.0),
        direction1=(0.0, 1.0, 0.0),
        direction2=(0.0, 0.0, 1.0),
        width=0.3
    )
    vg.add_shape(wall2)

    wall3 = Wall(
        origin=(14.0, 17.0, 0.0),
        direction1=(1.0, 1.0, 0.0),
        direction2=(0.0, 0.0, 1.0),
        width=0.3
    )
    vg.add_shape(wall3)

    # --- LOOPS (reuse the three from build_map_loops) ---
    loop_params = [
        ((6.0,  6.0,  3.0),   0.0),
        ((6.0, 14.0,  3.0),   np.pi/2),
        ((14.0, 10.0, 3.0),   np.pi/4),
    ]
    for (cx, cy, cz), yaw in loop_params:
        loop = Loop((cx, cy, cz), yaw, 1.0, 2.0)
        vg.add_shape(loop)

    # --- CYLINDERS (two total) ---
    # Cylinder((x, y, z), (dx, dy, dz), radius). Height=3.0 → axis=(0.0, 0.0, 3.0).
    cylinder_positions = [
        (8.0,  12.0, 0.0),   # upper-left region
        (12.0, 6.0,  0.0),   # lower-right region
    ]
    for (cx, cy, cz) in cylinder_positions:
        cyl = Cylinder((cx, cy, cz), (0.0, 0.0, 3.0), 0.5)
        vg.add_shape(cyl)

    write_config_file("env_map_mixed", vg, seed)

if __name__ == "__main__":
    print("Start generation of obstacles ...")
    #build_map_walls(seed=1)
    #build_map_loops(seed=2)
    build_map_mixed(seed=3)
    print("Done.")
