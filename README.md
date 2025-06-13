
# Guide d'installation et de configuration

Ce document détaille l'installation de Gurobi, la compilation et l'exécution des différents packages nécessaires pour le projet.

---

## 1. Installation de Gurobi

Pour installer Gurobi, suivez ces instructions :

### 1.1. Télécharger Gurobi

- Téléchargez Gurobi 10.0.* depuis le site officiel : [Gurobi Downloads](https://www.gurobi.com/downloads/).
- Suivez les instructions d'installation de Gurobi pour votre système.

### 1.2. Installer la licence Gurobi

- Rendez-vous sur le [Centre de licences Gurobi](https://www.gurobi.com/downloads/end-user-license-agreement/), créez votre licence et suivez les instructions affichées pour l'installer.

### 1.3. Compiler Gurobi et copier la bibliothèque

1. Rendez-vous dans le répertoire de build de Gurobi :

   ```bash
   cd /opt/gurobi1002/linux64/src/build
   # Remarque : le nom du dossier peut varier selon la version (exemple : gurobi1002)
   ```

2. Compilez Gurobi :

   ```bash
   sudo make
   ```

3. Copiez la bibliothèque compilée :

   ```bash
   sudo cp libgurobi_c++.a ../../lib/
   ```

---

## 2. Compilation des packages ROS2

Après avoir installé Gurobi, procédez à la compilation des packages suivants.

1. **Compilation du premier ensemble de packages :**

   ```bash
   colcon build --symlink-install --packages-select jps3d decomp_util convex_decomp_util path_finding_util voxel_grid_util decomp_ros_msgs decomp_ros_utils
   ```

2. **Sourcer le setup :**

   ```bash
   source install/setup.bash
   ```

3. **Compilation du second ensemble de packages :**

   ```bash
   colcon build --symlink-install --packages-select env_builder_msgs env_builder mapping_util multi_agent_planner_msgs multi_agent_planner global_map rl_interface
   ```

---

## 3. Lancement et exécution du système

D'abord copiez ppo_drone_exploration_model.zip dans le dossier clean_swarm
Utilisez les commandes ci-dessous pour démarrer les différents composants de votre système :

1. **Lancer RViz2 :**

   ```bash
   cd ~/clean_swarm
   source install/setup.bash
   rviz2 -d ~/clean_swarm/src/multi_agent_pkgs/multi_agent_planner/rviz/rviz_config_multi.rviz
   ```

2. **Lancer l'assembleur d'environnement :**

   ```bash
   cd ~/clean_swarm
   source install/setup.bash
   ros2 launch env_builder env_builder.launch.py
   ```

3. **Lancer le multi-agent planner avec reinforcement learning :**

   ```bash
   cd ~/clean_swarm
   source install/setup.bash
   ros2 launch multi_agent_planner multi_agent_planner_RL.launch.py
   ```

4. **Lancer le constructeur de carte globale :**

   ```bash
   cd ~/clean_swarm
   source install/setup.bash
   ros2 launch global_map global_map_builder.launch.py
   ```

5. **Lancer l'interface de reinforcement learning :**

   ```bash
   cd ~/clean_swarm
   source venv/bin/activate
   source install/setup.bash
   ros2 launch rl_interface rl.launch.py
   ```

---

## Notes complémentaires

- Assurez-vous d'être dans le bon environnement (par exemple, activer le virtualenv si nécessaire) avant d'exécuter les commandes.
- Les chemins indiqués (tel que `~/clean_swarm`) doivent être adaptés à votre structure de répertoire.
- Certaines étapes, notamment le téléchargement de Gurobi et l'installation de la licence, doivent être réalisées manuellement en suivant les instructions fournies.

---

Ce fichier README fournit une documentation complète pour configurer et lancer votre projet. En cas de problème, consultez la documentation spécifique de Gurobi.