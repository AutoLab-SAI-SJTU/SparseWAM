from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import numpy.typing as npt
from pathlib import Path

import torch
# from torchvision import transforms

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
# from .bevformer.bev_feature_build import _get_bev_feature

from PIL import Image
from scipy.interpolate import CubicSpline

class DrivoRFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: Dict):
        self._config = config
        self._sparse_depth_debug_count = 0

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "drivor_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = {}
        data_camera = self._get_camera_feature(agent_input)
        features.update(data_camera)


        if len(self._config.lidar_pc) > 0 and self._config.get("use_lidar_backbone", True):
            data_lidar = self._get_lidar_feature(agent_input)
            features.update(data_lidar)

        ego_feature_list=[]

        for ego_status in agent_input.ego_statuses:
            if ego_status is None:
                continue
            pose=torch.tensor(ego_status.ego_pose, dtype=torch.float32)
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
            driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)
            ego_feature=torch.cat([pose,velocity, acceleration, driving_command], dim=-1)

            ego_feature_list.append(ego_feature)

        features["ego_status"] =torch.stack(ego_feature_list)

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]
        default_camera_order = ["cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2"]
        camera_order = self._config.get("camera_order", default_camera_order)
        cameras = [
            getattr(cameras, camera_name)
            for camera_name in camera_order
            if len(self._config.get(camera_name, [])) > 0
        ]

        lidar_points = self._get_current_lidar_points(agent_input)
        use_sparse_depth = bool(self._config.get("use_sparse_depth", False))

        images = []
        cam_Ks = []
        lidar2cams = []
        sparse_depths = []
        debug_images = []
        for cam in cameras:
            if cam.image is None:
                continue

            im = Image.fromarray(cam.image)
            cam_K = np.array(cam.intrinsics)
            sensor2lidar_rotation = np.asarray(cam.sensor2lidar_rotation)
            sensor2lidar_translation = np.asarray(cam.sensor2lidar_translation)
            sensor2lidar_rt = np.eye(4)
            sensor2lidar_rt[:3, :3] = sensor2lidar_rotation
            sensor2lidar_rt[:3, 3] = sensor2lidar_translation
            lidar2cam_rt = np.linalg.inv(sensor2lidar_rt)

            # intrinsics resize
            original_size = im.size
            cam_K = cam_K.clone() if isinstance(cam_K, torch.Tensor) else cam_K.copy() # torch.Size([8, 3, 3])
            cam_K[0, 0] = cam_K[0, 0] * self._config.image_size[0] / original_size[0]
            cam_K[1, 1] = cam_K[1, 1] * self._config.image_size[1] / original_size[1]
            cam_K[0, 2] = cam_K[0, 2] * self._config.image_size[0] / original_size[0]
            cam_K[1, 2] = cam_K[1, 2] * self._config.image_size[1] / original_size[1]

            # image resize
            im = im.resize(self._config.image_size)
            im_np = np.asarray(im)

            sparse_depth = None
            if use_sparse_depth:
                sparse_depth = self._project_lidar_to_sparse_depth(
                    lidar_points=lidar_points,
                    lidar2cam=lidar2cam_rt,
                    cam_K=cam_K,
                    image_size=self._config.image_size,
                )

            # PIL to numpy and normalize
            im = im_np.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            im = (im - mean) / std

            # convert to torch
            im = torch.from_numpy(im).permute(2, 0, 1)
            cam_K = torch.from_numpy(cam_K)
            lidar2cam_rt = torch.from_numpy(lidar2cam_rt)

            images.append(im)
            cam_Ks.append(cam_K)
            lidar2cams.append(lidar2cam_rt)
            if use_sparse_depth:
                sparse_depths.append(torch.from_numpy(sparse_depth).unsqueeze(0))
                debug_images.append(im_np)


        # Collect all camera images in a list for easier processing
        data = {
            "image": torch.stack(images),
            "cam_K": torch.stack(cam_Ks),
            "world_2_cam": torch.stack(lidar2cams)
        }
        if use_sparse_depth:
            data["sparse_depth"] = torch.stack(sparse_depths).float()
            self._maybe_save_sparse_depth_debug(debug_images, sparse_depths)
        
        # raise NotImplementedError


        # data["image"] = torch.stack([transforms.ToTensor()(img) for img in data["image"]])
        # data["cam_K"] = torch.stack([torch.from_numpy(cam) for cam in data["cam_K"]])
        # data["world_2_cam"] = torch.stack([torch.from_numpy(world_2_cam) for world_2_cam in data["world_2_cam"]])

        # data["image"] = data["image"].unsqueeze(0) # add time dimension
        # data["cam_K"] = data["cam_K"].unsqueeze(0) # add time dimension
        # data["world_2_cam"] = data["world_2_cam"].unsqueeze(0) # add time dimension

        return data

    def _get_current_lidar_points(self, agent_input: AgentInput) -> Optional[np.ndarray]:
        if not agent_input.lidars:
            return None
        lidar = agent_input.lidars[-1]
        if lidar is None or lidar.lidar_pc is None:
            return None
        return np.asarray(lidar.lidar_pc[LidarIndex.POSITION].T, dtype=np.float32)

    def _project_lidar_to_sparse_depth(
        self,
        lidar_points: Optional[np.ndarray],
        lidar2cam: np.ndarray,
        cam_K: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        width, height = image_size
        sparse_depth = np.zeros((height, width), dtype=np.float32)
        if lidar_points is None or lidar_points.size == 0:
            return sparse_depth

        points = lidar_points[np.isfinite(lidar_points).all(axis=1)]
        if points.size == 0:
            return sparse_depth

        points_h = np.concatenate(
            [points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        points_cam = (lidar2cam.astype(np.float32) @ points_h.T).T[:, :3]
        z = points_cam[:, 2]
        min_depth = float(self._config.get("sparse_depth_min_depth", 0.1))
        max_depth = float(self._config.get("sparse_depth_max_depth", 150.0))
        valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
        points_cam = points_cam[valid]
        z = z[valid]
        if z.size == 0:
            return sparse_depth

        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        u = fx * points_cam[:, 0] / z + cx
        v = fy * points_cam[:, 1] / z + cy
        cols = np.rint(u).astype(np.int32)
        rows = np.rint(v).astype(np.int32)
        in_image = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        if not np.any(in_image):
            return sparse_depth

        cols = cols[in_image]
        rows = rows[in_image]
        z = z[in_image].astype(np.float32)

        flat = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(flat, rows * width + cols, z)
        sparse_depth = flat.reshape(height, width)
        sparse_depth[~np.isfinite(sparse_depth)] = 0.0
        return sparse_depth

    def _maybe_save_sparse_depth_debug(
        self,
        images: List[np.ndarray],
        sparse_depths: List[torch.Tensor],
    ) -> None:
        if not bool(self._config.get("sparse_depth_debug", False)):
            return
        max_samples = int(self._config.get("sparse_depth_debug_max_samples", 16))
        if self._sparse_depth_debug_count >= max_samples:
            return
        output_dir = Path(self._config.get("sparse_depth_debug_dir", "debug_sparse_depth"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for cam_idx, (image, sparse_depth_t) in enumerate(zip(images, sparse_depths)):
            sparse_depth = sparse_depth_t.squeeze(0).numpy()
            overlay = self._draw_sparse_depth_overlay(image, sparse_depth)
            filename = output_dir / f"{self._sparse_depth_debug_count:06d}_cam{cam_idx}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        self._sparse_depth_debug_count += 1

    def _draw_sparse_depth_overlay(
        self,
        image: np.ndarray,
        sparse_depth: np.ndarray,
    ) -> np.ndarray:
        overlay = np.ascontiguousarray(image.copy())
        valid = sparse_depth > 0
        if not np.any(valid):
            return overlay
        depth_values = sparse_depth[valid]
        vmax = np.percentile(depth_values, 95)
        vmin = np.percentile(depth_values, 5)
        denom = max(float(vmax - vmin), 1e-6)
        norm = np.zeros_like(sparse_depth, dtype=np.uint8)
        norm[valid] = np.clip((sparse_depth[valid] - vmin) / denom * 255.0, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
        radius = int(self._config.get("sparse_depth_debug_point_radius", 2))
        rows, cols = np.where(valid)
        for row, col in zip(rows, cols):
            color = tuple(int(c) for c in colors[row, col])
            cv2.circle(overlay, (int(col), int(row)), radius + 1, (0, 0, 0), thickness=-1)
            cv2.circle(overlay, (int(col), int(row)), radius, color, thickness=-1)
        return overlay


    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # # only consider (x,y,z) & swap axes for (N,3) numpy array
        # lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # lidar_feature = voxelize_with_feature_averaging(lidar_pc, grid_dims=self._config.grid_dims, grid_range=self._config.grid_range)

        # return {"lidar_feature": lidar_feature}

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                self._config.lidar_image_size[0]+1,
                # (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                self._config.lidar_image_size[1]+1,
                # (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.lidar_hist_max_per_pixel] = self._config.lidar_hist_max_per_pixel
            overhead_splat = hist / self._config.lidar_hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.lidar_max_height]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.lidar_use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        features = np.expand_dims(features, axis=0) # add a dimension for the number of sensors (actually 1s)

        # return torch.tensor(features)
        return {"lidar_feature": torch.tensor(features)}

class DrivoRTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: Dict):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "drivor_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(
                num_trajectory_frames=self._config.trajectory_sampling.num_poses
            ).poses
        )
        # frame_idx = scene.scene_metadata.num_history_frames - 1
        # annotations = scene.frames[frame_idx].annotations
        # ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        # agent_states, agent_labels = self._compute_agent_targets(annotations)
        # bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        if self._config.long_trajectory_additional_poses > 0:
            try:
                trajectory_long = scene.get_future_trajectory(
                        num_trajectory_frames=self._config.trajectory_sampling.num_poses + self._config.long_trajectory_additional_poses
                    ).poses
                x = np.arange(trajectory_long.shape[0], dtype=np.float32)
                alpha = 2 * self._config.long_trajectory_additional_poses / (self._config.trajectory_sampling.num_poses*(self._config.trajectory_sampling.num_poses+1))
                x_new = np.arange(trajectory.shape[0], dtype=np.float32)
                off_sets = np.cumsum((x_new+1)*alpha)
                x_new += off_sets
                traj_ = []
                for i in range(3):
                    y = trajectory_long[:,i]
                    cs = CubicSpline(x, y)
                    traj_.append(cs(x_new))
                trajectory_long = np.stack(traj_, axis=1)

                trajectory_long = torch.tensor(trajectory_long)
                return {
                    "trajectory": trajectory,
                    "trajectory_long": trajectory_long,
                    "token":scene.scene_metadata.initial_token
                }
            except:
                return {
                    "trajectory": trajectory,
                    "trajectory_long": trajectory,
                    # "agent_states": agent_states,
                    # "agent_labels": agent_labels,
                    # "bev_semantic_map": bev_semantic_map,
                    "token":scene.scene_metadata.initial_token
                }
        else:

            return {
                "trajectory": trajectory,
                # "agent_states": agent_states,
                # "agent_labels": agent_labels,
                # "bev_semantic_map": bev_semantic_map,
                "token":scene.scene_metadata.initial_token
            }

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: DrivoRConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[0],
                box[1],
                box[6],
                box[3],
                box[4],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., slice(0, 1 + 1)], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)#128,256
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
