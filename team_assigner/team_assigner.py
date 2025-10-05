from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        # Reshape image to 2D (pixels × RGB)
        image_2d = image.reshape(-1, 3)

        # Perform K-Means with 2 clusters (player vs background)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp values to frame boundaries
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Validate bounding box
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0])

        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return np.array([0, 0, 0])

        # Focus on top half (usually contains jersey color)
        top_half_image = image[0:int(image.shape[0] / 2), :]
        if top_half_image.size == 0:
            return np.array([0, 0, 0])

        # Cluster colors
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Determine which cluster represents background (corners)
        corner_clusters = [
            labels[0, 0], labels[0, -1],
            labels[-1, 0], labels[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, video_frames, player_tracks, num_frames=10):
        """Train KMeans on player colors from multiple frames."""
        all_player_colors = []

        for frame_idx in range(min(num_frames, len(video_frames))):
            frame = video_frames[frame_idx]
            player_detections = player_tracks[frame_idx]

            for _, player_detection in player_detections.items():
                color = self.get_player_color(frame, player_detection["bbox"])
                all_player_colors.append(color)

        # Train team-level KMeans on all collected player colors
        if all_player_colors:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=20)
            kmeans.fit(all_player_colors)

            self.kmeans = kmeans
            self.team_colors[1] = kmeans.cluster_centers_[0]
            self.team_colors[2] = kmeans.cluster_centers_[1]
        else:
            raise ValueError("No player colors found for team assignment!")

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        self.player_team_dict[player_id] = team_id

        return team_id
