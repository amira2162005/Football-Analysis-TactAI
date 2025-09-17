import os
import numpy as np
from utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner import TeamAssigner


def main():
    video_path = "input_videos/CV_Task.mkv"
    output_path = "output_videos/output_with_teams.mp4"

    # Read video
    video_frames = read_video(video_path)

    # Initialize tracker with YOLO model
    tracker = Tracker("models/best.pt")

    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames)

    # Add positions for tracking logic (foot/center)
    tracker.add_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])  # detect main colors

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw annotations (detections + teams)
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video safely
    if output_video_frames:
        save_video(output_video_frames, output_path)
        print(f"Done! Saved tracking + team assignment video at {output_path}")
    else:
        print("No frames to save! Check YOLO model, confidence threshold, or input video.")


if __name__ == "__main__":
    main()
