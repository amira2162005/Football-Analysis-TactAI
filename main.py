import os
import cv2
import csv
import numpy as np
from utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner import TeamAssigner
from number_recognizer import NumberRecognizer


def main():
    video_path = "input_videos/CV_Task.mkv"
    output_path = "output_videos/output_with_teams_and_numbers.mp4"
    csv_output_path = "output_videos/player_tracking_log.csv"

    # Step 1: Read video
    video_frames = read_video(video_path)

    # Step 2: Initialize tracker (YOLO)
    tracker = Tracker("models/best.pt")

    # Step 3: Get object tracks (players + ball)
    tracks = tracker.get_object_tracks(video_frames)
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 4: Train team assigner on first 10 frames
    team_assigner = TeamAssigner()
    print("🔹 Training team color KMeans model on first 10 frames...")
    team_assigner.assign_team_color(video_frames, tracks['players'], num_frames=10)
    print("✅ Team color model trained successfully.")

    # Step 5: Assign team ID & color to each player
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team_id = team_assigner.get_player_team(video_frames[frame_num],
                                                    track['bbox'],
                                                    player_id)
            tracks['players'][frame_num][player_id]['team'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_id]

    # Step 6: Initialize Jersey Number Recognizer
    number_recognizer = NumberRecognizer("models/yolo11m.pt", conf=0.4, gamma=1.4)

    # Step 7: Detect jersey numbers for each player with tracking linkage
    print("🔹 Recognizing player jersey numbers with tracking linkage...")
    for frame_num, player_track in enumerate(tracks['players']):
        frame = video_frames[frame_num]
        for player_id, track in player_track.items():
            bbox = track['bbox']
            number = number_recognizer.recognize_number(frame, bbox, player_id, frame_num)
            tracks['players'][frame_num][player_id]['number'] = number

        # Clean up inactive players periodically
        number_recognizer.cleanup(frame_num)
    print("✅ Jersey number recognition complete.")

    # Step 8: Prepare CSV file for logging
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    fps = 30  # Default FPS (you can modify or extract dynamically if needed)
    with open(csv_output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame Number", "Timestamp (s)", "Player ID"])

        # Step 9: Draw results and write CSV rows
        for frame_num, frame in enumerate(video_frames):
            timestamp = round(frame_num / fps, 2)

            for player_id, track in tracks['players'][frame_num].items():
                bbox = track['bbox']
                team_color = track.get('team_color', (255, 255, 255))
                number = track.get('number', None)

                x1, y1, x2, y2 = map(int, bbox)
                label = f"Team {track.get('team', '?')}"
                if number:
                    label += f" | #{number}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, team_color, 2)

                # Write player tracking info to CSV
                writer.writerow([frame_num, timestamp, player_id])

    # Step 10: Save output video
    save_video(video_frames, output_path)
    print(f"🎉 Done! Saved tracking + team + number video at {output_path}")
    print(f"🧾 CSV log saved at {csv_output_path}")


if __name__ == "__main__":
    main()
