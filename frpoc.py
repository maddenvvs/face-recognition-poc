#!/usr/bin/env python3

# This is a demo of running face recognition on live video from your webcam.
# It's a little more complicated than the other example, but it includes some
# basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed
# only to read from your webcam. OpenCV is *not* required to use the
# face_recognition library. It's only required if you want to run this specific demo.
# If you have trouble installing it, try any of the other demos that don't require it instead.

from pathlib import Path

import face_recognition
import cv2
import numpy as np

FACES_FOLDER = "faces"
SCALE_RATIO = 4
FRAME_DETECTION_AT = 2

UNKNOWN_NAME = "Unknown"
UNKNOWN_COLOR = (0x84, 0x75, 0x45)
BORDER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def try_retrieve_encodings_of(image_path: Path):
    try:
        image = face_recognition.load_image_file(image_path)
        return face_recognition.face_encodings(image)
    except Exception as e:
        print(f"Error during retrieving encoding of image '{image_path}':", e)
        return []


def load_faces():
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    colors = {
        UNKNOWN_NAME: UNKNOWN_COLOR
    }
    color_idx = 0

    for sub_path in Path(FACES_FOLDER).iterdir():

        # Skip nested directories
        if (sub_path.is_dir()):
            continue

        face_encoding = try_retrieve_encodings_of(sub_path)
        for encoding in face_encoding:
            # Assume that we have only one face on the image
            file_name = sub_path.stem
            known_face_encodings.append(encoding)
            known_face_names.append(file_name)
            colors[file_name] = BORDER_COLORS[color_idx]

        color_idx = (color_idx + 1) % len(BORDER_COLORS)

    print(f"Encodings loaded: {len(known_face_encodings)}")

    return known_face_encodings, known_face_names, colors


def main() -> None:
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    known_face_encodings, known_face_names, colors = load_faces()

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    detection_frame = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Jump out of the loop if there is no frame available
        if not ret:
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(
            frame, (0, 0), fx=1/SCALE_RATIO, fy=1/SCALE_RATIO)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Only process every other frame of video to save time
        if detection_frame == 0:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = UNKNOWN_NAME

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        detection_frame = (detection_frame + 1) % FRAME_DETECTION_AT

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled
            top *= SCALE_RATIO
            right *= SCALE_RATIO
            bottom *= SCALE_RATIO
            left *= SCALE_RATIO

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), colors[name], 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom),
                          (right, bottom + 35), colors[name], cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom + 23),
                        font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow("Video", frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
