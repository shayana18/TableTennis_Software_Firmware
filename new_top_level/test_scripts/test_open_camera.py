import cv2


WINDOW_LEFT = "Left Camera Feed"
WINDOW_RIGHT = "Right Camera Feed"


def prompt_camera_id(label: str) -> int:
    while True:
        raw_value = input(f"Enter the {label} camera ID: ").strip()
        try:
            return int(raw_value)
        except ValueError:
            print(f"Invalid {label} camera ID '{raw_value}'. Please enter an integer.")


def open_camera(camera_id: int, label: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open the {label} camera with ID {camera_id}.")
    return capture


def annotate_frame(frame, label: str, camera_id: int):
    text = f"{label} Camera | ID: {camera_id}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    left_camera_id = prompt_camera_id("left")
    right_camera_id = prompt_camera_id("right")

    left_capture = None
    right_capture = None

    try:
        left_capture = open_camera(left_camera_id, "left")
        right_capture = open_camera(right_camera_id, "right")

        print("Both camera feeds opened. Press 'q' in a video window to quit.")

        while True:
            left_ok, left_frame = left_capture.read()
            right_ok, right_frame = right_capture.read()

            if not left_ok:
                print(f"Failed to read a frame from the left camera (ID {left_camera_id}).")
                break

            if not right_ok:
                print(f"Failed to read a frame from the right camera (ID {right_camera_id}).")
                break

            cv2.imshow(WINDOW_LEFT, annotate_frame(left_frame, "Left", left_camera_id))
            cv2.imshow(WINDOW_RIGHT, annotate_frame(right_frame, "Right", right_camera_id))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as error:
        print(error)
    finally:
        if left_capture is not None:
            left_capture.release()
        if right_capture is not None:
            right_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
