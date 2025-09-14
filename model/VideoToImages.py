import cv2
import os

# https://labelstud.io/
# Use this to label the model

def getVideo():
    # Open the default camera
    cam = cv2.VideoCapture(1)


    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()

def video_to_image(desired_images = 150):
    cam = cv2.VideoCapture("output.mp4")

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print ('Error: Creating directory of data')

    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // desired_images)
    currentframe = 0
    saved = 0

    while True:
        ret, frame = cam.read()
        if not ret or saved >= desired_images:
            break

        if currentframe % step == 0:
            name = f'./data/frame{saved}.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, frame)
            saved += 1

        currentframe += 1

    cam.release()
    cv2.destroyAllWindows()

def main():
    getVideo()
    video_to_image(150)


if __name__ == "__main__":
    main()