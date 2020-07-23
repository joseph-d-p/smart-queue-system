import numpy as np
import time
import os
import cv2
import argparse
import sys

from queue import Queue
from person import PersonDetect

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=int(args.max_people)
    queue_param=args.queue_param
    threshold=float(args.threshold)
    output_path=args.output_path

    start_model_load_time=time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()

    """
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except Exception as e:
        print(e)
        print("error loading queue param file")
    """

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_video = cv2.VideoWriter(
        os.path.join(output_path, 'output_video.mp4'),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        fps,
        (initial_w, initial_h),
        True)

    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1

            coords, image= pd.predict(frame, initial_w, initial_h)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            sys.stdout.flush()
            out_text=""
            y_pixel=25

            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)

        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--max_people', default=None)
    parser.add_argument('--queue_param', default="")
    parser.add_argument('--threshold', default=0.6)
    parser.add_argument('--output_path', default='results')

    args=parser.parse_args()
    main(args)
