import pandas as pd
import os
import re
import cv2
import random
from glob import glob
from urllib import request
from tqdm import tqdm


def download_actions_clips():
    for file in tqdm(
        os.listdir("data/games_csv"), desc="Downloading clips...", ascii=" >="):
        game = re.search(r"\d+-([a-z]+-vs-[a-z]+)\.csv$", file).group(1)
        idx = len(glob(f"data/actions/{game}*"))
        df = pd.read_csv("data/games_csv/" + file)
        sample = df.sample()
        video_url = sample["video_url"].iloc[0].split(" ")[0]
        request.urlretrieve(video_url, f"data/actions/{game}_{idx}.mp4")


def extract_frames(n_frames):
    for action in tqdm(
        os.listdir("data/actions"), desc="Extracting frames...", ascii=" >="):
        idx = len(glob(f"data/frames/{action.removesuffix('.mp4')}*"))
        video = cv2.VideoCapture("data/actions/" + action)
        lenght = video.get(cv2.CAP_PROP_FRAME_COUNT)
        for _ in range(n_frames):
            frame_idx = random.randint(0, lenght)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                continue
            cv2.imwrite(f"data/frames/{action.removesuffix('.mp4')}_{idx}.png", frame)
            idx += 1


def extract_boxes():
    for frame in os.listdir("data/frames"):
        idx = len(glob(f"data/boxes/{frame.removesuffix('.png')}*"))
        image = cv2.imread("data/frames/" + frame)
        while True:
            box = cv2.selectROI("box", image)
            cropped = image[
                int(box[1]) : int(box[1] + box[3]), int(box[0]) : int(box[0] + box[2])
            ]
            if cropped.size == 0:
                break
            cv2.imshow("box", cropped)
            cv2.waitKey(5)
            number = input("Insert jersey number: ")
            cv2.imwrite(
                f"data/boxes/{number}_{frame.removesuffix('.png')}_{idx}.png", cropped
            )
            idx += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    download_actions_clips()
    extract_frames()
    extract_boxes()
