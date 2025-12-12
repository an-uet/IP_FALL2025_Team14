import random
import time
import os
import pandas as pd

from naive_background_style_transfer import NaiveBackgroundStyleTransfer

t_start = time.time()
processing_times = []
def running(content_folder, style_folder, epochs=150, gif_speed=2):
    files = os.listdir(content_folder)
    style = os.path.join(style_folder)

    print(f"Found {len(files)} images in {content_folder}")
    for file in files:
        style = random.choice(os.listdir(style_folder))
        style_path = os.path.join(style_folder, style)
        t = time.time()
        print(f"Processing image: {file}")
        content_path = os.path.join(content_folder, file)
        nbst = NaiveBackgroundStyleTransfer(number_of_epochs=epochs, verbose = True, name=file.split('.')[0] + f"_{style}")
        nbst.perform(content_path, style_path)
        nbst.generate_gif(speed=gif_speed, name=file.split('.')[0] + f"_{style}")
        print("Total time: ", time.time() - t)
        processing_times.append((file, time.time() - t))
        df = pd.DataFrame(processing_times, columns=["Image", "Processing Time"])
        df.to_csv("processing_times.csv", index=False)
    print("Average processing time per image: ", sum(t for _, t in processing_times) / len(processing_times))
        


if __name__ == "__main__":
    content_folder = "../dataset/PPM-100/image/"
    style_folder = "../dataset/PPM-100/style/"
    running(content_folder, style_folder, epochs=100, gif_speed=1.5)
    print("Total execution time: ", time.time() - t_start)
