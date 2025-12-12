import time

from naive_background_style_transfer import NaiveBackgroundStyleTransfer

t1 = time.time()
content = "/mnt/HDD1/anlt/image_processing/Naive-Background-Style-Transfer/naive_background_style_transfer/Input_Images/Content/portrait.jpg"
style = "Input_Images/Style/Starry_Night.jpg"
nbst = NaiveBackgroundStyleTransfer(number_of_epochs=150, verbose = True)
nbst.perform(content, style)
nbst.generate_gif(speed=1.5)
print("Total time: ", time.time() - t1)