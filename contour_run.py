from contour import Contour
import os

## Run specific file

# instance = Contour("data/drive_10110-back.jpg", "result/BIPED2CLASSIC/fused/drive_10110-back.png")
# instance.run()


## Run all files in data folder

for img in os.listdir("data/"):
    img_path = "data/"+img
    instance = Contour(img_path, "result/BIPED2CLASSIC/avg/"+img.split(".")[0]+".png")
    instance.run()