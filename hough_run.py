from hough_transform import HoughTransform
import os


# instance = HoughTransform("data/drive_10110-back.jpg", "result/BIPED2CLASSIC/fused/drive_10110-back.png")
# instance.run()

for img in os.listdir("data/"):
    try:
        print(f"{img}\n{20*'*'}")
        img_path = "data/"+img
        instance = HoughTransform(img_path, "result/BIPED2CLASSIC/avg/"+img.split(".")[0]+".png")
        instance.run()
    except:
        print(f"HATA: ----->   {img}\n{20*'*'}")
