# Cropping with Pretrained Edge Detection Model (DexiNed)

# Contents

### DexiNed Codes
---
* **checkpoints** : Pretrained edge detection model
* **result** : Edge detection results
* **utils** : Utils for DexiNed
* **datasets.py** : Datasets file for DexiNed
* **dexi_utils.py** : Dexi utils
* **losses** : Losses for DexiNed
* **main.py** : Run edge detection model
* **model.py** : Model for DexiNed

### My additions
---
* **Notebook** : Jupyter notebooks for implementations
* **contour_run.py** : Run contour codes and get cropped image
* **contour.py** : Contour class
* **hough_transform.py** : Hough Transform class
* **hough_run.py** : Run hough transform codes and get cropped image
* **myutils.py** : Utils
* **data** : Image data
* **contourResults** : Contour cropping results
* **houghResults** : Hough cropping results

# Run

1. Download the checkpoint and save into "./checkpoints/BIPED/10/" :
```
https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu/view
```
2. Run edge detection model:
```py
python ./main.py
```
3. Run contour to get cropped image:
```py
python contour_run.py
```
or Run hough transform to get cropped image:
```
python hough_run.py
```

# Sources
```
https://github.com/xavysp/DexiNed
```
```
https://arxiv.org/pdf/2112.02250.pdf
```
```
https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
```