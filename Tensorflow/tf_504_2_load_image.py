import os
import sys

import matplotlib.pyplot as plt

from PIL import Image

img_path=sys.argv[1]
image=Image.open(img_path).convert("RGB")


plt.figure(figsize=(25,20))
plt.imshow(image)
plt.show()
plt.close()