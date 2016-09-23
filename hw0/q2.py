import sys
from PIL import Image

pic = Image.open(sys.argv[1])
res = pic.rotate(180)
res.save('ans2.png')