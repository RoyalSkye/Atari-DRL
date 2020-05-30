import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from config import *

def find_max_lifes(env):
	env.reset()
	_, _, _, info = env.step(0)
	return info['ale.lives']

def check_live(life, cur_life):
	if life > cur_life:
		return True
	else:
		return False

def get_frame(X):
	#convert rgb to grayscale
	x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
	return x

def get_init_state(history, s):
	for i in range(HISTORY_SIZE):
		history[i, :, :] = get_frame(s)
