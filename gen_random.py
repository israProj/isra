import numpy as np
from collections import deque

import settings 

rand_record = []
rand_deque = deque()
choosed_node = set()

# print("seed=", settings.SEED)

np.random.seed(settings.SEED)

REPLAY = False
REDUCE = False

def ran(x, y):
	# print("Rand[%d,%d]" % (x, y))
	if REPLAY:
		global rand_deque
		r = rand_deque.popleft()
		if x + r >= y:
			raise Exception("REPLAY FAIL!")
		# print(x, y, x + r)
		return x + r

	r = np.random.randint(x, y)
	global rand_record
	rand_record += [r - x]
	# print(x, y, r)
	return r

def ran_ord(l):
	x_ord = np.arange(l)
	for x_index in range(l):
		swap_ind = ran(x_index, l)
		x_ord[x_index], x_ord[swap_ind] = x_ord[swap_ind], x_ord[x_index]
	return x_ord

def ran_ord_b(l):
	c = 5
	if l <= c:
		return ran_ord(l)
	tmp = ran_ord(c)
	return [i + l - c for i in tmp] + list(ran_ord(l - c))

def ran_input(shape):
	return np.random.random(tuple(shape))
