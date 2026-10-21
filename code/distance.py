__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'


import argparse, sys, json
import numpy as np
from scipy import stats, spatial, special

def load_config(json_fn):
	"""Load config file."""
	with open(json_fn, 'r', encoding='utf-8') as f: out = json.load(f)
	if 'distance_func' in out:
		out['distance_func'] = getattr(sys.modules[__name__], out['distance_func'])
	if 'translate_func' in out:
		out['translate_func'] = getattr(sys.modules[__name__], out['translate_func'])
	return out

def get_slice(x, i, index_pair):
	"""Get a slice of x adjacent to i delimited by index_pair."""
	return x[:, i+index_pair[0]:i+index_pair[1]+1]

def len_slice(index_pair):
	"""Calculate the length of the vector slice implied by index_pair."""
	return index_pair[1] - index_pair[0] + 1

def translate_slice(s, func=None, kwargs={}):
	"""Translate a vector slice s using func along with kwargs."""
	if func is None: return s
	else: return func(s, kwargs)

def flatten_slice(s, kwargs={}):
	"""Flatten slice to a 1D-array."""
	return s.flatten()

def mean_slice(s, kwargs={}):
	"""Compute the mean of the slice."""
	return s.mean(axis=1)

def mean_sd_slice(s, kwargs={}):
	"""Compute the mean and sd of the slice."""
	return np.concatenate((s.mean(axis=1), s.std(axis=1)))

def sum_softmax_slice(s, kwargs={}):
	# temp
	"""Compute the softmax after summing the slice per dimension."""
	return special.softmax(s.sum(axis=1))

def slope_slice(s, kwargs={}):
	# temp
	"""Compute the slope of linear regression per dimension."""
	idx = np.arange(s.shape[1])
	return np.polyfit(idx, s.T, deg=1)[0]

def distance_sequence(x, kwargs={}):
	"""Compute distance for each point in x.

	Args:
	- x: vector sequence; numpy array of shape (D, T)
	- kwargs: keyword arguments; dictionary
	-- 'left' : relative index pair indicating the left slice
	-- 'right' : relative index pair indicating the right slice
	-- 'translate_func' : translation function
	-- 'translate_func_kwargs' : keyword arguments for translation function
	-- 'distance_func' : distance function
	-- 'distance_func_kwargs' : keyword arguments for distance function
	"""
	left = kwargs.get('left', (-1,-1))
	right = kwargs.get('right', (0,0))
	#x = np.pad(x, ((0,0), (len_slice(left),len_slice(right))), mode='edge')
	x = np.pad(x, ((0,0), (-left[0],right[1])), mode='edge')
	translate_func = kwargs.get('translate_func', None)
	translate_func_kwargs = kwargs.get('translate_func_kwargs', {})
	distance_func = kwargs.get('distance_func', None)
	distance_func_kwargs = kwargs.get('distance_func_kwargs', {})
	return [distance_func(translate_slice(get_slice(x, i, left), translate_func, translate_func_kwargs), translate_slice(get_slice(x, i, right), translate_func, translate_func_kwargs), distance_func_kwargs) for i in range(-left[0], x.shape[1]-right[1])]

def delta(ls, rs):
	"""Delta

	Given vector sequence x and window size w,

	delta[i, t] = \sum_{k=1}^w k * (x[i,t+k] - x[i,t-k]) / (2 * \sum_{k=1}^w k^2)

	Args:
	- ls, rs: slices to the left and right of x[:, t]; numpy array of shape (D, T)
	"""
	win_size = ls.shape[1] # window size
	assert win_size == rs.shape[1], 'left slice and right slice have different lengths!'
	ls = np.fliplr(ls)
	weights = np.arange(1, win_size+1)
	d = (weights * (rs - ls)).sum(axis=1)
	den = 2 * np.sum(weights ** 2)
	return d / den

def spectral_transition_measure(ls, rs, kwargs={}):
	"""Spectral transition measure: mean delta squared across dimensions"""
	return (delta(ls, rs)**2).sum(axis=0) / ls.shape[0]

def max_distance(x, y, kwargs={}):
	return np.max(np.abs(x-y))

def argmax_distance(x, y, kwargs={}):
	return float(np.argmax(x) != np.argmax(y))

def cosine_distance(x, y, kwargs={}):
	return 1 - x@y/(np.linalg.norm(x)*np.linalg.norm(y))

def euclidean_distance(x, y, kwargs={}):
	return spatial.distance.euclidean(x, y, w=kwargs.get('w', np.ones(len(x))))

def mag_weighted_euclidean_distance(x, y, kwargs={}):
	"""Euclidean distance using magnitudes as weights
	Args:
	- x, y: arrays of scores
	"""
	#w = np.stack((np.abs(x), np.abs(y))).max(axis=0)
	w = np.stack((x, y)).max(axis=0)
	w = np.abs(w) 
	#w = np.exp(w)
	#w = special.softmax(w)
	return spatial.distance.euclidean(x, y, w=w)


def sd_weighted_euclidean_distance(x, y, kwargs={}):
	"""Euclidean distance using SD as weights
	Args:
	- x, y: arrays of scores first 1/2 are values to compare, latter 1/2 are weights
	"""
	n = int(len(x) / 2)
	#w = (x[:n]+y[:n]) / 2
	w = np.stack((x[n:], y[n:])).max(axis=0)
	return spatial.distance.euclidean(x[:n], y[:n], w=w)

def jensenshannon(x, y, kwargs={}):
	return spatial.distance.jensenshannon(x, y)

def canberra_distance(x, y, kwargs={}):
	return spatial.distance.canberra(x, y, w=kwargs.get('w', np.ones(len(x))))

def braycurtis_distance(x, y, kwargs={}):
	return spatial.distance.braycurtis(x, y, w=kwargs.get('w', np.ones(len(x))))

def manhattan_distance(x, y, kwargs={}):
	return spatial.distance.cityblock(x, y, w=kwargs.get('w', np.ones(len(x))))

def wasserstein_distance_1d(x, y, kwargs={}):
	"""Wasserstein distance (1d) between x and y."""
	return stats.wasserstein_distance(x, y)

def wasserstein_distance(x, y, kwargs={}):
	"""Wasserstein distance (2d) between x and y:

	Args:
	- x, y: row vector(s), i.e. each row is a sample and each column is a feature dimension
	"""
	return stats.wasserstein_distance_nd(x, y)

def total_variation_distance(p, q, kwargs={}):
	"""Total variation distance:
	the maximum possible distance between the two probabilities that the two distributions assign to the same event,
	visually, 1/2 area of the non-overlapping regions between the two pdfs

	d(p, q) = 1/2 \int_{-\inf}^{\inf} |p(x) - q(x)| dx
	"""
	return np.sum(np.abs(p-q)) / 2

def covariance_recording(representation, diagonal=True):
	"""Compute covariance matrix across frames in the recording.
	Args:
	- representation: (D, T)
	- diagonal: whether to compute a diagonal covariance matrix
	"""
	if diagonal: return np.diag(np.var(representation, axis=1))
	else: return np.cov(representation, rowvar=True)

def mahalanobis_distance(x, y, kwargs={}):
	"""Compute Mahalanobis distance."""
	vi = kwargs.get('VI', np.diag(np.ones(x.shape[0])))
	d = spatial.distance.mahalanobis(x, y, vi)
	return d 

def chebyshev_distance(x, y, kwargs={}):
	"""Top-k Chebyshev distance: average Chebyshev distance over dimensions with k-largest distance."""
	k = kwargs.get('k', 1)
	diff = np.abs(x.ravel() - y.ravel())
	return np.mean(np.partition(diff, -k)[-k:])

def normalize(x, eps=1e-10):
	mx = np.max(x) + eps
	mn = np.min(x)
	return (x-mn) / (mx-mn)

def z_score(x, eps=1e-10):
	x = np.array(x)
	m = np.mean(x)
	if len(x) == 1: s = 0.0
	else: s = np.std(x)
	return (x - m) / (s + eps)

def test():
	x = np.random.random((3,5))
	try:
		kwargs = load_config('distance_config.json')
	except: 
		kwargs = {'left':(-2,-1), 'right':(1,2), 'translate_func':mean_slice, 'translate_func_kwargs':{}, 'distance_func':cosine_distance, 'distance_func_kwargs':{}}
	d = distance_sequence(x, kwargs)
	print(d)
	

if __name__ == '__main__': test()
