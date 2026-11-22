"""Measure stability of representations within an interval."""

import argparse, glob
import librosa
import numpy as np
from scipy import stats, spatial
import matplotlib
import matplotlib.pyplot as plt
import ssl_features, plot

def load_ssl(args_ssl_config, args_ssl_hidden_layer=None):
	ssl_config = ssl_features.load_config(args_ssl_config)
	ssl_features.load_model(ssl_config)
	if args_ssl_hidden_layer: ssl_config['hidden_layer'] = args_ssl_hidden_layer
	return ssl_config

def load_map(fn):
	with open(fn) as f:
		out = {line.strip().split(',')[0]:line.strip().split(',')[1] for line in f}
	return out

def find_match(wav, phns):
	"""Find phn matching wav."""
	handle = wav.split('/')[-1].split('.')[0]
	out = None
	for phn in phns:
		ph = phn.split('/')[-1].split('.')[0]
		if ph == handle:
			out = phn
			break
	return out

def load_phn(phn):
	"""Load phone boundaries and labels from phn.""" 
	boundaries = []; labels = []
	with open(phn) as f:
		for line in f:
			ll = line.strip().split()
			boundaries.append(int(ll[1]))
			labels.append(ll[-1])
	return boundaries[:-1], labels

def list_intervals(representation, boundaries):
	"""List intervals from representation
	Args:
	- representation: (D, T)
	"""
	out = []
	prev = 0
	for b in boundaries:
		interval = representation[:, prev:b]
		out.append(interval)
		prev = b
	interval = representation[:, boundaries[-1]:]
	out.append(interval)
	return out

def logmel(x, fs, n_mels=40, frame_shift=0.01, frame_size=0.025):
	x = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_mels, n_fft=int(fs*frame_size), hop_length=int(fs*frame_shift))
	return log(normalize(x))

def inverse_variance_weights(x, eps=1e-20):
	var = np.var(x, axis=1)
	w = 1.0 / (var + eps)
	w = w / np.mean(w)   # average weight = 1
	return w

def distance_sequence(x, distance_func, distance_kwargs):
	"""Distance computed at multiple points in the sequence.
	Args:
	- x: representation; np.array of shape (D, T)
	"""
	left = distance_kwargs.get('left', 1)
	right = distance_kwargs.get('right', 0)
	x = np.pad(x, ((0,0), (left,right)), mode='edge')
	return [distance_func(x[:, i-left], x[:, i+right], distance_kwargs) for i in range(left, x.shape[1]-right)]

def argmax_distance(x, y, kwargs={}):
	return float(np.argmax(x) != np.argmax(y))

def cosine_distance(x, y, kwargs={}):
	return 1 - x@y/(np.linalg.norm(x)*np.linalg.norm(y))

def euclidean_distance(x, y, kwargs={}):
	return spatial.distance.euclidean(x, y, w=kwargs.get('w', np.ones(len(x))))

def jensenshannon(x, y, kwargs={}):
	return spatial.distance.jensenshannon(x, y)

def canberra_distance(x, y, kwargs={}):
	return spatial.distance.canberra(x, y, w=kwargs.get('w', np.ones(len(x))))

def braycurtis_distance(x, y, kwargs={}):
	return spatial.distance.braycurtis(x, y, w=kwargs.get('w', np.ones(len(x))))

def manhattan_distance(x, y, kwargs={}):
	return spatial.distance.cityblock(x, y, w=kwargs.get('w', np.ones(len(x))))

def wasserstein_distance(x, y, kwargs={}):
	"""Wasserstein distance between x and y:

	Args:
	- x, y: row vector(s), i.e. each row is a sample and each column is a feature dimension
	"""
	return stats.wasserstein_distance_nd(x, y)

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

def log(x, eps=1e-10):
	mn = np.min(x)
	return np.log(x-mn+eps)

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

def distance_from_mean(xs, mean):
	return np.mean([cosine_distance(xs[:, i], mean) for i in range(xs.shape[1])])

def between_over_within(x, y, distance_func=cosine_distance, distance_func_kwargs={}, eps=1e-20):
	"""Compute between-distance to within-distance ratio."""
	x_mean = np.mean(x, axis=1)
	y_mean = np.mean(y, axis=1)
	within = (distance_from_mean(x, x_mean) + distance_from_mean(y, y_mean)) / 2
	between = distance_func(x_mean, y_mean, distance_func_kwargs=distance_func_kwargs)
	score = between / (within + eps)
	return score

def get_biphones(phn):
	out = {}
	with open(phn) as f:
		ps = 0; pl = None
		for line in f:
			s, e, l = line.strip().split()
			s = int(s); e = int(e)
			if not pl is None:
				bi_phone = (pl, l)
				bi_boundaries = (ps, s, e)
				out[bi_boundaries] = bi_phone
			ps = s; pl = l
	return out

def length(bid):
	x = []
	for b1, b2, b3 in sorted(bid):
		x.append(b3-b1)
	x = np.array(x)
	return misc.z_score(x)

def voiced(x, fs, bid, fmin=50, fmax=600, frame_shift=0.01, frame_size=0.025):
	hl = int(fs*frame_shift)
	wl = int(fs*frame_size)
	out = []
	for b1, b2, b3 in sorted(bid):
		interval = x[b1:b3]
		f0, voiced_flag, voiced_prob = librosa.pyin(interval, sr=fs, fmin=fmin, fmax=fmax, hop_length=hl, frame_length=wl)
		voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
		out.append(voiced_ratio)
	out = np.array(out)
	return misc.z_score(out)

def rms(x, fs, bid, frame_shift=0.01, frame_size=0.025):
	hl = int(fs*frame_shift)
	wl = int(fs*frame_size)
	out = []
	for b1, b2, b3 in sorted(bid):
		interval = x[b1:b3]
		r = librosa.feature.rms(y=interval, frame_length=wl, hop_length=hl)
		out.append(np.mean(r))
	out = np.array(out)
	return misc.z_score(out)

def zcr(x, fs, bid, frame_shift=0.01, frame_size=0.025):
	hl = int(fs*frame_shift)
	wl = int(fs*frame_size)
	out = []
	for b1, b2, b3 in sorted(bid):
		interval = x[b1:b3]
		z = librosa.feature.zero_crossing_rate(y=interval, frame_length=wl, hop_length=hl)
		out.append(np.mean(z))
	out = np.array(out)
	return misc.z_score(out)

