__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'


import argparse, glob, sys, pickle
import librosa
import numpy as np
import ssl_features, misc

	
def log(x, eps=1e-10):
	mn = np.min(x)
	return np.log10(x - mn + eps)

def normalize(x, eps=1e-10):
	mx = np.max(x) + eps
	mn = np.min(x)
	return (x - mn) / (mx - mn)

def voiced(x, frame_boundaries, sampling_rate, fmin=50, fmax=600, frame_shift=0.01, frame_size=0.025, z_transform=False):
	if type(frame_boundaries) != list: frame_boundaries = list(frame_boundaries)
	hl = int(sampling_rate * frame_shift)
	frame_boundaries.append(len(x)//hl)
	if not 0 in frame_boundaries: frame_boundaries.insert(0, 0)
	sfb = sorted(frame_boundaries)
	wl = int(sampling_rate*frame_size)
	out = []
	for i in range(1, len(sfb)):
		b = sfb[i-1]*hl; e = sfb[i]*hl
		interval = x[b:e]
		f0, voiced_flag, voiced_prob = librosa.pyin(interval, sr=sampling_rate, fmin=fmin, fmax=fmax, hop_length=hl, frame_length=wl)
		voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
		out.append(voiced_ratio)
	out = np.array(out)
	if z_transform: out = misc.z_score(out)
	return out 

def rms(x, frame_boundaries, sampling_rate, frame_shift=0.01, frame_size=0.025, z_transform=False):
	if type(frame_boundaries) != list: frame_boundaries = list(frame_boundaries)
	hl = int(sampling_rate * frame_shift)
	frame_boundaries.append(len(x)//hl)
	if not 0 in frame_boundaries: frame_boundaries.insert(0, 0)
	sfb = sorted(frame_boundaries)
	wl = int(sampling_rate*frame_size)
	out = []
	for i in range(1, len(sfb)):
		b = sfb[i-1]*hl; e = sfb[i]*hl
		interval = x[b:e]
		r = librosa.feature.rms(y=interval, frame_length=wl, hop_length=hl)
		out.append(np.mean(r))
	out = np.array(out)
	if z_transform: out = misc.z_score(out)
	return out 

def zcr(x, frame_boundaries, sampling_rate, frame_shift=0.01, frame_size=0.025, z_transform=False):
	if type(frame_boundaries) != list: frame_boundaries = list(frame_boundaries)
	hl = int(sampling_rate * frame_shift)
	frame_boundaries.append(len(x)//hl)
	if not 0 in frame_boundaries: frame_boundaries.insert(0, 0)
	sfb = sorted(frame_boundaries)
	wl = int(sampling_rate*frame_size)
	out = []
	for i in range(1, len(sfb)):
		b = sfb[i-1]*hl; e = sfb[i]*hl
		interval = x[b:e]
		z = librosa.feature.zero_crossing_rate(y=interval, frame_length=wl, hop_length=hl)
		out.append(np.mean(z))
	out = np.array(out)
	if z_transform: out = misc.z_score(out)
	return out 

def silence_intervals(x, frame_boundaries, kwargs={}):
	"""Check if each interval in x is silent."""
	sampling_rate = kwargs.get('sampling_rate', 16000)
	frame_shift = kwargs.get('frame_shift', 0.01)
	frame_size = kwargs.get('frame_size', 0.025)
	hl = kwargs.get('hop_length', int(sampling_rate*frame_shift))
	wl = kwargs.get('win_length', int(sampling_rate*frame_size))
	rms_z_threshold = kwargs.get('rms_z_threshold', -1)
	fmin = kwargs.get('fmin', 50)
	fmax = kwargs.get('fmax', 600)
	v_intervals = voiced(x, frame_boundaries, sampling_rate, fmin=fmin, fmax=fmax, frame_shift=frame_shift, frame_size=frame_size)
	r_intervals = rms(x, frame_boundaries, sampling_rate, frame_shift=frame_shift, frame_size=frame_size, z_transform=True)
	out = np.ones_like(v_intervals)
	out *= (v_intervals == 0)
	out *= (r_intervals < rms_z_threshold) 
	return out

if __name__ == '__main__': pass
