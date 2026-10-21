__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'


import argparse, glob, sys, pickle
import librosa
import numpy as np
from scipy import ndimage
import distance, segfilter, ssl_features

def load_offsets_or_durations(fn):
	x = {}
	with open(fn) as f:
		header = f.readline()
		for line in f:
			fname, val = line.strip().split(',')
			x[fname] = float(val)
	return x

def log(x, eps=1e-10):
	mn = np.min(x)
	return np.log10(x - mn + eps)

def normalize(x, eps=1e-10):
	mx = np.max(x) + eps
	mn = np.min(x)
	return (x - mn) / (mx - mn)

def find_peaks(d, threshold):
	bigger_than_left = d[1:-1] > d[:-2]
	bigger_than_right = d[1:-1] > d[2:]
	bigger_than_threshold = d[1:-1] > threshold
	peaks = bigger_than_left * bigger_than_right * bigger_than_threshold
	return np.where(peaks == 1)[0] + 1

def peaks(d, peak_height, offset, frame_shift):
	p = find_peaks(d, peak_height)
	return p * frame_shift + offset

def save_as_textgrid(ofn, boundaries, duration):
	bs = [round(b, 6) for b in boundaries]
	bs = [0.0] + bs + [duration]
	stamps = [(bs[i], bs[i+1]) for i in range(len(bs)-1)]
	interval_name = 'phones'
	header = 'File type = "ooTextFile"\nObject class = "TextGrid"'
	header += '\n\nxmin = 0\nxmax = ' + str(duration)
	header += '\ntiers? <exists>\nsize = 1'
	subheader = '\nitem []:\n\titem[1]:'
	subheader += '\n\t\tclass = "IntervalTier"\n\t\tname = "' + interval_name + '"'
	subheader += '\n\t\txmin = 0\n\t\txmax = ' + str(duration)
	subheader += '\n\t\tintervals: size = ' + str(len(stamps)) 
	out = header + subheader
	for i, (st, et) in enumerate(stamps):
		if i == len(stamps)-1: et = min(et, duration)
		entry = '\n\t\t\tintervals [' + str(i+1) + ']'
		entry += '\n\t\t\txmin = ' + str(st)
		entry += '\n\t\t\txmax = ' + str(et)
		entry += '\n\t\t\ttext = ""'
		out += entry
	with open(ofn, 'w') as f: f.write(out)


def between_over_within(x, y, distance_kwargs, correction=False):
	whole = np.concatenate((x, y), axis=1)
	d = distance.distance_sequence(whole, kwargs=distance_kwargs)
	within = np.mean(d)
	between = distance_kwargs['distance_func'](distance_kwargs['translate_func'](x, kwargs=distance_kwargs['translate_kwargs']), distance_kwargs['translate_func'](y, kwargs=distance_kwargs['translate_kwargs']))
	c = 1
	if correction: c = 2 * np.sqrt(x.shape[1]*y.shape[1]) / (x.shape[1] + y.shape[1])
	return c * between / within

def silence_filter_b2w(w, fbs, is_silence_kwargs={}, sx=None, distance_kwargs={}, b2w_threshold=1.5, frame_shift=0.01, frame_size=0.025, sampling_rate=16000, rms_z_threshold=0):
	"""Remove silence false positives.
	Args:
	- w: waveform
	- fbs: boundaries in terms of frames 
	- sx: SSL representation (upsampled); numpy array of shape (D, T)
	- distance_kwargs: kwargs for computing distance
	"""
	sis = segfilter.silence_intervals(w, fbs, kwargs={'sampling_rate':sampling_rate, 'frame_shift':frame_shift, 'frame_size':frame_size, 'rms_z_threshold':rms_z_threshold})
	sil_sil = (sis[:-1]==1) * (sis[1:]==1)
	if not sx is None:
		bs = sorted(list(fbs))
		bs.insert(0, 0)
		bs.append(sx.shape[1])
		b2ws = np.array([between_over_within(sx[:, bs[i-1]:bs[i]], sx[:, bs[i]:bs[i+1]], distance_kwargs, correction=True) for i in range(1, len(bs)-1)])
		sil_sil *= (b2ws < b2w_threshold)
	to_remove = fbs[np.where(sil_sil==1)[0]]
	return to_remove



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--wav', type=str)
	parser.add_argument('--seg_ssl', type=str)
	parser.add_argument('--seg_ssl_layer', type=str)
	parser.add_argument('--sil_ssl', type=str)
	parser.add_argument('--sil_ssl_layer', type=str)
	parser.add_argument('--distance_config', type=str, default='./distance_config.json', help='json file specifying distance kwargs')
	parser.add_argument('--normalize_distance', action='store_true')
	parser.add_argument('--silence_filter', action='store_true')
	parser.add_argument('--silence_filter_nobwr', action='store_true')
	parser.add_argument('--silence_filter_threshold', type=float, default=1.0)
	parser.add_argument('--rms_z_threshold', type=float, default=-0.5, help='rms z-score threshold for silence detection')
	parser.add_argument('--offset', type=float, default=0.0)
	parser.add_argument('--offset_file', type=str)
	parser.add_argument('--duration_file', type=str)
	parser.add_argument('--peak_height', type=float, default=0.05)
	parser.add_argument('--frame_shift', type=float, default=0.02)
	parser.add_argument('--outdir', type=str)
	parser.add_argument('--textgrid', action='store_true')
	parser.add_argument('--upsample', action='store_true', help='upsample to change frameshift from 20ms to 10ms when using SSL; do not use for whisper')
	parser.add_argument('--duplicate_frames', action='store_true', help='duplicate each frame to halve effective frame shift')
	args = parser.parse_args()
	if args.seg_ssl:
		seg_ssl_config = ssl_features.load_config(args.seg_ssl)
		ssl_features.load_model(seg_ssl_config)
		if args.seg_ssl_layer: seg_ssl_config['hidden_layer'] = args.seg_ssl_layer
		if args.upsample: seg_ssl_config['upsample'] = True
		if args.duplicate_frames: seg_ssl_config['duplicate_frames'] = True
	if args.sil_ssl:
		sil_ssl_config = ssl_features.load_config(args.sil_ssl)
		ssl_features.load_model(sil_ssl_config)
		if args.sil_ssl_layer: sil_ssl_config['hidden_layer'] = args.sil_ssl_layer
		sil_ssl_config['upsample'] = True
	offsets = {}
	if args.offset_file: offsets = load_offsets_or_durations(args.offset_file)
	durations = {}
	if args.duration_file: durations = load_offsets_or_durations(args.duration_file)
	default_distance_kwargs = distance.load_config(args.distance_config)
	for wav in glob.glob(args.wav):
		handle = wav.split('/')[-1]
		sys.stderr.write('# Segmenting ' + handle + '...\r')
		w, fs = librosa.load(wav, sr=None)
		if args.seg_ssl:
			x = ssl_features.get_features(wav, seg_ssl_config).detach().numpy()
			x = x.T
		else:
			x = librosa.feature.melspectrogram(y=w, sr=fs, n_mels=40, n_fft=int(fs*0.025), hop_length=int(fs*args.frame_shift))
			x = log(normalize(x))
		d = distance.distance_sequence(x, default_distance_kwargs)
		d = np.array(d)
		if args.normalize_distance: d = normalize(d)
		fps = find_peaks(d, args.peak_height)
		# filter
		if args.silence_filter:
			if args.silence_filter_nobwr:
				sx = None
			else: 
				if args.sil_ssl:
					sx = ssl_features.get_features(wav, sil_ssl_config).detach().numpy()
					sx = sx.T # (D, T)
				else:
					sx = x		
			sil_bs = silence_filter_b2w(w, fps, is_silence_kwargs={}, sx=sx, distance_kwargs={'left':[-1,-1], 'right':[1,1], 'distance_func':distance.cosine_distance, 'distance_kwargs':{}, 'translate_func':distance.mean_slice, 'translate_kwargs':{}}, b2w_threshold=args.silence_filter_threshold, frame_shift=0.01, frame_size=0.025, sampling_rate=fs, rms_z_threshold=args.rms_z_threshold) 
			fps = np.array([p for p in fps if not p in sil_bs])
		# convert boundaries in frame indices to boundaries in seconds
		p = fps * args.frame_shift + offsets.get(handle, args.offset)
		ofn = args.outdir + '/' + '.'.join(handle.split('.')[:-1])
		duration = durations.get(handle, args.frame_shift * len(d) + offsets.get(handle, args.offset))
		if args.textgrid:
			ofn = ofn + '.TextGrid'
			save_as_textgrid(ofn, p, duration)
		else:
			with open(ofn + '.BND', 'w') as of:
				for b in p: of.write(str(b) + '\n')
	sys.stderr.write('\n# Segmentation complete. Outputs saved in ' + args.outdir + '.\n')

