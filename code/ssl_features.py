__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import sys, re, argparse, glob, torch, torchaudio, transformers
import numpy as np

def load_config(fn):
	out = {}
	with open(fn) as f:
		for line in f:
			line = re.sub('#.*', '', line).strip()
			if line != '':
				ll = line.split(',')
				attr = ll[0].strip()
				val = ll[1].strip()
				if re.match(r'\d+$', val): val = int(val)
				elif re.match(r'\d*\.\d*$', val): val = float(val)
				out[attr] = val
	return out

def load_model(config):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	get_attention = 'attention_head' in config
	if config['feature'] == 'wav2vec2':
		if get_attention: config['model'] = transformers.Wav2Vec2Model.from_pretrained(config['model'], attn_implementation='eager').to(device).eval()
		else: config['model'] = transformers.Wav2Vec2Model.from_pretrained(config['model']).to(device).eval()
	elif config['feature'] == 'hubert':
		if get_attention: config['model'] = transformers.HubertModel.from_pretrained(config['model'], attn_implementation='eager').to(device).eval()
		else: config['model'] = transformers.HubertModel.from_pretrained(config['model']).to(device).eval()
	elif config['feature'] == 'wavlm':
		if get_attention: config['model'] = transformers.WavLMModel.from_pretrained(config['model'], attn_implementation='eager').to(device).eval()
		else: config['model'] = transformers.WavLMModel.from_pretrained(config['model']).to(device).eval()
	elif config['feature'] == 'data2vec':
		if get_attention: config['model'] = transformers.Data2VecAudioModel.from_pretrained(config['model'], attn_implementation='eager').to(device).eval()
		else: config['model'] = transformers.Data2VecAudioModel.from_pretrained(config['model']).to(device).eval()
	elif config['feature'] == 'whisper':
		config['processor'] = transformers.AutoProcessor.from_pretrained(config['model'])
		if get_attention: config['model'] = transformers.WhisperModel.from_pretrained(config['model'], attn_implementation='eager').to(device).eval()
		else: config['model'] = transformers.WhisperModel.from_pretrained(config['model']).to(device).eval()
	elif config['feature'] == 'wav2vec-LnNor-IPA-ft':
		config['processor'] = transformers.Wav2Vec2Processor.from_pretrained(config['model'])
		if get_attention: config['model'] = transformers.Wav2Vec2ForCTC.from_pretrained(config['model'], attn_implementation='eager').to(device).eval()
		else: config['model'] = transformers.Wav2Vec2ForCTC.from_pretrained(config['model']).to(device).eval()

def load_wav(wav, config):
	"""Load wav file."""
	x, fs = torchaudio.load(wav)
	x = x.mean(dim=0) # to mono (in case stereo)
	if fs != config['sampling_rate']:
		x = torchaudio.functional.resample(x, fs, config['sampling_rate'])
	if config.get('upsample', False): # double the sampling rate so the frame shift is 10ms rather than 20ms
		x = torchaudio.functional.resample(x, orig_freq=config['sampling_rate'], new_freq=2*config['sampling_rate'])
	return x

def get_features(wav, config):
	current_module = sys.modules[__name__]
	func = getattr(current_module, config['load_func'])
	x = func(wav, config) # (T, D)
	if config.get('duplicate_frames', False): x = torch.repeat_interleave(x, repeats=2, dim=0)
	return x

def huggingface_pretrained(wav, config):
	"""Default for pretrained models available on huggingface."""
	with torch.inference_mode():
		x = load_wav(wav, config)
		out = config['model'](input_values=x.unsqueeze(0), output_hidden_states=True)
		layer = str(config.get('hidden_layer', '-1'))
		if layer == 'all':
			hidden = out.hidden_states
			hidden = np.concatenate(hidden, axis=0)
		elif layer == 'all_flat':
			hidden = torch.stack(out.hidden_states).squeeze(1)
			hidden = hidden.transpose(0, 1).reshape(hidden.shape[1], -1) # (N, T, D) -> (T, N*D)
		else:
			layer = layer.split(':')
			if len(layer) == 1: si = int(layer[0]); ei = si + 1
			elif len(layer) > 1: si, ei = layer[0].strip(), layer[1].strip()
			if ei == 0 or ei == '': hidden = out.hidden_states[int(si):]
			else: hidden = out.hidden_states[int(si):int(ei)]
			hidden = torch.vstack(hidden).mean(axis=0)
	return hidden

def huggingface_pretrained_attention(wav, config):
	"""Default for pretrained models available on huggingface."""
	with torch.inference_mode():
		x = load_wav(wav, config)
		out = config['model'](input_values=x.unsqueeze(0), output_hidden_states=True, output_attentions=True)
		layer = int(str(config.get('hidden_layer', '-1')))
		head = int(str(config.get('attention_head', 0)))
		attn = out.attentions[layer].squeeze(0)[head]
	return attn 

def whisper(wav, config):
	"""Whisper"""
	with torch.inference_mode():
		x = load_wav(wav, config)
		inputs = config['processor'](x, sampling_rate=config['sampling_rate'], return_tensors='pt')
		out = config['model'].encoder(input_features=inputs['input_features'], output_hidden_states=True)
		layer = str(config.get('hidden_layer', '-1'))
		valid_frames = int(len(x)/config['sampling_rate'] / 0.02)
		if layer == 'all':
			hidden = out.hidden_states
			hidden = np.concatenate(hidden, axis=0)
			hidden = hidden[:, :valid_frames, :]
		elif layer == 'all_flat':
			hidden = torch.stack(out.hidden_states).squeeze(1)
			hidden = hidden.transpose(0, 1).reshape(hidden.shape[1], -1) # (N, T, D) -> (T, N*D)
			hidden = hidden[:valid_frames, :]
		else:
			layer = layer.split(':')
			if len(layer) == 1: si = int(layer[0]); ei = si + 1
			elif len(layer) > 1: si, ei = layer[0].strip(), layer[1].strip()
			if ei == 0 or ei == '': hidden = out.hidden_states[int(si):]
			else: hidden = out.hidden_states[int(si):int(ei)]
			hidden = torch.vstack(hidden).mean(axis=0)
			hidden = hidden[:valid_frames, :]
	return hidden

def wav2vec_LnNor_IPA_ft(wav, config):
	"""Wav2vec-LnNor-IPA-ft"""
	with torch.inference_mode():
		x, fs = torchaudio.load(wav)
		x = x.mean(dim=0) # to mono (in case stereo)
		if fs != config['sampling_rate']:
			x = torchaudio.functional.resample(x, fs, config['sampling_rate'])
		x = config['processor'](x, sampling_rate=config['sampling_rate'], return_tensors='pt').input_values
		out = config['model'](x, output_hidden_states=True)
		layer = str(config.get('hidden_layer', '-1'))
		if layer == 'logits':
			hidden = out.logits.squeeze(0)
		elif layer == 'all':
			hidden = out.hidden_states
			hidden = np.concatenate(hidden, axis=0)
		else:
			layer = layer.split(':')
			if len(layer) == 1: si = int(layer[0]); ei = si + 1
			elif len(layer) > 1: si, ei = layer[0].strip(), layer[1].strip()
			if ei == 0 or ei == '': hidden = out.hidden_states[int(si):]
			else: hidden = out.hidden_states[int(si):int(ei)]
			hidden = torch.vstack(hidden).mean(axis=0)
	return hidden

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--wav', type=str)
	parser.add_argument('--config', type=str)
	args = parser.parse_args()
	config = load_config(args.config)
	load_model(config)
	h = get_features(args.wav, config)
	print(h.shape)
