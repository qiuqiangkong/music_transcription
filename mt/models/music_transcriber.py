from torch import Tensor, LongTensor
import torch.nn as nn
from einops import rearrange


def get_model(configs: dict, ckpt_path=None) -> nn.Module:
	n_insts = configs["n_insts"]
	frontend = get_frontend(configs["frontend"])
	encoder = get_encoder(configs["encoder"])
	decoder = get_decoder(configs["decoder"])
	frame_head = get_frame_head(configs["frame_head"])

	return MusicTranscriber(n_insts, frontend, encoder, decoder, frame_head)


def get_frontend(configs: dict) -> nn.Module:
	name = configs["name"]

	if name == "LogMel":
		from mt.models.frontend.logmel import LogMel
		return LogMel(**configs)

	else:
		return ValueError(name)


def get_encoder(configs: dict) -> nn.Module:
	name = configs["name"]

	if name == "CRNN":
		from mt.models.encoders.crnn import CRNN
		return CRNN(**configs)

	# TODO
	elif name == "Conformer":
		return Conformer(**configs)

	# TODO
	elif name == "PerceiverTF":
		return PerceiverTF(**configs)

	else:
		return ValueError(name)


def get_decoder(configs: dict) -> nn.Module:
	name = configs["name"]

	if name == "Transformer":
		from mt.models.decoders.transformer import Transformer
		return Transformer(**configs)

	else:
		return ValueError(name)


def get_frame_head(configs: dict) -> nn.Module:
	name = configs["name"]

	if name == "MLP":
		from mt.models.frame_heads.mlp import MLP
		return MLP(**configs)

	else:
		return ValueError(name)


class MusicTranscriber(nn.Module):
	def __init__(
		self, 
		n_insts: int,
		frontend: nn.Module, 
		encoder: nn.Module, 
		decoder: nn.Module, 
		frame_head: nn.Module
	):
		super().__init__()
		self.n_insts = n_insts
		self.frontend = frontend
		self.encoder = encoder
		self.decoder = decoder
		self.frame_head = frame_head

	def forward(self, in_dict: dict) -> Tensor:
		"""
		b: batch_size
		c: audio_channels
		s: n_insts
		t: n_frames
		f: n_freqs
		d: dims
		l_q: len_query
		l_v: len_value

		Args:
			in_dict: {
				"audio": (b, c, l_audio)
				"token": (b*s, l)
				"self_attn_mask": (b*s, 1, l_q, l_q)
				"cross_attn_mask": None | (b*s, 1, l_q, l_v)
			}

		Returns:
			out_dict: {
				"token_logits": (b*s, l)
				"frame_prob": (b*s, l_frame)
			}
		"""
		x = self.frontend(in_dict["audio"])  # (b, c, t, f)
		
		x = self.encoder(x)  # (b, t, d)

		# To individual instrument latents
		latent = rearrange(x, 'b t (s d) -> (b s) t d', s=self.n_insts)

		# Token decoder
		token_logits = self.decoder(
			token=in_dict["token"], 
			seq=latent, 
			self_attn_mask=in_dict["self_attn_mask"],
			cross_attn_mask=in_dict["cross_attn_mask"]
		)  # (b*s, l, n_vocab)
		
		# Frame decoder
		onset_roll = self.frame_head(latent)  # (b*s, t, p)
		
		# Output
		out_dict = {
			"token_logits": token_logits,
			"onset_roll": onset_roll,
		}
		
		return out_dict
