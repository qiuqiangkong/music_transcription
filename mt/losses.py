import torch.nn.functional as F
from einops import rearrange


def lm_and_frame_loss(output_dict: dict, target_dict: dict):
	# LM loss
	out_logits = rearrange(output_dict["token_logits"], 'b l v -> (b l) v')
	tgt_token = rearrange(target_dict["token"], 'b l -> (b l)')
	mask = rearrange(target_dict["token_mask"], 'b l -> (b l)')
	lm_loss = F.cross_entropy(out_logits[mask], tgt_token[mask])

	# Frame loss
	frame_loss = F.binary_cross_entropy(output_dict["onset_roll"], target_dict["onset_roll"])

	# Total loss
	total_loss = lm_loss + frame_loss

	return total_loss