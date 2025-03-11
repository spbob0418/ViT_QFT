import torch
import re

def rename_keys(checkpoint, is_transfer_learning):
    """ 체크포인트의 키를 모델 키와 맞춰서 변환 """
    if is_transfer_learning: 
        for k in ['classifier.weight', 'classifier.bias']:
            if k in checkpoint:
                del checkpoint[k]

    key_mapping = {}
    qkv_weights = {}
    qkv_biases = {}

    print("🔍 Key Mapping Start...")

    for i, old_key in enumerate(checkpoint.keys()):  # 루프 카운트 추가
        
        new_key = old_key

        # Embeddings 변환
        new_key = new_key.replace("vit.embeddings.cls_token", "cls_token")
        new_key = new_key.replace("vit.embeddings.position_embeddings", "pos_embed")
        new_key = new_key.replace("vit.embeddings.patch_embeddings.projection.weight", "patch_embed.proj.weight")
        new_key = new_key.replace("vit.embeddings.patch_embeddings.projection.bias", "patch_embed.proj.bias")
        
        # Transformer Blocks 변환
        match = re.match(r'vit\.encoder\.layer\.(\d+)\.(.*)', new_key)
        if match:
            layer_num, sub_key = match.groups()
            layer_num = int(layer_num)

            # qkv를 합쳐야 하므로 query, key, value를 별도로 저장
            if "attention.attention.query.weight" in sub_key:
                qkv_weights.setdefault(layer_num, {})["query"] = checkpoint[old_key]
                continue
            if "attention.attention.key.weight" in sub_key:
                qkv_weights.setdefault(layer_num, {})["key"] = checkpoint[old_key]
                continue
            if "attention.attention.value.weight" in sub_key:
                qkv_weights.setdefault(layer_num, {})["value"] = checkpoint[old_key]
                continue

            if "attention.attention.query.bias" in sub_key:
                qkv_biases.setdefault(layer_num, {})["query"] = checkpoint[old_key]
                continue
            if "attention.attention.key.bias" in sub_key:
                qkv_biases.setdefault(layer_num, {})["key"] = checkpoint[old_key]
                continue
            if "attention.attention.value.bias" in sub_key:
                qkv_biases.setdefault(layer_num, {})["value"] = checkpoint[old_key]
                continue

            sub_key = sub_key.replace("attention.output.dense", "attn.proj")
            sub_key = sub_key.replace("intermediate.dense", "mlp.fc1")
            sub_key = sub_key.replace("output.dense", "mlp.fc2")
            sub_key = sub_key.replace("layernorm_before", "norm1")
            sub_key = sub_key.replace("layernorm_after", "norm2")
            new_key = f"blocks.modules_list.{layer_num}.{sub_key}"

        # LayerNorm 변환
        new_key = new_key.replace("vit.layernorm.weight", "norm.weight")
        new_key = new_key.replace("vit.layernorm.bias", "norm.bias")

        if not is_transfer_learning: 
            print("head 넘기기")
            new_key = new_key.replace("classifier.weight", "head.weight")
            new_key = new_key.replace("classifier.bias", "head.bias")

        key_mapping[old_key] = new_key  

    return key_mapping, qkv_weights, qkv_biases
