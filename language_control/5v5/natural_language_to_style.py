import time
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer, Qwen2Model


class StyleInterpreterModel(nn.Module):
    def __init__(self, cand_info_dict, gpu):
        super(StyleInterpreterModel, self).__init__()
        self.cand_info_dict = cand_info_dict
        self.gpu = gpu
        self.basemodel = Qwen2Model.from_pretrained("Qwen/Qwen2.5-0.5B")

        self.out_features = 896
        self.hidden_size = 128

        self.ln_query = nn.LayerNorm(self.out_features)

        # 投影层
        self.projection_q = nn.Linear(self.out_features, self.hidden_size)

        self.adaLN_modulation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.out_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.ln_final = nn.LayerNorm(self.hidden_size)
        self.final_out = nn.Linear(self.hidden_size, 10)

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_out.weight, 0)
        nn.init.constant_(self.final_out.bias, 0)

    def modulate(self, x, param):
        # 例如[output_dim, 2]，其中param[:, 0]是shift，param[:, 1]是scale
        shifts = param[:, 0].unsqueeze(0)  # [1, output_dim]
        scales = param[:, 1].unsqueeze(0)  # [1, output_dim]

        # 向量化操作，避免循环和就地修改
        result = x * (1 + scales) + shifts

        return result

    def forward(self, input_ids, attention_mask, last_token_position):
        # 编码查询序列
        query_outputs = self.basemodel(input_ids, attention_mask=attention_mask)
        query_hidden_states = query_outputs[0]

        # 提取句子表示 (最后一个token或指定位置)
        last_token_position = last_token_position.unsqueeze(-1).expand(-1, -1, query_hidden_states.size(-1))
        query_sent_repr = torch.gather(query_hidden_states, 1, last_token_position)
        query_sent_repr = query_sent_repr.squeeze(1)  # [batch, hidden]

        # 应用层标准化和投影
        query_sent_repr = self.ln_query(query_sent_repr)
        query_sent_repr = self.dropout(query_sent_repr)
        query_sent_repr = self.projection_q(query_sent_repr)  # [batch, hidden_size]
        query_sent_repr = self.relu(query_sent_repr)

        # prompt instruction process
        candidate_ids = self.cand_info_dict['input_ids'].cuda(self.gpu)
        candidate_mask = self.cand_info_dict['attn_masks'].cuda(self.gpu)
        candidate_last_token_pos = self.cand_info_dict['last_token_pos'].cuda(self.gpu)

        candidate_outputs = self.basemodel(candidate_ids, attention_mask=candidate_mask)
        candidate_hidden_states = candidate_outputs[0]

        cand_last_token_pos = candidate_last_token_pos.unsqueeze(-1).expand(-1, -1, candidate_hidden_states.size(-1))
        cand_sent_repr = torch.gather(candidate_hidden_states, 1, cand_last_token_pos)
        cand_sent_repr = cand_sent_repr.squeeze(1)  # [cand_num, hidden]
        cand_sent_repr = self.adaLN_modulation(cand_sent_repr)  # [cand_num, 2]

        query_sent_repr = self.ln_final(query_sent_repr)
        query_sent_repr = self.dropout(query_sent_repr)
        scores = self.final_out(query_sent_repr)
        scores = self.modulate(scores, cand_sent_repr)

        return scores.detach().cpu().numpy()


def test(model, input_ids, attn_masks, last_token_pos, gpu):
    """测试模型并输出结果"""
    model.eval()

    with torch.no_grad():
        begin_time = time.time()

        seq, attn_masks, last_token_pos = \
            input_ids.cuda(gpu), attn_masks.cuda(gpu), last_token_pos.cuda(gpu)

        # 获取预测和损失
        logits = model(seq, attn_masks, last_token_pos)
        inference_time = (time.time() - begin_time) / len(logits[0])
        print('inference_time: ', inference_time)

    return logits


def initialize_style_candidate_dict(tokenizer_name):
    """初始化风格候选字典（对style相关模型）"""
    if 'bert' in tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    texts = [
        'The importance of winning; the higher the value, the more the team desires to win. A value of 0 means they do not care about winning at all.',
        'The importance of scoring goals; the higher the value, the more the team desires to score. A value of 0 means they do not care about scoring at all.',
        'The importance of not conceding goals; the higher the value, the less the team wants to concede. A value of 0 means they do not care about conceding goals at all.',
        'The importance of individual ball control; a value of 0 means they do not intentionally hold the ball but decide based on the situation and other tendencies, while a value of 10 means they prefer to hold the ball individually.',
        'Regaining possession; a value of 0 means they do not care about regaining or losing possession, while a value of 10 means they place great importance on regaining possession from the opponent and minimizing loss of possession.',
        'Emphasis on passing; a value of 0 means they do not pass intentionally but decide based on the situation on the field and other tendencies, while a value of 10 means they prefer to pass.',
        'Relative distance between players; a value of 0 means the team is compact with small relative distances, while a value of 10 means the team has larger relative distances and is more spread out.',
        'The overall positioning of the team on the field; a value of 0 means the team is closer to their own goal line, while a value of 10 means the team is closer to the opponent\'s goal line.',
        'Shooting distance; a value of -1 means there is no tendency regarding shooting distance, 0 means they prefer to shoot from close range, and 1 means they prefer to shoot from long range.',
        'Dribbling style; a value of -1 means there is no tendency regarding dribbling style, 0 means they prefer a moderate running speed while dribbling, making it moderately easy to be dispossessed, 1 means they prefer to dribble closely while moving slower and being harder to dispossess, and 2 means they prefer to sprint past opponents quickly while dribbling but are more easily dispossessed.'
    ]

    input_ids_list = []
    attn_masks_list = []
    last_token_pos_list = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids_list.append(encoding['input_ids'])
        attn_masks_list.append(encoding['attention_mask'])
        last_token_pos_list.append((encoding['attention_mask'].sum(-1) - 1).reshape(1, -1))

    input_ids_tensor = torch.cat(input_ids_list, 0)
    attn_masks_tensor = torch.cat(attn_masks_list, 0)
    last_token_pos_tensor = torch.cat(last_token_pos_list, 0)

    return {
        'input_ids': input_ids_tensor,
        'attn_masks': attn_masks_tensor,
        'last_token_pos': last_token_pos_tensor
    }


class StyleInterpreterPredictor:
    def __init__(self, model_path, gpu=0, seed=0):
        """初始化预测器，加载模型"""
        # 设置随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.gpu = gpu
        self.tokenizer_name = "Qwen/Qwen2.5-0.5B"

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # 加载模型
        cand_info_dict = initialize_style_candidate_dict(self.tokenizer_name)
        self.model = StyleInterpreterModel(cand_info_dict, gpu)

        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path))

        # 使用CUDA
        self.model = self.model.cuda(gpu)

    def predict(self, text):
        """对输入文本进行预测"""
        # 文本编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids']
        attn_masks = encoding['attention_mask']
        last_token_position = (attn_masks.sum(-1) - 1).reshape(1, -1)

        # 模型推理
        predictions = test(self.model, input_ids, attn_masks, last_token_position, self.gpu)

        predictions = predictions[0]
        predictions = np.round(predictions, 2)

        predictions[8] = np.round(predictions[8])
        predictions[9] = np.round(predictions[9])

        factors = self.predictions_to_factor(predictions)
        return factors

    def predictions_to_factor(self, predictions):
        factors = {
        'win': 0.5,
        'goal': 0.5,
        'lose_goal': 0.5,
        'hold_ball': 0.5,
        'get_possession': 0.5,
        'pass': 0.5,
        'active_area_x': [0, 0, 0],
        'active_area_y': [0, 0, 0],
        'spacing': 0.5,
        'shot': [1, 1],
        'move': [1, 1, 1],
        'formation': 0.5,
        }
        
        predictions_order = [
            'win', 'goal', 'lose_goal', 'hold_ball', 
            'get_possession', 'pass', 'spacing', 
            'shot', 'move', 'formation'
        ]
        
        for i, factor_name in enumerate(predictions_order):
            if factor_name in ['shot', 'move']:
                if predictions[i] < len(factors[factor_name]) and predictions[i] >= 0:
                    tmp_style = [0.0] * len(factors[factor_name])
                    tmp_style[int(predictions[i])] = 1
                    factors[factor_name] = tmp_style
            else:
                noise = 0
                factor_value = np.clip(predictions[i] / 10 + noise, 0, 1)
                factors[factor_name] = factor_value

        return factors

def main(text, gpu=0, seed=0, model_path=''):
    """保持向后兼容的主函数"""
    predictor = StyleInterpreterPredictor(model_path, gpu, seed)
    return predictor.predict(text)


if __name__ == "__main__":
    text = "Move forward with controlled aggression, ensuring precision and composure in every attack to maximize scoring opportunities."
    # 使用类的方式
    predictor = StyleInterpreterPredictor(model_path='E:\\work\\LCDSP_GRF_inference_code\\lcdsp_grf_inference_code\\language_models\\trained_model.pth')  # 需要提供实际模型路径
    for i in range(100):
        predictions = predictor.predict(text)
        print(predictions)