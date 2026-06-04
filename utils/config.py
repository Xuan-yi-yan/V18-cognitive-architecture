# utils/config.py — V18 v4.0 全链路2048D (16x扩容)
import torch, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WORD_LIST_PATH = os.path.join(DATA_DIR, "word_list_v2.txt")
SAVE_DIR = os.path.join(BASE_DIR, "P1_char_word", "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 全链路统一维度 (2048D) ===
POS_DIM = 8                # 位置编码 (不变)
CONTENT_DIM = 2040         # 语义内容
CHAR_DIM = POS_DIM + CONTENT_DIM    # 2048
WORD_DIM = CHAR_DIM                  # 2048
SENT_DIM = CHAR_DIM * 2             # 4096

# === 交叉注意力 (64头, 2048D内部) ===
ATTN_HEADS = 64
ATTN_DIM = CHAR_DIM                # 2048 (与模型维度统一)
ATTN_HEAD_DIM = ATTN_DIM // ATTN_HEADS  # 32
ATTN_DROPOUT = 0.1

# === 调制 ===
MOD_DIM = WORD_DIM                # 2048

# === 解码器隐层 ===
HIDDEN_DIM = 4096

# === 训练 ===
LEARNING_RATE = 0.005
BATCH_SIZE = 2000                # 大批次, 控制单轮计算压力
DEFAULT_EPOCHS = 500
DISPLAY_INTERVAL = 20
PEARSON_EPSILON = 1e-8
WEIGHT_DECAY = 1e-5
DISPLAY_MODE = "average"

# === Early Stopping ===
PATIENCE = 30
