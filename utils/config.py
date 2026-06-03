# utils/config.py — V18 v3.0 全链路128D统一 (无投影层)
import torch, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WORD_LIST_PATH = os.path.join(DATA_DIR, "word_list.txt")
SAVE_DIR = os.path.join(BASE_DIR, "P1_char_word", "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 全链路统一维度 (128D) ===
POS_DIM = 8           # 位置编码 (sin/cos, 不可压缩)
CONTENT_DIM = 120     # 语义内容
CHAR_DIM = POS_DIM + CONTENT_DIM    # 128
WORD_DIM = CHAR_DIM                  # 128 (词=字, 物理统一)
SENT_DIM = CHAR_DIM * 2             # 256

# === 交叉注意力 (64头) ===
ATTN_HEADS = 64
ATTN_DIM = 256                    # Q,K,V 投影维度
ATTN_HEAD_DIM = ATTN_DIM // ATTN_HEADS  # 4
ATTN_DROPOUT = 0.1

# === 调制 ===
MOD_DIM = WORD_DIM                # 128

# === 解码器隐层 ===
HIDDEN_DIM = 256

# === 训练 ===
LEARNING_RATE = 0.01
BATCH_SIZE = 32
DEFAULT_EPOCHS = 300
DISPLAY_INTERVAL = 40
PEARSON_EPSILON = 1e-8
WEIGHT_DECAY = 1e-5
DISPLAY_MODE = "average"

# === Early Stopping ===
PATIENCE = 20
