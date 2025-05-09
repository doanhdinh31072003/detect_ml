import numpy as np
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 50 văn bản tự nhiên
natural_texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is powerful and fun",
    "Natural language processing helps computers understand human language",
    "Artificial intelligence is transforming industries",
    "Data science combines statistics and programming",
    "Reading books expands knowledge and imagination",
    "The sun rises in the east and sets in the west",
    "Music is a universal language",
    "Rainbows appear after rain when sunlight reflects",
    "He went to the market to buy fresh vegetables",
    "Education is the key to success",
    "Traveling opens your mind to new cultures",
    "He reads the newspaper every morning",
    "The garden is full of beautiful flowers",
    "She enjoys painting landscapes during weekends",
    "Honesty is the best policy",
    "The children are playing in the park",
    "Birds fly high in the sky",
    "Winter is colder than summer",
    "A healthy diet improves well-being",
    "Libraries are quiet places to study",
    "He always drinks coffee in the morning",
    "Technology evolves rapidly",
    "She is writing a new novel",
    "This cake tastes delicious",
    "The river flows gently through the valley",
    "We watched a documentary on space",
    "Students must complete their homework on time",
    "The city was silent at night",
    "There are many stars in the galaxy",
    "They planted trees along the roadside",
    "Her handwriting is neat and elegant",
    "The plane landed safely at the airport",
    "He enjoys jogging in the early morning",
    "Reading helps develop vocabulary",
    "The artist displayed her paintings at the gallery",
    "He fixed the broken chair with glue",
    "People gathered to celebrate the festival",
    "The train arrived five minutes late",
    "Sunlight provides vitamin D",
    "My dog loves to chase butterflies",
    "The baby smiled at her mother",
    "She wore a blue dress to the party",
    "Cooking at home is healthier and cheaper",
    "The scientist conducted several experiments",
    "The moon glowed brightly in the night sky",
    "Children must be taught kindness and empathy",
    "He listened to classical music while studying",
    "A good sleep schedule improves focus",
    "They walked through the forest trail"
]

# 50 văn bản giấu tin (chứa dấu hiệu như nhiều khoảng trắng bất thường, cấu trúc lặp, hoặc mẫu không tự nhiên)
stego_texts = [
    "This    is  a   secret    message",
    "Hello  world  this   is    stego   text",
    "Hiding   info     in   spaces",
    "This  sentence  is    not   normal",
    "Every   space   hides something",
    "Too    many  spaces    here",
    "This  is     an   embedded    message",
    "Notice  the     gaps  in  spacing",
    "Why     is  this   text   so    weird",
    "A  message    may     be   here",
    "Double   space   can   be  a   sign",
    "Extra   gaps    are    suspicious",
    "H e l l o   w o r l d",
    "White   space    steganography",
    "Hidden      in     plain    sight",
    "The   fox    hides     info",
    "Text      may     contain     secrets",
    "Every     extra     space     matters",
    "Some    letters     are    hidden",
    "T  h  i  s     t  e  x  t     i  s     s  u  s",
    "This   is  not   what   it   seems",
    "Hidden     message    inside    text",
    "Find     the     anomaly",
    "Spacing     is     everything",
    "Look   closer     at    the     structure",
    "This      could     be     nothing",
    "Maybe      it's     something",
    "Odd     formatting     detected",
    "Spacing    game    is    on",
    "Not     your     average     sentence",
    "Does     this     look     normal",
    "Suspect     the     unusual     spaces",
    "Observe     the     weirdness",
    "This     is     not     obvious",
    "Everything     is     deliberate",
    "Find      the      oddity",
    "Structured      noise      detected",
    "Maybe      there's      a      key",
    "Invisible     ink     between     words",
    "Can     you     see     the     pattern",
    "Message      within      message",
    "Spaces     tell     the     truth",
    "Looks     natural     but     isn't",
    "Spot      the      difference",
    "Patterns      in     punctuation",
    "Subtle      spacing      clues",
    "Encryption      through      space",
    "Hidden      trail      of      blanks",
    "A      spacey      message",
    "Hacked       text       revealed"
]

# Gộp lại và tạo nhãn
texts = natural_texts + stego_texts
labels = [0] * len(natural_texts) + [1] * len(stego_texts)

# Hàm trích xuất đặc trưng
def extract_features(text):
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    whitespace_ratio = text.count(' ') / len(text) if len(text) > 0 else 0
    double_spaces = text.count("  ")

    num_letters = sum(c.isalpha() for c in text)
    letter_ratio = num_letters / len(text) if len(text) > 0 else 0

    first_letters = [w[0].lower() for w in words if w]
    first_letter_counts = [first_letters.count(ch) for ch in string.ascii_lowercase]
    most_common_first_letter = max(first_letter_counts) if first_letter_counts else 0

    return [
        avg_word_len,
        whitespace_ratio,
        double_spaces,
        letter_ratio,
        most_common_first_letter
    ]

# Chuẩn bị dữ liệu
X = np.array([extract_features(text) for text in texts])
y = np.array(labels)

# Chia dữ liệu huấn luyện / kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Huấn luyện mô hình
clf = RandomForestClassifier()
clf.fit(X_train, y_train)


# Nhập văn bản từ người dùng để dự đoán
print("\nNhập văn bản để kiểm tra (tự nhiên hoặc giấu tin):")
user_input = input(">> ")

# Trích xuất đặc trưng và dự đoán
user_features = np.array([extract_features(user_input)])
prediction = clf.predict(user_features)[0]
probs = clf.predict_proba(user_features)[0]

# Kết quả
label_map = {0: "TỰ NHIÊN", 1: "GIẤU TIN"}
print(f"\n🧠 Kết quả dự đoán: {label_map[prediction]}")
print(f"🔍 Xác suất:")
print(f" - Tự nhiên (0): {probs[0]*100:.2f}%")
print(f" - Giấu tin (1): {probs[1]*100:.2f}%")  