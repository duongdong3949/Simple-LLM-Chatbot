# Chatbot hỗ trợ đặt món ăn online

Một chatbot thông minh sử dụng RAG (Retrieval-Augmented Generation) để hỗ trợ khách hàng đặt món ăn online. Dự án sử dụng LangChain, Hugging Face models và ChromaDB để tạo ra trải nghiệm tương tác tự nhiên.

## 🚀 Tính Năng

- **Trợ lý thực đơn thông minh**: Trả lời câu hỏi về món ăn, giá cả, thời gian chuẩn bị
- **Tìm kiếm ngữ nghĩa**: Sử dụng vector embeddings để tìm kiếm món ăn liên quan
- **Trí nhớ cuộc trò chuyện**: Ghi nhớ ngữ cảnh cuộc hội thoại
- **Hỗ trợ tiếng Việt**: Giao diện và phản hồi hoàn toàn bằng tiếng Việt
- **Tùy chọn món ăn**: Hướng dẫn chi tiết về các tùy chọn và giá

## 🛠️ Công Nghệ Sử Dụng

- **LangChain**: Framework cho ứng dụng LLM
- **Hugging Face**: 
  - Model: Qwen2.5-3B-Instruct (text generation)
  - Embeddings: paraphrase-multilingual-mpnet-base-v2
- **ChromaDB**: Vector database cho RAG
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library

## 📋 Yêu Cầu Hệ Thống

- Python 3.8+
- CUDA (khuyến nghị cho GPU acceleration)
- RAM: Tối thiểu 8GB (khuyến nghị 16GB+)
- VRAM: Tối thiểu 4GB (nếu sử dụng GPU)
- **NVIDIA Container Toolkit** (nếu sử dụng Docker với GPU)

## 🔧 Cài Đặt NVIDIA Container Toolkit 

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 🚀 Sử Dụng

### Chạy chatbot
```bash
python main.py
```

### Sử dụng Docker
```bash
# Build image
docker build -t chatbot .

# Chạy container
docker run --gpus all -it chatbot
```

## 💬 Cách Sử Dụng

1. Khởi động chương trình
2. Nhập câu hỏi về thực đơn (ví dụ: "Phở bò giá bao nhiêu?")
3. Chatbot sẽ trả lời dựa trên thông tin thực đơn
4. Gõ `quit` để thoát

### Ví dụ câu hỏi:
- "Phở bò có những tùy chọn gì?"
- "Món nào có giá dưới 50,000 đồng?"
- "Bún chả Hà Nội mất bao lâu để chuẩn bị?"
- "Có món chay nào không?"

## 📁 Cấu Trúc Dự Án

```
LLM/
├── main.py                    # Entry point
├── chat_bot.py               # Core chatbot logic
├── setup_knowledge_base.py   # Knowledge base setup
├── menu.json                 # Restaurant menu data
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── chroma_db/              # Vector database (auto-generated)
└── README.md               # This file
```

## 🔧 Cấu Hình

### Model Configuration
- **Text Generation Model**: Qwen2.5-3B-Instruct (mặc định)
- **Embedding Model**: paraphrase-multilingual-mpnet-base-v2
- **Max Tokens**: 512
- **Retrieval**: Top 3 similar documents

### 🚀 Nâng Cấp Model (Khuyến Nghị)

Nếu tài nguyên GPU cho phép, có thể nâng cấp lên các model lớn hơn để có trải nghiệm chatbot thông minh hơn:

#### Model Text Generation (trong `chat_bot.py`):
```python
# Model hiện tại (3B parameters)
model_name = "Qwen/Qwen2.5-3B-Instruct"

# Model 7B parameters - cần ~14GB VRAM
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Model 14B parameters - cần ~28GB VRAM
model_name = "Qwen/Qwen2.5-14B-Instruct"

### Memory Settings
- **Memory Type**: ConversationBufferMemory
- **Context Window**: Full conversation history

## 📊 Dữ Liệu Thực Đơn

File `menu.json` chứa thông tin chi tiết về:
- Tên món ăn và mô tả
- Giá cơ bản
- Thời gian chuẩn bị
- Các tùy chọn và giá thêm
- Phân loại theo danh mục

## 🔍 RAG Pipeline

1. **Document Processing**: Chuyển đổi menu.json thành documents
2. **Embedding**: Tạo vector embeddings cho mỗi món ăn
3. **Storage**: Lưu trữ trong ChromaDB
4. **Retrieval**: Tìm kiếm documents liên quan
5. **Generation**: Tạo phản hồi với LLM

## 🚨 Lưu Ý

- Lần đầu chạy sẽ mất thời gian để download models
- Cần kết nối internet để download models từ Hugging Face
- ChromaDB sẽ được tạo tự động trong thư mục `chroma_db/`
- **NVIDIA Container Toolkit** là bắt buộc nếu sử dụng Docker với GPU support

## 💡 Lời Khuyên Tối Ưu Hóa

### 🎯 Chọn Model Phù Hợp:
- **GPU yếu (4-8GB VRAM)**: Sử dụng model mặc định (3B parameters)
- **GPU trung bình (8-16GB VRAM)**: Nâng cấp lên Qwen2.5-7B-Instruct
- **GPU mạnh (16GB+ VRAM)**: Sử dụng Qwen2.5-14B-Instruct hoặc lớn hơn
- **Chỉ có CPU**: Sử dụng model nhỏ hơn hoặc quantization

### ⚡ Tăng Hiệu Suất:
- Model lớn hơn = Chatbot thông minh hơn, phản hồi chính xác hơn
- Sử dụng GPU sẽ tăng tốc độ xử lý đáng kể
- Quantization (4-bit/8-bit) giúp giảm VRAM usage

**Lưu ý**: Dự án này sử dụng các model AI có thể yêu cầu tài nguyên hệ thống đáng kể. Đảm bảo hệ thống đáp ứng yêu cầu tối thiểu trước khi chạy.
