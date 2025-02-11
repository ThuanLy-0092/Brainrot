# Sử dụng image Python có sẵn
FROM python:3.12

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 imagemagick fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Sửa policy của ImageMagick để xử lý ảnh
RUN sed -i 's/none/read,write/g' /etc/ImageMagick-6/policy.xml

# Đặt thư mục làm việc
WORKDIR /app

# Copy tất cả file vào container
COPY . .

# Cài đặt các thư viện Python cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Chạy Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
