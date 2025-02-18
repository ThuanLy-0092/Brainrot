# Sử dụng image Python có sẵn
FROM python:3.12

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 imagemagick fonts-dejavu ttf-mscorefonts-installer \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*
    
# Sửa policy của ImageMagick để xử lý ảnh
RUN sed -i 's/none/read,write/g' /etc/ImageMagick-6/policy.xml

# Cài đặt virtualenv
RUN pip install --upgrade pip
RUN pip install virtualenv

# Tạo môi trường ảo (virtual environment)
RUN python -m venv /venv

# Sử dụng môi trường ảo (activate)
ENV PATH="/venv/bin:$PATH"

# Đặt thư mục làm việc
WORKDIR /app

# Copy tất cả file vào container
COPY . .

# Cài đặt các thư viện Python cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Cấp quyền cho thư mục
RUN chmod -R 777 /app

# Chạy Streamlit
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
