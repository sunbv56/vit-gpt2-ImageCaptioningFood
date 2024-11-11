import requests
import csv
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time
import os

# Biến toàn cục để theo dõi các URL đã thấy
seen_urls = set()

# Hàm để lấy dữ liệu từ API
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        return response.json()
    except requests.RequestException as e:
        print(f"Lỗi khi lấy dữ liệu từ {url}: {e}")
        return None

# Hàm để trích xuất thông tin từ dữ liệu JSON
def extract_data(json_data, items_search):
    global seen_urls  # Sử dụng biến toàn cục
    if json_data is None:
        return []
    items = json_data.get('searchResults', {}).get('items', [])
    extracted_data = []
    for item in items:
        title = item.get('title', '')
        img_url = item.get('image', {}).get('url', '')
        if img_url and img_url not in seen_urls:
            extracted_data.append((title, img_url))
            seen_urls.add(img_url)
    return extracted_data

# Hàm để lưu dữ liệu vào tệp CSV
def save_to_csv(data, filename):
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Description', 'Image URL'])
            writer.writerows(data)
    except PermissionError as e:
        print(f"Lỗi quyền truy cập: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong đợi: {e}")

# Số trang cần lấy
num_pages = 100
all_data = []

# Lấy dữ liệu từ các trang
items_search = ['breakfast', 'lunch', 'dinner', 'burger', 'noodle', 'fried', 'seafood', 'egg', 'chicken', 'pizza', 'cocktail', 'coffee', 'cake']
base_url = 'https://www.bbcgoodfood.com/api/search-frontend/search?search='

for item_search in items_search:
    print(f"Đang xử lý URL cơ bản: {base_url+item_search}&page=")
    count_zero = 0
    for page in range(1, num_pages + 1):
        url = base_url + f'{item_search}&page=' + str(page)
        json_data = fetch_data(url)
        # Nếu fetch_data không thành công, dừng vòng lặp
        if json_data is None:
            print(f"Dừng lại do lỗi với URL: {url}")
            continue
        page_data = extract_data(json_data, items_search)
        if len(page_data) == 0:
            count_zero += 1
        else:
            count_zero = 0  # Reset count_zero nếu tìm thấy dữ liệu
        if count_zero == 5:
            print(f"Dừng lại vì không có dữ liệu với URL: {url}")
            break
        all_data.extend(page_data)
        print(f"Lấy dữ liệu trang {page}, Số lượng dữ liệu: {len(page_data)}")
    # Lưu dữ liệu vào tệp CSV
    save_to_csv(all_data, 'FoodCaptionData_bbcgoodfood.csv')
    all_data = []  # Xóa dữ liệu sau khi lưu vào CSV để tránh trùng lặp trong tìm kiếm tiếp theo
print("Dữ liệu đã được lưu vào FoodCaptionData_bbcgoodfood.csv")