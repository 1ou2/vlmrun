import requests

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(image_url, headers=headers, timeout=10)
if response.status_code == 200:
    with open("test_image.jpg", "wb") as f:
        f.write(response.content)
    print("✅ Image saved as test_image.jpg")
else:
    print("❌ Failed to download image:", response.status_code)