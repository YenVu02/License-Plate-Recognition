from flask import Flask

app = Flask(__name__)

@app.route("/plate_number")
def plate_number():
    global detected_plate_number  # Sử dụng biến toàn cục từ chương trình nhận diện
    if detected_plate_number:
        return detected_plate_number
    else:
        return "Không có biển số được nhận diện"

if __name__ == "__main__":
    app.run(debug=True)
