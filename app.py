from flask import Flask, render_template, request
import os
from color import predict_colors, load_model, load_color_chart

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and color chart once at startup
model, device = load_model(r'C:\Users\rkyam\Desktop\task\corn_deeplab_v3.pth')
color_chart = load_color_chart(r'C:\Users\rkyam\Desktop\task\RHS + colour names.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    colors = None
    image_path = None
    if request.method == 'POST':
        img = request.files['image']
        if img:
            img_path = os.path.join(UPLOAD_FOLDER, img.filename)
            img.save(img_path)
            colors = predict_colors(img_path, model, color_chart, device)
            # Fix path for HTML display (convert Windows \ to /)
            image_path = '/' + img_path.replace('\\', '/')
    return render_template('index.html', colors=colors, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
