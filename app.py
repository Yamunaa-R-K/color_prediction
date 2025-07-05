from flask import Flask, render_template, request
import os
from color import predict_colors, load_model, load_color_chart

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and color chart - relative path inside project
model, device = load_model('corn_deeplab_v3.pth')
color_chart = load_color_chart('RHS + colour names.csv')

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
            image_path = '/' + img_path.replace('\\', '/')
    return render_template('index.html', colors=colors, image_path=image_path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Render provides PORT environment variable
    app.run(host='0.0.0.0', port=port)
