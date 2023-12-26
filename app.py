from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from src import data_preprocessing, solution
from PIL import Image

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image and solve Sudoku
            processed_image = data_preprocessing.preprocess_image(file_path)
            largest_contour = data_preprocessing.find_largest_contour(
                processed_image)
            sudoku_grid = data_preprocessing.extract_sudoku_grid(
                processed_image, largest_contour)

            with Image.open(file_path, 'r') as img:
                img = img.resize((550, 550))
                img.save(file_path)
            # file.save(file_path)

            # Save the preprocessed image (sudoku_grid)
            processed_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], 'processed_' + filename)
            image = Image.fromarray(sudoku_grid)
            image = image.convert('RGB')  # Convert to RGB if necessary
            image.save(processed_filename)

            cells = data_preprocessing.split_into_cells(sudoku_grid)
            sudoku_puzzle = data_preprocessing.recognize_digits(cells)
            grid = data_preprocessing.get_sudoku_puzzle(sudoku_puzzle)
            result = solution.solve(grid)

            if isinstance(result, dict):
                return render_template('display.html', filename=filename,
                                       processed_filename=processed_filename,
                                       sudoku_puzzle=sudoku_puzzle,
                                       puzzle=result, solved=True)
            else:
                return render_template('display.html', filename=filename,
                                       processed_filename=processed_filename,
                                       sudoku_puzzle=sudoku_puzzle,
                                       solved=False)

    return render_template('upload.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/images/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
