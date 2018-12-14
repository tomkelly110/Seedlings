import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template
from flask import send_from_directory
#import one_image
import vgg1999
import training

UPLOAD_FOLDER = '/Users/y/web/uploads'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/show/<name>")#name没有指定类型，使用的是默认的类型，string
def show(name):
    return name


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return training.evaluate_one_image()
            return vgg1999.accuracy()
            # return 'eee'
            #return one_image.accuracy()
            #return redirect(url_for('uploaded_file',filename=filename)),training.evaluate_one_image()
    return render_template("upload.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


@app.route('/output', methods=['post'])
def output():
    return render_template('home.html')

@app.route('/upload1', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return training.evaluate_one_image()
            #return vgg1999.accuracy()
            #return one_image.accuracy()
            #return redirect(url_for('uploaded_file',filename=filename)),training.evaluate_one_image()
    return render_template("upload1.html")

if __name__ == '__main__':
 app.run(debug=True)
