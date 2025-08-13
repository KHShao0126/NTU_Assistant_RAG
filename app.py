from flask import Flask, render_template, request
from bm25_version1 import search_bm25

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        query = request.form['query']
        result = search_bm25(query)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)