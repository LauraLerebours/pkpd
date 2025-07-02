from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import shafer

app = Flask(__name__)







if __name__ == '__main__':
    app.run(debug=True)