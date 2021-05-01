""" webStreamingApp.py
    Summay: 
        RaspberryPi camera streaming application.
"""

import os
from camera import Camera
from flask import Flask, render_template, Response

def webStreaming():
    """ Web streaming application.

    Returns:
        : Application object.
    """
    return app

app = Flask(__name__)

def gen(camera):
    """ Video streaming generator function.

    Args:
        camera (Camera): Camera class.

    Yields:
        : Camera frame.
    """
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """ Welcom page.

    Returns:
        : inex.html.
    """
    return render_template('index.html')

@app.route('/stream')
def stream():
    """ Video streaming home page.

    Returns:
        : stream.html.
    """
    return render_template('stream.html')

@app.route('/video_feed')
def video_feed():
    """ Video streaming route. Put this in the src attribute of an img tag.

    Returns:
        : camera frame img tag.
    """
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.context_processor
def add_staticfile():
    """ Streaming web page css.
    """
    def staticfile_cp(fname):
        path = os.path.join(app.root_path, 'static', fname)
        mtime =  str(int(os.stat(path).st_mtime))
        return '/static/' + fname + '?v=' + str(mtime)
    return dict(staticfile=staticfile_cp)