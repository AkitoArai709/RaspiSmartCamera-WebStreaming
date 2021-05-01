#!/usr/bin/env python
""" main.py
    Summay: 
        RaspberryPi camera streaming application.
"""

from webStreamingApp import webStreaming 

def main():
    webStreamingApp = webStreaming()
    webStreamingApp.run(host='0.0.0.0', threaded=True, debug=True)

if __name__ == '__main__':
    main()