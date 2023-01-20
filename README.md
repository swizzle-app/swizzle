<p align=center>
    <a href="www.google.com" target="_blank"><img src="app/media/swizzle_logo_light.png" alt="swizzle-logo"></a>
</p>

---

<p align=center>
<b>AI generated music notation from guitar recordings</b>
</p>
<p align=center>
<img src="https://img.shields.io/static/v1?label=version&message=1.0&color=blueviolet" alt="version=1.0">
    <img src="https://img.shields.io/static/v1?label=python&message=v3.9.8&color=g" alt="python=3.9.8+">
    <img src="https://img.shields.io/static/v1?label=license&message=MIT&color=blue" alt="license=MIT">
</p>

<p align=center>
    <a href="#intro">Intro</a>&nbsp;&bullet;&nbsp;
    <a href="#bts">Behind the scenes</a>&nbsp;&bullet;&nbsp;
    <a href="#use">How To Use</a>&nbsp;&bullet;&nbsp;
    <a href="#ref">Credits</a>
</p>

<a id="intro"></a>

# Intro
Guitar players know it all too well: you have this idea you want to share with others, but in order for them also being able to play it, you need to transcribe it.

This not only takes time, but it can also kill the creative process.

**Well. Now there's an app for that!**

*swizzle*, a verbification of the combination of *song-writing* and *wizard*, listens to you jam and employs a convolutional neural network (CNN) to transcribe tabs you can share!

**And all this, easily done from your browser.**

<a id="bts"></a>

# Workflow of swizzle
Here is how it actually works:

1. **Upload audio**: After uploading an audio file (a guitar recording), it is transformed into a numeric representation (constant-Q transformation) that can be thought of as an image (i.e. a spectrogram).
1. **Transform data**: This image is split into multiple images, the exact number varies depending on how long the audio file is.
1. **Smart transcription**: On all of these images, the swizzle model (a CNN) will detect and classify the notes played.
1. **Output tabs**: After post-processing to bring the predictions into the correct shape, the frontend shows the output tabs.

<a id="use"></a>

# How to use swizzle

There are two ways to use our app.

## Simply visit our web-app
We deployed our web-app and you can find it [here](www.google.com)!

## Clone our repo and set it up on your machine

:information_source: Note: you will need git for this. You can download it <a href="https://www.git-scm.com">here</a>

1. **Clone repo and install dependencies in a local environment**: Just run these commands in your terminal.


        # clone this repository
        git clone https://github.com/swizzle-app/swizzle

        # go into repository
        cd swizzle

        # setup virtual environment
        # and install dependencies
        make setup

        # activate local environment
        source .venv/bin/activate

2. **Download the dataset**: download the dataset (if you want to use it, e.g. for training) and extract it in the repository under "data/raw/". The dataset can be downloaded [here](https://guitarset.weebly.com).

3. **Run the app**: simply navigate to the "app" folder in your terminal and run the command:

        streamlit run frontend.py

    a browser window will open, running the app.

4. **Have fun!**

    
<a id="ref"></a>

# Credits
The data was created and published by [Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "​Guitarset: A Dataset for Guitar Transcription", in 19th International Society for Music Information Retrieval Conference, Paris, France, Sept. 2018.](https://guitarset.weebly.com/uploads/1/2/1/6/121620128/xi_ismir_2018.pdf) and can be downloaded [here](https://guitarset.weebly.com).

This project was lead by Qingyang Xi at NYU's Music and Audio Research Lab, along with Rachel Bittner, Xuzhou Ye and Juan Pablo Bello from the same lab, as well as Johan Pauwels at the Center for Digital Music at Queen Mary University of London.
