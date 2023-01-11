<p align=center>
    <a href="www.google.com" target="_blank"><img src="media/logo.png" alt="swizzle-logo"></a>
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
    <a href="#intro">Intro</a>
    &nbsp;&bullet;&nbsp;<a href="#use">Behind the scenes</a>
    &nbsp;&bullet;&nbsp;<a href="#use">How To Use</a>
    &nbsp;&bullet;&nbsp;<a href="#ref">Credits</a>
</p>

<a id="intro"></a>

# Intro
Guitar players know it all too well: you have this idea you want to share with others, but in order for them also being able to play it, you need to transcribe it.

This not only takes time, but it can also kill the creative process.

**Well. Now there's an app for that!**

*swizzle*, a verbification of the combination of *song-writing* and *wizard*, listens to you jam and employs a convolutional neural network (CNN) to transcribe tabs you can share!

**And all this, easily done from your browser.**

<a id="bhs"></a>

# Behind the scenes
Here is how it actually works:

1. After uploading an audio file, it is transformed into a numeric representation (constant-Q transformation) that can be thought of as an image (i.e. a spectrogramm).
1. This image is split into multiple images, the exact number varies depending on how long the audio file is.
1. On all of these images, the CNN will detect the notes played.
1. After some post-processing to bring the predictions into the correct shape, the frontend shows the output.

<a id="use"></a>

# How to use

There are two ways to use our app.

1. Simply visit our web-app [here](www.google.com)
1. Or clone our repo and set it up on your machine (Note: you will need git for this. You can download it [here](https://www.git-scm.com))

        # clone this repository
        $ git clone https://github.com/swizzle-app/swizzle

        # go into repository
        $ cd swizzle

        # create local environment
        $ pyenv local 3.9.8
        $ python -m venv .venv

        # activate local environment
        $ source .venv/bin/activate

        # install dependencies
        $ make setup
    Finally, you will need to download the dataset (if you want to use it) and extract it in the repository under "data/raw/". The dataset can be downloaded [here](https://guitarset.weebly.com).

    
<a id="ref"></a>

# Credits
The data was created and published by [Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "​Guitarset: A Dataset for Guitar Transcription", in 19th International Society for Music Information Retrieval Conference, Paris, France, Sept. 2018.](https://guitarset.weebly.com/uploads/1/2/1/6/121620128/xi_ismir_2018.pdf) and can be downloaded [here](https://guitarset.weebly.com).

This project was lead by Qingyang Xi at NYU's Music and Audio Research Lab, along with Rachel Bittner, Xuzhou Ye and Juan Pablo Bello from the same lab, as well as Johan Pauwels at the Center for Digital Music at Queen Mary University of London.