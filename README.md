# ScribeSmith

## Abstract

The [report](docs/report/report.pdf) outlines an implementation of a **Generative Adversarial Network** that attempts to generate realistic-looking images of handwritten text from ASCII text input along with a **text Recognizer** model. We present architecture, training, data, and a discussion of our model's qualitative and quantitative results on different datasets. The IAM handwriting Database is processed for training and validation, while the model is tested on a set of randomly generated and handwritten Shakespeare text. We also prepare baseline models for the recognizer and generator to gauge our primary model's performance. We also utilize **Fréchet Inception Distance** and **Character Error Rate** metrics to evaluate different sub-networks within the GAN. The final result of the project is a Generator that does not perform as expected, and a successful Recognizer. Potential ethical considerations for this project are also addressed.

| ![high-lvl-overview.png](docs/report/Figs/high-lvl-overview.png) |
|:--:|
| *High-level overview of the ScribeSmith model* |

## Contributors

The group agrees that there were only slight differences in individual contribution proportions.

- Aniruddh Aragola ([Aniruddh00001](https://github.com/Aniruddh00001))
- Nabeth Ghazi ([nabethg](https://github.com/nabethg))
- Ran (Andy) Gong ([AG2048](https://github.com/AG2048))
- Zerui (Kevin) Wang ([Togohogo1](https://github.com/Togohogo1))

## Dataset File Structure

As stated above, our training data is from the IAM Handwriting Database, available [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database). To run the Jupyter Notebook files without error:

- Download and unzip `data/ascii.tgz` and move only `lines.txt` into the `data/` folder
- Download and unzip `data/lines.tgz` into `data/lines/`

The end result should look something like this:

```text
./
├── data/
│   ├── chars/
│   │   ├── 02/
│   │   │   ...
│   │   └── 72/
│   ├── lines/
│   │   ├── a01/
│   │   │   ...
│   │   └── r06/
│   └── lines.txt
```

## Acknowledgments

This project was part of the [APS360](https://engineering.calendar.utoronto.ca/course/aps360h1) course offered by the [University of Toronto Faculty of Applied Science and Engineering](https://www.engineering.utoronto.ca/).

The [report](docs/report) Latex template has been derived from the assignment guideline.

### Notes

Visit <https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database>:

- Download data/ascii.tgz and unzip only lines.txt into the data folder
- Download data/lines.tgz - Contains t and unzip to

The final file directory should look smth like this:

```text
./
├── data/
│   ├── lines/
│   ├── sentences/
│   ├── words/
│   ├── lines.txt
│   ├── sentences.txt
│   └── words.txt
```

<https://github.com/nabethg/ScribeSmith/tree/010124a50fc1e47df7b9611344bbf29f2d4c1dd6/testing_stuff> is the last commit with the raw files before reformat

Some basic written instructions:

- have conda installed (mini or normal)
  - conda version
- install the following from environment.yml
  - for both windows and linux
- some basic info pulled from conda env?
  - conda env create and conda activate
- how to install from conda env
  - conda env update

When going through all the code, decide on what to remove and include in the .gitignore
