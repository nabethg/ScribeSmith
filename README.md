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

## Acknowledgments

This project was part of the [APS360](https://engineering.calendar.utoronto.ca/course/aps360h1) course offered by the [University of Toronto Faculty of Applied Science and Engineering](https://www.engineering.utoronto.ca/).

The [report](docs/report) Latex template has been derived from the assignment guideline.
