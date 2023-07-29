# aps360-project

## Project File Tree

```text
./
├── baseline_model/
│   ├── model_graphs/
│   └── model_snapshots/
├── data/
│   ├── lines/
│   ├── sentences/
│   ├── words/
│   ├── lines.txt
│   ├── sentences.txt
│   └── words.txt
├── main_model/
│   ├── model_graphs/
│   ├── model_snapshots/
│   └── main_model.pth
├── baseline_model.ipynb
├── main_model.ipynb
└── README.md

10 directories, 7 files
```

## Notes

- Network might be learning that white background = real and black background = fake
  - White pixels disappear during later epochs
- Possible hyperparameter: different learning rates for the different networks
