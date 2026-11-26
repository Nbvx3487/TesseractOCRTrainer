# Tesseract Font Trainer

A local web page that fine-tunes Tesseract's English model on a font you upload,
then measures before-and-after accuracy on held-out lines. It also checks the
other fonts you list, so you can see whether training on one font hurts the rest.

## What it does

1. You upload a `.ttf` or `.otf` font.
2. You paste a corpus of text lines (English or Latin script; Cyrillic lines are skipped).
3. The app splits the lines 80/20 into training and control sets. No line appears in both.
4. It renders the lines with Tesseract's `text2image`, then fine-tunes the English model.
5. It scores the original and the trained model on the control lines, for the uploaded
   font and any extra fonts you name.

## Setup (Windows, through WSL)

Tesseract's training tools run on Linux, so run the app inside WSL Ubuntu.
The page is then opened from Windows at http://localhost:8000.

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng python3-pip
pip3 install -r requirements.txt
```

Check that the training tools are present:

```bash
which lstmtraining combine_tessdata text2image
```

Download the best English model into the `base_models` folder. Fine-tuning needs this
floating-point model; the fast model that ships with Tesseract cannot be trained:

```bash
mkdir -p base_models
curl -L -o base_models/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
```

## Make a corpus

`tools/make_corpus.py` writes 500 Minecraft and mod-loader lines to `data/corpus.txt`.

    python tools/make_corpus.py --count 500 --out data/corpus.txt

Load that file with **Load text file...** in the window.

## Desktop window (Windows)

`gui.py` is a window with a drop area for your font, settings, and a **Make good** button.
Training runs in WSL, and the window calls it for you.

Setup on Windows (Python 3.10+):

    pip install -r requirements-gui.txt

Run it:

    python gui.py

Settings: model name, font size, iterations, check fonts (tested to catch forgetting),
and the corpus (paste it or load a text file). The log shows progress, then the
before-and-after results and the saved model path.

WSL must have Pillow and the training tools (see Setup above). The window converts
Windows paths such as C:\Users\me\font.ttf to /mnt/c/Users/me/font.ttf for WSL.

## Run

```bash
python3 app.py
```

Open http://localhost:8000. Upload a font, paste at least 10 lines, set the model name
(letters, digits, `-`, `_`), and start training. Progress appears on the page.

## Notes

- Iterations default to 400. More iterations take longer; check the control scores to decide.
- Each run writes to `data/runs/<name>/`. The trained model is `data/runs/<name>/model/<name>.traineddata`.
- To use the trained model with Tesseract, copy that file into a folder and pass it
  with `--tessdata-dir`, using `-l <name>`.
- Russian stays on Tesseract's separate `rus` model. This trainer covers English and Latin text only.

## Status

The rendering, splitting, scoring, and web endpoints were tested. The fine-tuning step
has not yet been run end to end, because it needs the best model file and the WSL
training tools on the target machine. Check the log on the page for the first real run.
