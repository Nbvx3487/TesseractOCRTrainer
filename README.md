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

`gui.py` is a settings-style window with three pages: Fonts, Corpus, and Training (model
name, the **adjust the font** button, and the log). It follows the Windows light or dark setting.
Training runs in WSL, and the window calls it for you.

Setup on Windows (Python 3.10+):

    pip install -r requirements-gui.txt

Run it by double-clicking `gui.pyw`. It opens without a console window. `python gui.py`
also works, but it shows a console window behind the app.

Fonts: the common Windows fonts (Arial, Times New Roman, Verdana and others) are added
at startup if they are installed. Drop more font files anywhere in the window, or use
**Add font...**. Remove a font with its **Remove** button. Each
font gets a weight (1 to 10): the number of times each training line is rendered and
trained in that font, so a higher weight means the model sees that font more. Every font
is also scored on the held-out lines, which shows any loss on the other fonts.

Base model: the base is the existing English model, not a row in the list. Its own
training images are not stored in the model file, so the window cannot replay them. To
keep the base's fonts from being forgotten, add common fonts you have (for example Arial
or Times New Roman) as extra rows, with a weight, and they are trained alongside yours.

Model name: one name for the whole run, typed in the Model name field.

Corpus: one shared box for all fonts. Paste or load text, or press **Generate 200 lines**
to append Minecraft and mod-loader lines. The log shows progress, then the before-and-after
results for each font and the saved model path.

WSL must have Pillow and the training tools (see Setup above). The window converts
Windows paths such as C:\Users\me\font.ttf to /mnt/c/Users/me/font.ttf for WSL.
Before a run starts, the window checks the model name and that the corpus has at least 10 lines.

## Run

```bash
python3 app.py
```

Open http://localhost:8000. Upload a font, paste at least 10 lines, set the model name
(letters, digits, `-`, `_`), and start training. Progress appears on the page.

## Notes

- Training runs 400 iterations by default. More takes longer; check the per-font scores to decide.
- Each run writes to `data/runs/<name>/`. The trained model is `data/runs/<name>/model/<name>.traineddata`.
- To use the trained model with Tesseract, copy that file into a folder and pass it
  with `--tessdata-dir`, using `-l <name>`.
- Russian stays on Tesseract's separate `rus` model. This trainer covers English and Latin text only.

## Status

The rendering, splitting, scoring, and web endpoints were tested. The fine-tuning step
has not yet been run end to end, because it needs the best model file and the WSL
training tools on the target machine. Check the log on the page for the first real run.
