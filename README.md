# LKMLspeech - Linux Kernel Mailing List Email Analysis

This project is a **replication of a 2015 study** that analyzes emails from the Linux Kernel Mailing List comparing the writing styles of **Linus Torvalds** and **Greg Kroah-Hartman**.

[Original study repository](https://github.com/FLOSSmole/LKMLspeech)

## Python 2 Environment Setup

### Why Python 2?
This project replicates the original **2015 study** which was written in **Python 2.x**. Therefore, we maintain Python 2 compatibility to ensure accurate replication. The original code uses legacy libraries that are not compatible with Python 3:
- `print` statement syntax (without parentheses)
- `.iteritems()` method (removed in Python 3)
- `sklearn.cross_validation` (deprecated)
- General Python 2 syntax

### Step 1: Check if you have Python 2.7 installed

```bash
python2 --version
# or
python --version
```

**Important**: You need Python 2.7.x installed. Most Linux distributions include Python 2.7, but you can also install it:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python2.7 python2.7-dev
```

**Fedora/RHEL/CentOS:**
```bash
sudo yum install python2
```

**Arch Linux:**
```bash
sudo pacman -S python2
```

### Step 2: Install pip for Python 2.7

Python 2.7 may not come with pip installed. Download and install it:

```bash
cd ~/LKMLspeech  # or your project directory
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
python2 get-pip.py --user
```

Or install via package manager:

**Ubuntu/Debian:**
```bash
sudo apt-get install python-pip
```

**Fedora/RHEL/CentOS:**
```bash
sudo yum install python2-pip
```

### Step 3: Install virtualenv for Python 2.7

```bash
python2 -m pip install --user virtualenv
```

### Step 4: Create virtual environment with Python 2.7

**IMPORTANT**: Use Python 2 explicitly to ensure the venv is created with the correct version:

```bash
python2 -m virtualenv venv_py2
```

You should see a confirmation message indicating the Python 2.7 environment was created.

### Step 5: Activate the virtual environment

```bash
source venv_py2/bin/activate
```

You will see `(venv_py2)` at the beginning of the prompt.

### Step 6: Verify Python version

Confirm you are using Python 2.7:

```bash
python --version
```

Should display something like: `Python 2.7.18` or `Python 2.7.x`

### Step 7: Install dependencies

With the environment activated, install the required libraries with specific compatible versions:

```bash
pip install numpy==1.16.6 scipy==1.2.3
pip install scikit-learn==0.20.4 matplotlib==2.2.5
pip install nltk==3.4.5
```

**Why these versions?**
- `numpy 1.16.6` and `scipy 1.2.3`: Last versions compatible with Python 2.7
- `scikit-learn 0.20.4`: Compatible with Python 2.7 and has `sklearn.cross_validation`
- `matplotlib 2.2.5`: Last version for Python 2.7
- `nltk 3.4.5`: Natural Language Toolkit for Python 2.7

**Note**: On Linux, you may need development packages:
```bash
# Ubuntu/Debian
sudo apt-get install python-dev build-essential

# Fedora/RHEL/CentOS
sudo yum install python-devel gcc gcc-c++
```

### Step 8: Download NLTK data

Download the necessary data for natural language processing:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Step 9: Test installation

Verify that everything was installed correctly:

```bash
python -c "import sklearn; import nltk; import numpy; import scipy; import matplotlib; print 'All libraries imported successfully!'"
```

If the success message appears, you're all set!

### Step 10: Deactivate the virtual environment

When finished:

```bash
deactivate
```

## Data File

The `lkml_200_each.csv` file contains emails from the Linux Kernel Mailing List with the following structure:
- **from**: Email author ID
- **subject**: Email subject
- **date**: Email date
- **message-id**: Message ID
- **in-reply-to**: Replied message ID
- **references**: References to other messages
- **raw_body**: Email body (used for analysis)
- **author_id**: Author ID

### Author IDs

- **Linus Torvalds**: `38d186e8d1752771441f67080ca38409d807c50a`
- **Greg Kroah-Hartman**: `36c1ce7670ded355614f5635c72a01bef15f1c61`

## Dataset (LKML5Ws) Exploratory Script (test.py)

In addition to the CSV-based replication scripts, this repository contains an exploratory script (`test.py`) used during dataset selection and validation. The purpose of this script is to run lightweight checks directly on the decompressed **LKML5Ws** Parquet partitions (before converting anything to CSV), in order to verify message coverage and identify dataset limitations relevant to the replication.

### Where to place it

After decompressing LKML5Ws (see the dataset instructions in the LKML5Ws repository), place `test.py` **inside the dataset folder**, at the same level as the `LKML5Ws/` directory:


### How to run

From the dataset folder:

python test.py

(Note: this script may require a Python environment with Parquet support, e.g., polars or pyarrow. It is intended for exploratory inspection and is not part of the Python 2 replication pipeline.)

## Available Scripts

### 1. `partsOfSpeech_csv.py`
Analyzes parts of speech (POS tags) in Linus and Greg's emails.

**How to run:**
```bash
# Activate virtual environment
source venv_py2/bin/activate

# Run script
python partsOfSpeech_csv.py
```

**What it does:**
- Loads emails from CSV
- Separates by author (Linus vs Greg)
- Removes author names
- Analyzes POS tags using NLTK
- Shows count of each POS tag type for each author

### 2. `classifier_csv.py`
Creates a machine learning classifier to identify whether an email was written by Linus or Greg.

**How to run:**
```bash
python classifier_csv.py
```

**What it does:**
- Loads emails from CSV
- Trains Naive Bayes classifier using TF-IDF
- Adds custom features:
  - Expletive count
  - Exclamation mark count
  - Adverb count
- Runs 10 iterations with cross-validation
- Shows:
  - Average accuracy
  - Confusion matrix
  - Most important words
  - Linus:Greg ratios for each word

### 3. `countExpletives_csv.py`
Counts expletives in each author's emails.

**How to run:**
```bash
python countExpletives_csv.py
```

**What it does:**
- Loads expletive list from `googleListOfExpletives` file
- Counts occurrences of each expletive by author
- Shows expletive ranking for Linus and Greg

### 4. `descriptiveStatistics_csv.py`
Calculates descriptive statistics of emails.

**Additional dependency:**
```bash
pip install textstat
```

**How to run:**
```bash
python descriptiveStatistics_csv.py
```

**What it does:**
- Calculates for each author:
  - Average word count
  - Average sentence count
  - Average lexical diversity
  - Flesch Reading Ease Score
  - Grade Level (education level required to read)

## Running Original Scripts

You can also run the original scripts (without _csv suffix):

```bash
python partsOfSpeech
python classifier
python countExpletives
python descriptiveStatistics
```

## Differences Between Original and CSV Scripts

### Original Scripts
- Read emails from multiple text files in separate directories
- Used `os.listdir()` to iterate over files
- Hardcoded paths like `/path/todirectory/where/stuff/is`

### Adapted Scripts (_csv.py)
- Read from a single CSV file
- Use Python's `csv` module
- Identify authors by IDs in the `from` field
- Process the `raw_body` field for content
- Use relative path to script with `os.path.dirname(os.path.abspath(__file__))`

## CSV Scripts Processing Structure

All scripts follow this pattern:

1. **Define author IDs**
```python
LINUS_ID = "38d186e8d1752771441f67080ca38409d807c50a"
GREG_ID = "36c1ce7670ded355614f5635c72a01bef15f1c61"
```

2. **Read CSV and separate by author**
```python
with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        author_id = row['from']
        contents = row['raw_body']
        # process...
```

3. **Preprocess text**
- Convert to lowercase
- Remove author names
- Remove sequences of x's (email separators)

4. **Execute specific analysis**

## Required Files

For the scripts to work, you need these files in the same folder:

- ✅ `lkml_200_each.csv` - Dataset with emails
- ✅ `summaryOfPOSTags` - POS tags guide (for partsOfSpeech_csv.py)
- ✅ `googleListOfExpletives` - Expletives list (for classifier_csv.py and countExpletives_csv.py)

## Project Structure

```
LKMLspeech/
├── venv_py2/                          # Virtual environment (do not commit)
├── classifier                         # Classification script (original)
├── classifier_csv.py                  # Classification script (CSV version)
├── countExpletives                    # Expletive counter (original)
├── countExpletives_csv.py             # Expletive counter (CSV version)
├── partsOfSpeech                      # Parts of speech analysis (original)
├── partsOfSpeech_csv.py               # Parts of speech analysis (CSV version)
├── descriptiveStatistics              # Descriptive statistics (original)
├── descriptiveStatistics_csv.py       # Descriptive statistics (CSV version)
├── lkml_200_each.csv                  # Main dataset
├── googleListOfExpletives             # Expletives list
├── summaryOfPOSTags                   # POS tags guide
├── results_*.csv                      # Results files
└── README.md                          # This file
```

## Installed Dependencies

- **numpy 1.16.6**: Numerical computing
- **scipy 1.2.3**: Scientific computing
- **scikit-learn 0.20.4**: Machine learning (compatible with Python 2.7 and has `sklearn.cross_validation`)
- **nltk 3.4.5**: Natural Language Toolkit
- **matplotlib 2.2.5**: Data visualization

These are the last versions compatible with Python 2.7.

## Notes

### Lint Errors
Lint errors shown by VS Code are **false positives**. The code is written in **Python 2.7**, where:
- `print` is a statement, not a function (no parentheses)
- The syntax `print "text"` is valid
- VS Code linter is configured for Python 3

These scripts **work correctly** when executed in the Python 2.7 virtual environment.

### Python 2 vs Python 3
Do not try to run these scripts with Python 3. They use:
- `print` without parentheses
- `.iteritems()` (removed in Python 3)
- `sklearn.cross_validation` (deprecated)
- Different integer division

⚠️ **Python 2.7 reached end of life (EOL) on January 1, 2020**. It no longer receives security updates.

💡 **Note**: This project maintains Python 2 for accurate replication of the 2015 study. For modern applications, Python 3 is recommended.

### Future Migration

If you want to **migrate to Python 3** in the future, the main changes would be:
- `print "text"` → `print("text")`
- `.iteritems()` → `.items()`
- `from sklearn.cross_validation` → `from sklearn.model_selection`
- Division: `8/10` → `8//10` (for integer division)

## Troubleshooting

### Error: "cannot import name cross_validation"
- You are using a too recent version of scikit-learn
- Solution: `pip install scikit-learn==0.20.4`

### Error: "SyntaxError: Missing parentheses in call to 'print'"
- You are using Python 3
- Solution: Activate the correct virtual environment with Python 2.7
- Verify: `python --version` should show `Python 2.7.18`

### Error: Compilation errors when installing packages
- This error occurs when trying to compile packages from source
- Solution: Install development packages:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python-dev build-essential
  
  # Fedora/RHEL/CentOS
  sudo yum install python-devel gcc gcc-c++
  ```

### Error: "No module named virtualenv"
- Solution: `python2 -m pip install --user virtualenv`

### The venv was created with Python 3 instead of Python 2.7
- Problem: You used `python -m virtualenv` and your default Python is Python 3
- Solution: Use Python 2 explicitly: `python2 -m virtualenv venv_py2`
- Verify: `./venv_py2/bin/python --version` should show `Python 2.7.x`

### Permission denied errors
- Use `--user` flag when installing packages: `pip install --user <package>`
- Or ensure your virtual environment is activated

## .gitignore File

Add to your `.gitignore`:
```
venv_py2/
*.pyc
__pycache__/
```

## License

See the LICENSE file for details.
