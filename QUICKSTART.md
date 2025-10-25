# 🚀 Quick Start Guide

## Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (Already Done!)
```bash
python model/train_model.py
```

### Step 3: Run the Web Application
```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

---

## What You Can Do

1. **Test URLs** - Enter any URL to check if it's phishing or legitimate
2. **View Confidence Scores** - See how confident the model is
3. **Analyze Features** - Understand which URL characteristics triggered the detection
4. **Use Example URLs** - Click pre-loaded examples for quick testing

---

## Example Test URLs

### ✅ Safe (Legitimate)
- https://www.google.com
- https://github.com
- https://www.amazon.com

### ⚠️ Suspicious (Phishing - DO NOT VISIT)
- http://paypal-secure-login.com/update
- http://secure-banking-verify.tk/login
- http://apple-id-verify.cf/account

---

## Project Structure

```
phishing-detection/
├── app.py                  # Flask web server
├── model/
│   ├── train_model.py      # ML training script
│   └── phishing_model.pkl  # Trained model
├── data/
│   └── dataset.csv         # Training data
├── templates/
│   └── index.html          # Web interface
├── static/
│   └── style.css           # Styling
└── report/
    └── report.md           # Full academic report
```

---

## Model Performance

- **Accuracy**: 95.24%
- **Precision**: 100.00%
- **Recall**: 90.00%
- **F1-Score**: 94.74%

---

## Need Help?

1. Check the full [README.md](README.md)
2. Read the detailed [report.md](report/report.md)
3. Open an issue on GitHub

---

**Happy Phishing Detection! 🔒**
