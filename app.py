from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Training Data (70 examples)
TRAINING_DATA = [
    ("Congratulations! You won Rs.50000. Send bank details to claim prize now", "FRAUD"),
    ("URGENT: Account will be blocked. Click link and enter OTP immediately", "FRAUD"),
    ("Update your KYC now or account suspended. Share PAN and Aadhar details", "FRAUD"),
    ("You won lottery Rs.100000. Provide card number and CVV to receive amount", "FRAUD"),
    ("Your UPI PIN compromised. Reset by clicking link and entering new PIN", "FRAUD"),
    ("Paytm: Verify account by sharing OTP 123456 received on phone now", "FRAUD"),
    ("Bank account needs verification. Send CVV card details and password urgently", "FRAUD"),
    ("Free iPhone 14 winner! Pay Rs.500 shipping. Enter debit card details", "FRAUD"),
    ("Suspicious activity detected. Share your password to secure account immediately", "FRAUD"),
    ("Selected for Rs.75000 loan. Send Aadhar PAN and bank password now", "FRAUD"),
    ("Your account credited Rs.25000 by mistake. Return money or legal action taken", "FRAUD"),
    ("Google Pay verification required. Share last 4 digits of card and OTP", "FRAUD"),
    ("RBI notice: Update PAN card by clicking link and entering details urgently", "FRAUD"),
    ("You are selected for credit card. Submit salary slip and bank password", "FRAUD"),
    ("PhonePe security alert. Confirm identity by sharing Aadhar and OTP", "FRAUD"),
    ("Income tax refund of Rs.15000 approved. Click link enter bank details", "FRAUD"),
    ("Congratulations! Won Rs.10 lakh. Transfer Rs.5000 processing fee first", "FRAUD"),
    ("Amazon gift voucher Rs.5000 free. Share mobile number and OTP received", "FRAUD"),
    ("Your KYC expired. Update immediately by providing PAN CVV and password", "FRAUD"),
    ("BHIM app needs verification. Enter UPI PIN and confirm phone number", "FRAUD"),
    ("Government subsidy Rs.20000 approved. Pay Rs.500 for document verification", "FRAUD"),
    ("SBI account locked. Reset by sharing net banking username and password", "FRAUD"),
    ("Free laptop offer! Only 5 left. Pay delivery charges with card details", "FRAUD"),
    ("Police complaint filed against you. Settle Rs.50000 to withdraw case", "FRAUD"),
    ("Your parcel stuck in customs. Pay Rs.2000 fine with debit card now", "FRAUD"),
    ("Instagram verification badge available. Pay Rs.1000 share card number", "FRAUD"),
    ("Netflix subscription expired. Renew by entering card CVV and OTP", "FRAUD"),
    ("Covid vaccine certificate ready. Pay Rs.500 download by sharing details", "FRAUD"),
    ("Job offer Rs.80000 salary. Pay Rs.3000 registration with bank password", "FRAUD"),
    ("Facebook account will be deleted. Verify by entering email password", "FRAUD"),
    ("Lucky draw winner Rs.2 lakh. Claim by transferring Rs.10000 tax", "FRAUD"),
    ("Electricity bill overdue Rs.15000. Pay now or connection cut share details", "FRAUD"),
    ("Investment opportunity double money in 1 month. Send Rs.50000 now", "FRAUD"),
    ("Your Aadhar linked to criminal activity. Clear name pay Rs.25000 fine", "FRAUD"),
    ("Flipkart sale exclusive access. Register with credit card and CVV details", "FRAUD"),
    ("Your UPI payment of Rs.500 to Amit Kumar successful. Ref ID: 234567890", "LEGITIMATE"),
    ("Received Rs.1000 from Priya Sharma via Google Pay on 15 Nov 2025", "LEGITIMATE"),
    ("Payment reminder: Electricity bill Rs.850 due on 20th November", "LEGITIMATE"),
    ("Transaction alert: Rs.2500 debited from account ending 1234 for online shopping", "LEGITIMATE"),
    ("Monthly salary Rs.45000 credited to your account on 1st November", "LEGITIMATE"),
    ("Your Google Pay request of Rs.300 to Rajesh declined insufficient balance", "LEGITIMATE"),
    ("OTP for HDFC net banking login is 456789. Valid for 10 minutes only", "LEGITIMATE"),
    ("Thank you for payment Rs.1200 Zomato order. Delivery in 30 minutes", "LEGITIMATE"),
    ("Ola ride payment Rs.180 successful. Trip from Home to Office completed", "LEGITIMATE"),
    ("PhonePe: Rs.50 cashback credited for recharge of Rs.500. Enjoy shopping", "LEGITIMATE"),
    ("Statement generated for credit card ending 5678. Due date 25th Nov", "LEGITIMATE"),
    ("Netflix subscription Rs.649 renewed successfully for next month", "LEGITIMATE"),
    ("Money sent Rs.5000 to Mom via UPI. Transaction ID: UPI2345678", "LEGITIMATE"),
    ("Paytm recharge of Rs.199 successful for mobile number ending 9876", "LEGITIMATE"),
    ("Swiggy payment Rs.450 completed. Order tracking number is 345678", "LEGITIMATE"),
    ("Amazon order Rs.2500 delivered successfully. Rate your purchase experience", "LEGITIMATE"),
    ("Rent payment Rs.15000 received from tenant via bank transfer today", "LEGITIMATE"),
    ("EMI of Rs.8500 debited from account for home loan. Next due 1st Dec", "LEGITIMATE"),
    ("Insurance premium Rs.12000 paid successfully. Policy renewed for year", "LEGITIMATE"),
    ("Petrol expense Rs.3000 paid at HP pump using debit card today", "LEGITIMATE"),
    ("College fees Rs.55000 payment confirmed. Receipt sent to email address", "LEGITIMATE"),
    ("Grocery shopping Rs.2100 at DMart paid via UPI successfully", "LEGITIMATE"),
    ("Your fixed deposit Rs.100000 matured. Amount credited with interest today", "LEGITIMATE"),
    ("Movie tickets Rs.800 booked for PVR. Show time 7 PM today", "LEGITIMATE"),
    ("Gym membership Rs.5000 renewed for 3 months via online payment", "LEGITIMATE"),
    ("Water bill Rs.450 payment successful. Next billing cycle starts 1st Dec", "LEGITIMATE"),
    ("Donation Rs.2000 made to charity via UPI. Tax receipt will be emailed", "LEGITIMATE"),
    ("Book purchase Rs.650 on Amazon completed. Estimated delivery 3 days", "LEGITIMATE"),
    ("Internet bill Rs.999 autopay deducted from account successfully", "LEGITIMATE"),
    ("Parking fee Rs.100 paid at mall via Google Pay today afternoon", "LEGITIMATE"),
    ("Medicine purchase Rs.1200 at Apollo pharmacy payment done via PhonePe", "LEGITIMATE"),
    ("Birthday gift Rs.3500 sent to sister via NEFT. Transaction successful", "LEGITIMATE"),
    ("Car insurance Rs.18000 paid online. Policy documents sent to registered email", "LEGITIMATE"),
    ("Spotify premium Rs.119 monthly subscription renewed automatically today", "LEGITIMATE"),
    ("Your savings account balance is Rs.45670 as on 15 November 2025", "LEGITIMATE"),
]

# Testing Data (30 examples)
TESTING_DATA = [
    ("Final warning: Share OTP 987654 within 1 hour or account permanently closed", "FRAUD"),
    ("Lottery winner Rs.5 crore! Pay Rs.25000 processing fee immediately", "FRAUD"),
    ("WhatsApp verification needed. Forward this message to 10 contacts with OTP", "FRAUD"),
    ("Court summon received. Pay fine Rs.35000 online to avoid arrest warrant", "FRAUD"),
    ("Your PAN card blocked by IT department. Update with Aadhar and password", "FRAUD"),
    ("Jio recharge offer 2GB daily free. Share OTP and card details to activate", "FRAUD"),
    ("Microsoft tech support: Your computer has virus. Pay Rs.8000 for removal", "FRAUD"),
    ("Marriage proposal for you! Pay Rs.10000 registration see profile photos", "FRAUD"),
    ("Land registry in your name. Pay Rs.15000 stamp duty share bank details", "FRAUD"),
    ("Credit score improved to 850. Claim certificate pay Rs.2000 processing", "FRAUD"),
    ("Aarogya Setu app premium version. Pay Rs.500 unlock all health features", "FRAUD"),
    ("Your mobile number selected for free 5G upgrade. Share OTP to activate", "FRAUD"),
    ("Income doubled work from home opportunity. Send Rs.5000 registration fee", "FRAUD"),
    ("Your Facebook account reported. Verify identity within 24 hours with password", "FRAUD"),
    ("Gold coin free delivery! Pay only Rs.1500 shipping with card CVV details", "FRAUD"),
    ("Flight ticket Rs.8500 booked for Mumbai to Delhi on 20th November", "LEGITIMATE"),
    ("Your loan application approved. EMI Rs.12000 starts from next month", "LEGITIMATE"),
    ("ATM withdrawal Rs.5000 at ICICI Bank ATM near your location today", "LEGITIMATE"),
    ("Mutual fund SIP Rs.10000 deducted successfully. Units allocated in NAV", "LEGITIMATE"),
    ("Hotel booking Rs.6000 for 2 nights confirmed. Check-in 18th November", "LEGITIMATE"),
    ("Credit card bill Rs.23000 due on 30th Nov. Minimum payment Rs.2300", "LEGITIMATE"),
    ("Fastag recharge Rs.1000 successful. Balance updated for toll payments", "LEGITIMATE"),
    ("Online course fee Rs.15000 paid. Access credentials sent to email", "LEGITIMATE"),
    ("Restaurant bill Rs.2800 paid at Barbeque Nation via credit card", "LEGITIMATE"),
    ("Car service payment Rs.4500 done at Maruti service center successfully", "LEGITIMATE"),
    ("Mobile phone Rs.25000 purchased on EMI. First installment next month", "LEGITIMATE"),
    ("Newspaper subscription Rs.600 renewed for yearly home delivery service", "LEGITIMATE"),
    ("AC repair charges Rs.1800 paid to technician via Google Pay today", "LEGITIMATE"),
    ("Uber ride Rs.350 payment completed. Trip from airport to home finished", "LEGITIMATE"),
    ("LIC premium Rs.8000 paid successfully. Policy continues for next year", "LEGITIMATE"),
]

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Global variables for model and vectorizer
model = None
vectorizer = None
model_stats = {}

def train_model():
    """Train the fraud detection model"""
    global model, vectorizer, model_stats
    
    # Prepare training data
    train_messages = [item[0] for item in TRAINING_DATA]
    train_labels = [item[1] for item in TRAINING_DATA]
    train_cleaned = [clean_text(msg) for msg in train_messages]
    
    # Prepare testing data
    test_messages = [item[0] for item in TESTING_DATA]
    test_labels = [item[1] for item in TESTING_DATA]
    test_cleaned = [clean_text(msg) for msg in test_messages]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_cleaned)
    X_test = vectorizer.transform(test_cleaned)
    
    y_train = np.array([1 if label == 'FRAUD' else 0 for label in train_labels])
    y_test = np.array([1 if label == 'FRAUD' else 0 for label in test_labels])
    
    # Train multiple models and select best
    models = {
        'Naive Bayes': MultinomialNB(alpha=0.5),
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'SVM': SVC(kernel='linear', C=1.0, random_state=42)
    }
    
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = mdl
            best_model_name = name
    
    model = best_model
    
    # Calculate statistics
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    model_stats = {
        'model_name': best_model_name,
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall': round(recall_score(y_test, y_pred) * 100, 2),
        'f1_score': round(f1_score(y_test, y_pred) * 100, 2),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_train': len(TRAINING_DATA),
        'total_test': len(TESTING_DATA)
    }

# Train model on startup
train_model()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a message is fraud or legitimate"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        # Clean and vectorize
        cleaned_message = clean_text(message)
        message_vector = vectorizer.transform([cleaned_message])
        
        # Predict
        prediction = model.predict(message_vector)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(message_vector)[0]
            confidence = round(max(proba) * 100, 2)
        
        result = {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'is_fraud': bool(prediction == 1),
            'confidence': confidence,
            'message': message
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get model statistics"""
    return jsonify(model_stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)