
import streamlit as st
import numpy as np
import joblib
from urllib.parse import urlparse
import requests
import re

# Define the feature extraction function
def extract_features(url):
    features = []

    # 1. Having IP Address in URL
    def having_ip(url):
        ip_address_pattern = re.compile(
            r"((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)")
        if ip_address_pattern.search(url):
            return 1
        else:
            return 0

    # 2. Having '@' Symbol in URL
    def have_at_sign(url):
        return 1 if "@" in url else 0

    # 3. URL Length
    def get_length(url):
        return 1 if len(url) >= 54 else 0

    # 4. URL Depth
    def get_depth(url):
        depth = urlparse(url).path.count('/')
        return depth

    # 5. Redirection (//) in URL
    def redirection(url):
        pos = url.rfind('//')
        return 1 if pos > 6 else 0

    # 6. 'https' in URL Domain
    def http_domain(url):
        return 1 if 'https' in urlparse(url).netloc else 0

    # 7. Tiny URL
    def tiny_url(url):
        shortening_services = r"bit\.ly|goo\.gl|shorte\.st|tinyurl\.com|tr\.im|is\.gd|cli\.gs|yfrog\.com"
        match = re.search(shortening_services, url)
        return 1 if match else 0

    # 8. Prefix or Suffix in URL Domain
    def prefix_suffix(url):
        return 1 if '-' in urlparse(url).netloc else 0

    # 9. Domain-based features (Placeholder examples)
    def domain_features(url):
        return [0, 0, 0]  # Replace these with actual domain-based feature extraction logic

    # 10. Web Traffic (Placeholder, requires external API)
    def web_traffic(url):
        try:
            # This is a placeholder value; implement actual web traffic check
            return 0  # Example return value
        except:
            return 1

    # 11-14. HTML & Javascript-based features (Placeholder examples)
    def html_features(response):
        iframe = 0 if re.search(r'<iframe', response.text) else 1
        mouse_over = 1 if re.search(r"onmouseover", response.text) else 0
        right_click = 0 if re.search(r"event.button ?== ?2", response.text) else 1
        forwarding = 1 if len(response.history) > 2 else 0
        return [iframe, mouse_over, right_click, forwarding]

    # Extracting the features
    features.append(having_ip(url))
    features.append(have_at_sign(url))
    features.append(get_length(url))
    features.append(get_depth(url))
    features.append(redirection(url))
    features.append(http_domain(url))
    features.append(tiny_url(url))
    features.append(prefix_suffix(url))

    # Append domain-based features (e.g., DNS features, etc.)
    features.extend(domain_features(url))

    # Web traffic feature
    features.append(web_traffic(url))

    # Placeholder for HTTP response-based features
    try:
        response = requests.get(url, timeout=5)
        features.extend(html_features(response))
    except:
        # If the request fails, use default values
        features.extend([0, 0, 0, 0])

    # Add additional placeholders to reach 49 features
    while len(features) < 49:
        features.append(0)  # Default placeholder value; replace with actual logic if needed

    return features

# Function to predict phishing using the SVM model
def predict_phishing(features):
    # Load the tuned SVM model using joblib
    loaded_model = joblib.load('tuned_svm_model.pkl')

    # Make predictions
    new_data = np.array([features])
    prediction = loaded_model.predict(new_data)

    return prediction

# Streamlit app main function
def main():
    st.title('Phishing URL Detector')
    st.write("Enter a URL to check if it's phishing or not.")
    
    # Input URL
    url = st.text_input("Enter URL:")
    
    if st.button("Check"):
        # Extract features
        st.write("Extracting features...")
        features = extract_features(url)

        # Make prediction
        st.write("Predicting...")
        prediction = predict_phishing(features)
        
        # Display prediction
        if prediction[0] == 0:
            st.error("Phishing Alert! This URL is classified as phishing.")
        else:
            st.success("No Phishing Detected. This URL seems safe.")
    
if __name__ == '__main__':
    main()