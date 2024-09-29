import requests
import json

# Endpoint URL
url = "https://gw.cortical.io/nlp/keywords"

# Request payload
payload = {"language": "en"}

# Headers
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "eyJvcmciOiI2NTNiOTllNjEzOGM3YzAwMDE2MDM5NTEiLCJpZCI6Ijk4ZjllYjQyOGE1NzQ0ZDg5OGU3NzIxZDdjNDM4N2I1IiwiaCI6Im11cm11cjEyOCJ9",  # Replace 'your_access_token_here' with your actual access token
}


def findMissingKeywords(reference_answer, student_answer):
    payload["text"] = reference_answer
    # Send POST request
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract keywords
        keywords = [keyword["word"] for keyword in data["keywords"]]
        student_answer_list = student_answer.split()
        missing_keywords = []
        for x in keywords:
            if x not in student_answer_list:
                missing_keywords.append(x)
        print(missing_keywords)
        return missing_keywords
    else:
        return [response.text]
