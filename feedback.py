# import requests
# import json

# # Endpoint URL
# url = "https://gw.cortical.io/nlp/keywords"

# # Request payload
# payload = {"language": "en"}

# # Headers
# headers = {
#     "Content-Type": "application/json",
#     "Accept": "application/json",
#     "Authorization": "eyJvcmciOiI2NTNiOTllNjEzOGM3YzAwMDE2MDM5NTEiLCJpZCI6Ijk4ZjllYjQyOGE1NzQ0ZDg5OGU3NzIxZDdjNDM4N2I1IiwiaCI6Im11cm11cjEyOCJ9",
# }


# def findMissingKeywords(reference_answer, student_answer):
#     try:
#         payload["text"] = reference_answer
#         # Send POST request with timeout
#         response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        
#         # Check if the request was successful (status code 200)
#         if response.status_code == 200:
#             # Parse the JSON response
#             data = response.json()
#             # Extract keywords
#             keywords = [keyword["word"] for keyword in data["keywords"]]
#             student_answer_list = student_answer.lower().split()
#             missing_keywords = []
            
#             for keyword in keywords:
#                 keyword_lower = keyword.lower()
#                 if keyword_lower not in student_answer_list:
#                     missing_keywords.append(keyword)
            
#             return missing_keywords if missing_keywords else ["No keywords missing!"]
#         else:
#             return [f"API Error: {response.status_code}"]
    
#     except requests.exceptions.Timeout:
#         return ["API request timed out. Please try again."]
#     except requests.exceptions.ConnectionError:
#         return ["Unable to connect to keyword service. Please check your internet connection."]
#     except Exception as e:
#         return [f"Error: {str(e)}"]




import re
import requests
import json

url = "https://gw.cortical.io/nlp/keywords"

payload = {"language": "en"}

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "YOUR_API_KEY_HERE",
}

STOPWORDS = {
    "the", "is", "a", "an", "and", "of", "to", "in", "for", "with", "on", "that",
    "this", "it", "as", "are", "be", "by", "from"
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return set(text.split())


def findMissingKeywords(reference_answer, student_answer):
    try:
        payload["text"] = reference_answer

        response = requests.post(
            url, data=json.dumps(payload), headers=headers, timeout=10
        )

        if response.status_code != 200:
            return [f"API Error: {response.status_code}"]

        data = response.json()

        keywords = {
            kw["word"].lower()
            for kw in data.get("keywords", [])
            if kw["word"].lower() not in STOPWORDS
        }

        student_words = normalize(student_answer)

        missing = sorted(list(keywords - student_words))

        return missing

    except requests.exceptions.Timeout:
        return ["Keyword service timed out"]
    except Exception as e:
        return [f"Error: {str(e)}"]
