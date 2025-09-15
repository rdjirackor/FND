import requests

# Your Facticity API key
API_KEY = "d9a0f52a-9f3d-472a-bb23-b85fabb0a3de"

# Headers using X-API-KEY as per API documentation
HEADERS = {
    "X-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

def fact_check(query, version="v3", mode="sync", timeout=60):
    """
    Fact-check a single claim using Facticity API.
    """
    url = "https://api.facticity.ai/fact-check"
    payload = {
        "query": query,
        "version": version,
        "mode": mode,
        "timeout": timeout
    }

    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=timeout + 10)
        if response.status_code == 200:
            data = response.json()
            print("\n=== Fact-Check Result ===")
            print("Input:", data.get("input"))
            print("Classification:", data.get("Classification"))
            print("Assessment:", data.get("overall_assessment"))
            print("Disambiguation:", data.get("disambiguation"))
            print("Task ID:", data.get("task_id"))
            print("\nSources:")
            for s in data.get("sources", []):
                print(s)
        elif response.status_code == 401:
            print("Error: Unauthorized. Check your API key or token balance.")
        elif response.status_code == 408:
            print("Error: Request timed out. Try increasing the timeout or use async mode.")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

def extract_claims(text):
    """
    Extract individual claims from a block of text using Facticity API.
    """
    url = "https://api.facticity.ai/extract-claim"
    payload = {
        "input": text,
        "content_type": "text"
    }

    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        if response.status_code == 200:
            claims = response.json().get("claims", [])
            print("\n=== Extracted Claims ===")
            if claims:
                for i, claim in enumerate(claims, start=1):
                    print(f"{i}. {claim}")
            else:
                print("No claims extracted.")
        elif response.status_code == 401:
            print("Error: Unauthorized. Check your API key or token balance.")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

if __name__ == "__main__":
    # Example usage
    fact_check("Coronavirus is a hoax")
    extract_claims("Barack Obama was born in Kenya and the Earth is flat.")
