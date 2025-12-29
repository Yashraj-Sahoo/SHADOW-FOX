import requests
from bs4 import BeautifulSoup

# URL to scrape
url = "https://www.shadowfox.org.in/internships"

# Send request
response = requests.get(url)

# Parse HTML
soup = BeautifulSoup(response.text, "html.parser")

# Extract all h2 headings
headings = soup.find_all("h2")

print("Headings found on the webpage:")
for h in headings:
    print(h.text.strip())