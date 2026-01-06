import requests
from bs4 import BeautifulSoup

url = "https://www.shadowfox.org.in/internships"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

headings = soup.find_all("h2")

print("Headings found on the webpage:")
for h in headings:
    print(h.text.strip())