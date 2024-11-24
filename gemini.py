import google.generativeai as genai

genai.configure(api_key="AIzaSyAqPZGrv4ZELWiEt0spuY-od33-GMSk7DY")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)