import boto3
import requests
from io import BytesIO
from PyPDF2 import PdfFileReader

import getListing 
import getvector
import kmean

from flask import jsonify
def makeTeam(idx, team):
    team_data = {
        "id": idx,
        "projectName": "Placeholder Project Name",
        "user_ids": team,
        "user_emails": ["placeholder@example.com" for _ in team],
        "user_names": ["Placeholder Name" for _ in team],
        "innovation_challenge_id": "Placeholder Challenge ID",
        "github_link": "https://github.com/placeholder",
        "figmaLink": "https://figma.com/placeholder",
        "descriptionOfProject": "Placeholder description of the project"
    }
    return jsonify(team_data)



#@users.route("/getAllUsers", methods=["GET"])
#def getAllUsers():
#    users = Canidate.query.all()
#    users_list = [user.to_dict1() for user in users]
#    return jsonify(users_list)|
#
#This is the output of the above code:
#[
#    {
#    "email": "smithshannon@example.net",
#    "firstName": "Michelle",
#    "github": "github.com/michelle",
#    "gradDate": "2026",
#    "id": 1,
#    "lastName": "Jones",
#    "linkedIn": "linked.com/michelle-jones",
#    "resume": "s3.amazonaws.com/michelle-jones/resume.pdf",
#    "location": "Piercetown",
#    "username": "linkatrina"
#    },
#    {
#    "email": "james59@example.net",
#    "firstName": "Sandy",
#    "github": "github.com/sandy",
#    "gradDate": "2028",
#    "id": 2,
#    "lastName": "Watts",
#    "linkedIn": "linked.com/sandy-watts",
#    "resume": "s3.amazonaws.com/sandy-watts/resume.pdf",
#    "location": "Roweland",
#    "username": "joannacruz"
#  },
#    ...
#]


def extract_text_from_pdf(pdf_content):
    pdf_reader = PdfFileReader(BytesIO(pdf_content))
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page_num).extract_text()
    return text

def main():
    # Call the function to get all users
    all_users = getAllUsers()

    # Initialize the S3 client
    s3_client = boto3.client('s3')

    # Array to store the extracted text and corresponding user ID
    extracted_texts = []

    for user in all_users:
        # Extract the S3 bucket name and key from the resume URL
        resume_url = user['resume']
        bucket_name = resume_url.split('/')[2]
        key = '/'.join(resume_url.split('/')[3:])

        # Download the resume from the S3 bucket
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        pdf_content = response['Body'].read()

        # Extract the text from the resume
        extracted_text = extract_text_from_pdf(pdf_content)

        # Append the extracted text and user ID to the array
        extracted_texts.append([extracted_text, user['id']])

    #print(extracted_texts)
    vectors = []
    listing_info = getListing.getListing()

    for text, user_id in extracted_texts:
        vector = getvector.get_vector(listing_info, text, user_id)
        vectors.append(vector)
    
    results = kmean.kmean(vectors)

    for idx, team in enumerate(results):   
        maketeam(idx, team)
            
            



    # Print the extracted texts for verification
    #for text in extracted_texts:
    #    vector = get_vector(listing_info)
    #    vectors.append(vector)
        



