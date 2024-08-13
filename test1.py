import google.generativeai as genai
import pandas as pd
import sys
from docx import Document

# Replace with your actual API key
GOOGLE_API_KEY = 'AIzaSyB1Yz_X5J5q_oaiT4dDIAb0vBziKg0iIF0'
genai.configure(api_key=GOOGLE_API_KEY)

def extract_medicine_info(feedback):
    """Extracts medicine info using the Gemini API."""
    prompt = f"""
    {feedback}

    The above is a patient's feedback regarding the medicine.
    Extract the patient name, medicine name,previous medical history and severity from the above context.
    Format the response as:
    patient name:<name>
    Date&Time of Notification from patient:<notified time>
    Date of Medicine Consumed:<medicine consumed date>
    medicine name:<medicine>
    Drug Dosage in mg:<dosage in mg>
    Symptoms Occured on:<date of symptoms started>
    side effects:<effects experienced>
    severity:<severity>
    previous medical history:<previous medical history>
    previous Medications:<previous medications>
    Address:<address of patient>
    """
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def clean_and_create_list(info):
    lines = info.strip().split('\n')
    cleaned_lines = [line.replace('*', '').strip() for line in lines if line.strip()]
    return cleaned_lines

def create_dict_from_list(cleaned_list):
    info_dict = {}
    for item in cleaned_list:
        if 'patient name' in item:
            info_dict['patient name'] = item.split('patient name')[1].strip()
        elif 'Date&Time of Notification from patient' in item:
            info_dict['Date&Time of Notification from patient'] = item.split('Date&Time of Notification from patient')[1].strip()
        elif 'Date of Medicine Consumed' in item:
            info_dict['Date of Medicine Consumed'] = item.split('Date of Medicine Consumed')[1].strip()
        elif 'medicine name' in item:
            info_dict['medicine name'] = item.split('medicine name')[1].strip()
        elif 'Drug Dosage in mg' in item:
            info_dict['Drug Dosage in mg'] = item.split('Drug Dosage in mg')[1].strip()
        elif 'Symptoms Occured on' in item:
            info_dict['Symptoms Occured on'] = item.split('Symptoms Occured on')[1].strip()
        elif 'side effects' in item:
            info_dict['side effects'] = item.split('side effects')[1].strip()
        elif 'severity' in item:
            info_dict['severity'] = item.split('severity')[1].strip()
        elif 'previous medical history' in item:
            info_dict['previous medical history'] = item.split('previous medical history')[1].strip()
        elif 'previous Medications' in item:
            info_dict['previous Medications'] = item.split('previous Medications')[1].strip()
        elif 'Address' in item:
            info_dict['Address'] = item.split('Address')[1].strip()    
    return info_dict

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_file.py <path_to_docx_file>")
        sys.exit(1)

    docx_file_path = sys.argv[1]
    doc = Document(docx_file_path)
    doc_data = "\n".join(para.text for para in doc.paragraphs)

    feed = extract_medicine_info(doc_data)
    info_list = clean_and_create_list(feed)
    info_dict = create_dict_from_list(info_list)

    df = pd.DataFrame(columns=['patinet name','Date&Time of Notification from patient', 'Date of Medicine Consumed', 'Drug name','Drug Dosage in mg','Symptoms Occured on','side effects','severity','previous medical history','previous Medications','Address'])
    new_row = pd.DataFrame([{
        'patient name': info_dict.get('patient name',''),
        'Date&Time of Notification from patient': info_dict.get('Date&Time of Notification from patient',''),
        'Date of Medicine Consumed': info_dict.get('Date of Medicine Consumed',''),
        'medicine name': info_dict.get('medicine name', ''),
        'Drug Dosage in mg': info_dict.get('Drug Dosage in mg', ''),
        'Symptoms Occured on': info_dict.get('Symptoms Occured on', ''),
        'side effects': info_dict.get('side effects', ''),
        'severity': info_dict.get('severity', ''),
        'previous medical history': info_dict.get('previous medical history', ''),
        'previous Medications': info_dict.get('previous Medications', ''),
        'Address': info_dict.get('Address', ''),
          
    }])
    df = pd.concat([df, new_row], ignore_index=True)

    output_file = "extracted_medicine_info.csv"
    df.to_csv(output_file, index=False)
    print("Data successfully saved to", output_file)
