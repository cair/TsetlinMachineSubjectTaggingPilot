import xml.etree.ElementTree as ET
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

ns = {'ns': 'http://data.stortinget.no'}

def get_all_sak_ids(session_id):
    url = f'https://data.stortinget.no/eksport/saker?sesjonid={session_id}'
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch session {session_id}, status code {response.status_code}")
            return []
        root = ET.fromstring(response.content)
        return [sak.find('ns:id', namespaces=ns).text for sak in root.findall('.//ns:sak', namespaces=ns)]
    except Exception as e:
        print(f"Exception while fetching sakIDs for {session_id}: {e}")
        return []

def parse_sak_detail(sakid):
    detalj_url = f'https://data.stortinget.no/eksport/sak?sakid={sakid}'
    try:
        detalj_response = requests.get(detalj_url)
        detalj_root = ET.fromstring(detalj_response.content)
    except:
        print(f"Failed to parse sakid {sakid}")
        return None

    tittel = detalj_root.find('.//ns:tittel', namespaces=ns)
    tittel = tittel.text if tittel is not None else 'Ikke oppgitt'

    emne_liste = [
        emne.find('ns:navn', namespaces=ns).text
        for emne in detalj_root.findall('.//ns:emne_liste/ns:emne', namespaces=ns)
        if emne.find('ns:navn', namespaces=ns) is not None
    ]

    publikasjoner = detalj_root.findall('.//ns:publikasjon_referanse_liste/ns:publikasjon_referanse', namespaces=ns)
    for publikasjon in publikasjoner:
        lenke_url = publikasjon.find('ns:lenke_url', namespaces=ns)
        if lenke_url is not None and lenke_url.text and 'inns' in lenke_url.text:
            full_url = f"https:{lenke_url.text}/?lvl=0"
            time.sleep(2)  # respect rate limit
            return extract_html_data(full_url, sakid, tittel, emne_liste)

    return {
        'sakid': sakid,
        'tittel': tittel,
        'emneord': emne_liste,
        'sammendrag': None,
        'html': None,
        'url': None,
    }

def extract_html_data(url, sakid, tittel, emneord):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        content_area = soup.find('div', class_='page-publication')
        sammendrag = content_area.get_text(separator='\n', strip=True).replace('\n', ' ') if content_area else None

        html_full = str(soup)

        all_text = soup.get_text(separator='\n', strip=True).replace('\n', ' ').replace('\r', ' ')

        vedtak_text = ''
        heading = soup.find('p', string='vedtak:')
        if heading:
            for sibling in heading.find_next_siblings('p'):
                vedtak_text += sibling.text.strip() + ' '
        else:
            for lvl in range(2, 50):
                sub_url = url.split('/?lvl=')[0].rstrip('/') + f'/{lvl}/'
                try:
                    sub_resp = requests.get(sub_url)
                    sub_soup = BeautifulSoup(sub_resp.text, 'html.parser')
                    heading = sub_soup.find('p', string='vedtak:')
                    if heading:
                        for sibling in heading.find_next_siblings(['p', 'ul']):
                            vedtak_text += sibling.text.strip() + ' '
                        break
                except:
                    continue

        return {
            'sakid': sakid,
            'tittel': tittel,
            'emneord': str(emneord),
            'url': url,
            'sammendrag': sammendrag,
            'vedtak': vedtak_text.strip(),
            'text': all_text,
            'html': html_full
        }

    except Exception as e:
        print(f"Error fetching HTML from {url}: {e}")
        return {
            'sakid': sakid,
            'tittel': tittel,
            'emneord': str(emneord),
            'url': url,
            'sammendrag': None,
            'vedtak': None,
            'text': None,
            'html': None
        }


def run_scraper(start_year=1999, end_year=2025, output_csv='XXX.csv'):
    all_data = []
    existing_ids = set()

    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        existing_ids = set(df_existing['sakid'].astype(str).tolist())
        print(f"Found existing data with {len(existing_ids)} cases, skipping duplicates...")

    for year in tqdm(range(start_year, end_year + 1), desc="Year sessions"):
        session_id = f"{year}-{year+1}"
        print(f"\nProcessing session: {session_id}")
        sak_ids = get_all_sak_ids(session_id)

        for idx, sakid in enumerate(tqdm(sak_ids, desc=f"Session {session_id}")):
            if sakid in existing_ids:
                continue

            print(f"  -> ({idx+1}/{len(sak_ids)}) Processing sakID: {sakid}")
            data = parse_sak_detail(sakid)
            if data:
                all_data.append(data)

            time.sleep(1.5) 

        if all_data:
            df = pd.DataFrame(all_data)
            if os.path.exists(output_csv):
                df.to_csv(output_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(output_csv, index=False)
            print(f"  - Saved {len(all_data)} cases from {session_id}")
            all_data = []

if __name__ == "__main__":
    run_scraper()