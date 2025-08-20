from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import pandas as pd
import os
import traceback
from django.core.files.storage import default_storage
from django.conf import settings
from datetime import datetime
import re
from pathlib import Path

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .ml_logs import MLModel
import csv
import numpy as np

class CleanDataView(APIView):
    def post(self, request):
        print("POST to /clean-data/ received")
        print("Files uploaded:", request.FILES)
       
        files = request.FILES.getlist('files')
        if not files:
            return Response({'detail': 'No files uploaded.'}, status=status.HTTP_400_BAD_REQUEST)
         
        cleaned_files = []
        now = datetime.now()
        current_year = now.strftime('%y')

        for file in files:
            try:
                path = default_storage.save(f"uploads/{file.name}", file)
                file_path = default_storage.path(path)

                if file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_path)
                elif file.name.endswith('.csv'):
                    df = pd.read_csv(file_path, engine='python')
                else:
                    continue  # skip unsupported

                # Clean column names
                name_col = next((col for col in df.columns if 'passenger name' in col.lower()), None)
                if name_col:
                    df[name_col] = df[name_col].astype(str).str.replace(r"/", " ", regex=True)

                flight_cols = [col for col in df.columns if any(k in col.lower() for k in ['flight', 'incoming', 'outgoing', 'arrival', 'departure'])]
                if len(flight_cols) >= 2:
                    for col in flight_cols:
                        df[col] = df[col].astype(str).str.replace(r'[â†→]', ' to ', regex=True)

                        def extract_and_format_date(text):
                            match = re.search(r'(\d{1,2}[A-Za-z]{3})', text)
                            if match:
                                raw_date = match.group(1)
                                formatted = f"{raw_date[:-3]}-{raw_date[-3:].upper()}-{current_year}"
                                return re.sub(r'(\d{1,2}[A-Za-z]{3})', formatted, text)
                            return text
                        df[col] = df[col].apply(extract_and_format_date)
                    def parse_flight_date(text):
                        try:
                            match = re.search(r'(\d{1,2})-([A-Z]{3})-(\d{2})', text)
                            if match:
                                return datetime.strptime(match.group(0), "%d-%b-%y")
                            return pd.NaT
                        except:
                            return pd.NaT

                    def compute_stay(row):
                        in_date = parse_flight_date(str(row[flight_cols[0]]))
                        out_date = parse_flight_date(str(row[flight_cols[1]]))
                        if pd.notnull(in_date) and pd.notnull(out_date):
                            duration = (out_date - in_date).days
                            return str(duration).replace('-', '')
                        return None
                    df['stay_duration'] = df.apply(compute_stay, axis=1)
                contact_col = next((col for col in df.columns if 'contact' in col.lower()), None)
                if contact_col:
                    df[contact_col] = df[contact_col].astype(str).apply(lambda x: re.sub(r'[a-zA-Z.]', '', x))
                dob_col = next((col for col in df.columns if 'date of birth' in col.lower() or 'dob' in col.lower()), None)
                if dob_col:
                    def compute_age(dob):
                        try:
                            dob_parsed = pd.to_datetime(dob, errors='coerce')
                            age = int((now - dob_parsed).days / 365.25) if pd.notnull(dob_parsed) else None
                            return str(age).replace('-', '') if age is not None else None
                        except:
                            return None
                    df['passenger_age'] = df[dob_col].apply(compute_age)

                # === End Cleaning Rules ===

                timestamp = now.strftime("%d-%b-%Y_%I-%M%p")
                base_filename = f"{timestamp}_{file.name}"
                cleaned_dir = Path(settings.MEDIA_ROOT) / "cleaned"
                cleaned_dir.mkdir(parents=True, exist_ok=True)
                cleaned_full_path = cleaned_dir / base_filename

                name, ext = os.path.splitext(base_filename)
                counter = 1
                while cleaned_full_path.exists():
                    cleaned_full_path = cleaned_dir / f"{name}_{counter}{ext}"
                    counter += 1

                if file.name.endswith(('.xls', '.xlsx')):
                    df.to_excel(cleaned_full_path, index=False)
                else:
                    df.to_csv(cleaned_full_path, index=False)

                relative_cleaned_path = str(cleaned_full_path).replace(str(settings.MEDIA_ROOT), settings.MEDIA_URL).replace('\\', '/').lstrip('/')
                cleaned_files.append('/' + relative_cleaned_path)
                os.remove(file_path)

            except Exception as e:
                traceback.print_exc()
                return Response({'detail': f'Error processing {file.name}: {str(e)}'},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({'cleaned_files': cleaned_files}, status=status.HTTP_200_OK)


def count_csv_records(file_path):
    try:
        with open(file_path, newline='', encoding="utf-8") as f:
            # Sample first 1KB to guess delimiter
            sample = f.read(1024)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel              
            reader = csv.reader(f, dialect)
            count = sum(1 for _ in reader)
            return max(count - 1, 0)  
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return 0

@api_view(['GET'])
def list_cleaned_files(request):
    cleaned_dir = Path(settings.MEDIA_ROOT) / "cleaned"
    files = []

    for file in cleaned_dir.glob("*"):
        if not file.is_file():
            continue

        name = file.name

        try:
            parts = name.split("_", 2)
            timestamp_str = f"{parts[0]}_{parts[1]}"
            original_filename = parts[2]
            dt = datetime.strptime(timestamp_str, "%d-%b-%Y_%I-%M%p")
            readable_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error parsing filename '{name}': {e}")
            original_filename = name
            readable_date = "Unknown"

        airline = original_filename.split("_")[0] if "_" in original_filename else "Unknown"
        file_size = file.stat().st_size
        total_records = count_csv_records(file)

        print(f"File: {name}, Airline: {airline}, Date: {readable_date}, Size: {file_size} bytes, Records: {total_records}")

        files.append({
            "filename": name,
            "airline": airline,
            "datetime": readable_date,
            "size": file_size,
            "cleanedRecords": total_records,
            "path": f"/media/cleaned/{name}"
        })

    return Response({"cleaned_files": files})

    


@api_view(['POST'])
def delete_cleaned_file(request):
    try:
        relative_path = request.data.get('path', '')
        if not relative_path:
            return Response({'detail': 'Missing file path.'}, status=status.HTTP_400_BAD_REQUEST)

        # Convert relative path to full path
        full_path = os.path.join(settings.MEDIA_ROOT, relative_path.replace(settings.MEDIA_URL, '').lstrip('/'))

        if os.path.exists(full_path):
            os.remove(full_path)
            return Response({'detail': 'File deleted successfully.'}, status=status.HTTP_200_OK)
        else:
            return Response({'detail': 'File not found.'}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
def log_repush_action(request):
    filename = request.data.get("filename")
    print(f"Repush action received for file: {filename}")
    if not filename:
        return Response({"detail": "Filename is required."}, status=status.HTTP_400_BAD_REQUEST)

    # Build the full path
    file_path = os.path.join(settings.MEDIA_ROOT, "cleaned", filename)  

    if not os.path.exists(file_path):
        return Response({"detail": f"File not found at {file_path}"}, status=status.HTTP_404_NOT_FOUND)

    try:
        results_df = MLModel.detect_anomalies(file_path)  
        return Response(
            {
                "detail": "Repush processed successfully.",
                "total_records": len(results_df),
                "anomalies_detected": int(results_df['is_potential_drug_trafficker'].sum())
            },
            status=status.HTTP_201_CREATED
        )
    except Exception as e:
        traceback.print_exc()
        return Response({"detail": f"Error processing file: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def count_high_status_passengers(file_path):
    """Count passengers with 'high' status in a CSV file."""
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return sum(1 for row in reader if row.get('status', '').strip().lower() == 'high')


@api_view(['GET'])
def list_anomalies_files(request):
    anomalies_dir = Path(settings.MEDIA_ROOT) / "anomalies"
    anomalies_dir.mkdir(parents=True, exist_ok=True)  

    files = []

    for file in anomalies_dir.glob("*"):
        name = file.name
        total_passengers = count_csv_records(file)
        high_status_passengers = count_high_status_passengers(file)

        readable_date = "Unknown"
        original_filename = name
        airline = "Unknown"

        try:
            # Expected format: anomalies_13-Aug-2025_03-26PM_file.csv
            parts = name.split("_", 3)
            timestamp_str = f"{parts[1]}_{parts[2]}"
            original_filename = parts[3]
            dt = datetime.strptime(timestamp_str, "%d-%b-%Y_%I-%M%p")
            readable_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            airline = original_filename.split("_")[0]
        except Exception as e:
            print(f"Error parsing filename '{name}': {e}")

        files.append({
            "filename": name,
            "airline": airline,
            "datetime": readable_date,
            "path": f"/media/anomalies/{name}",
            "total_passengers": total_passengers,
            "anomalies": high_status_passengers
        })

    return Response({"anomalies_files": files})


@api_view(['GET'])
def get_anomaly_file_content(request, filename):
    anomalies_dir = Path(settings.MEDIA_ROOT) / "anomalies"
    try:
        file_path = (anomalies_dir / filename).resolve()

        # Ensure inside anomalies dir
        if not str(file_path).startswith(str(anomalies_dir.resolve())):
            return Response({"detail": "Invalid file path."}, status=status.HTTP_400_BAD_REQUEST)

        if not file_path.exists():
            return Response({"detail": "File not found."}, status=status.HTTP_404_NOT_FOUND)

        # Read file
        if filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, engine='python')

        
        df = df.replace([np.nan, np.inf, -np.inf], None)

        # Convert to list of dicts
        data = df.to_dict(orient='records')

        return Response({
            "filename": filename,
            "total_rows": len(data),
            "data": data
        })

    except pd.errors.ParserError as e:
        return Response({"detail": f"CSV parse error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)