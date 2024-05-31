import pandas as pd
import os
import shutil
import glob
import chardet
import logging

def delete_specific_rows_csv(folder_path):
    # folder_path = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\SYD1AIG01CURR04'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        
        # Load the CSV file
        df = pd.read_csv(file_path, encoding='EUC-KR', header=None, names=['column3', 'timestamp', 'value', 'column2'], low_memory=False)
        # df = pd.read_csv(file_path, encoding='EUC-KR', header=None, names=['timestamp', 'value'], low_memory=False)
        df['timestamp'] = df['timestamp'].astype(str)
        df['timestamp'] = df['timestamp'].str.replace("오전", "AM").str.replace("오후", "PM")
        
        # Filter out rows where the timestamp is likely in the format MM/DD/YYYY by checking for the absence of ':' (assuming time always has ':' in your desired format)
        df = df[df['timestamp'].str.contains(':', na=False)]

        # Delete the first two rows
        df.drop(index=[0, 1], inplace=True, errors='ignore')
        # df.drop(index=[0], inplace=True, errors='ignore')
        
        # Delete the first and fourth columns
        df.drop(columns=[df.columns[0], df.columns[3]], inplace=True, errors='ignore')

        # Add 'index=False' if you do not want to write row numbers to the file
        df.to_csv(file_path, index=False, header=False, encoding='EUC-KR')

        print(f'Processed {file_path}')


def find_missing_tags(all_tags_file, processed_tags_file, missing_tags_file):
    # Read tags from the file containing all tags
    with open(all_tags_file, 'r') as file:
        all_tags = {line.strip() for line in file}

    # Read tags from the file containing processed tags
    with open(processed_tags_file, 'r') as file:
        processed_tags = {line.strip() for line in file}

    # Find tags that are in the all_tags set but not in the processed_tags set
    missing_tags = all_tags - processed_tags

    # Save the missing tags to a new file
    with open(missing_tags_file, 'w') as file:
        for tag in sorted(missing_tags):
            file.write(tag + '\n')

    print(f"Missing tags have been saved to {missing_tags_file}")

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

def read_csv_files(folder_path):
    # List all files in the given folder
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    
    # Print encoding for each file
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        encoding = detect_encoding(file_path)
        print(f'File: {file}, Encoding: {encoding}')

def standardize_dates(file_path, output_path):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, header=None, encoding='utf-8', names=['timestamp', 'value'], low_memory=False)

        def custom_date_parser(ts):
            ts = ts.strip().upper()  # Normalize the timestamp
            date_formats = [
                ('%Y-%m-%d %p %I:%M:%S', 'AM' in ts or 'PM' in ts),
                ('%m/%d/%Y %H:%M', '/' in ts and ':' in ts),
                ('%m/%d/%Y', '/' in ts),
                ('%Y-%m-%d %H:%M:%S', True),
                ('%Y-%m-%d %H:%M', True),
                ('%Y-%m-%d', True)
            ]
            for fmt, condition in date_formats:
                if condition:
                    try:
                        return pd.to_datetime(ts, format=fmt, errors='coerce')
                    except ValueError:
                        continue
            return pd.NaT

        df['timestamp'] = df['timestamp'].apply(custom_date_parser)

        if df['timestamp'].isnull().any():
            print(f"Warning: Some dates were not recognized in {os.path.basename(file_path)}.")

        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df = df.sort_values(by='timestamp')
        df.to_csv(output_path, index=False, header=False, encoding='utf-8')

    except Exception as e:
        # Log the error with the filename where it occurred
        logging.error(f"Error processing file {file_path}: {str(e)}")
        raise

def inspect_timestamp_format(file_path, row_index):
    df = pd.read_csv(file_path, header=None, encoding='utf-8', names=['timestamp', 'value'], low_memory=False, skiprows=row_index-1, nrows=1)
    
    print("Raw timestamp data at row {}: {}".format(row_index, df['timestamp'].iat[0]))

def combine_all_files():
    data_folder = r"D:\수자원공사 POC\3. 정상구간 데이터"
    csv_list_file = r"C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\csv_tag_list_missing.txt"
    output_folder = r'C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\ProcessedNormalData'

    os.makedirs(output_folder, exist_ok=True)

    with open(csv_list_file, 'r') as file:
        tag_names = [line.strip() for line in file]

    for tag_name in tag_names:
        combined_df_list = []
        
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                # Handle files with or without "_modified"
                base_name = os.path.splitext(file)[0]
                current_tag_name = base_name.replace("_modified", "")
                if current_tag_name == tag_name:
                    # Read the CSV file, skipping the first two rows
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path, skiprows=2, header=None, encoding='EUC-KR', names=['timestamp', 'value'], low_memory=False)
                    df.drop(index=[0], inplace=True, errors='ignore')
                    df['timestamp'] = df['timestamp'].str.replace("오전", "AM").str.replace("오후", "PM")

                    # Filter out rows where the timestamp is likely in the format MM/DD/YYYY by checking for the absence of ':' (assuming time always has ':' in your desired format)
                    # df = df[df['timestamp'].str.contains(':', na=False)]
                    # Delete the first and fourth columns
                    # df.drop(columns=[df.columns[0], df.columns[3]], inplace=True, errors='ignore')
                    combined_df_list.append(df)

        # Combine the data frames for the current tag name and save to a new file
        if combined_df_list:
            combined_df = pd.concat(combined_df_list, ignore_index=True)
            output_path = os.path.join(output_folder, f"{tag_name}.csv")
            combined_df.to_csv(output_path, index=False, header=False, encoding='EUC-KR')
            print(f"Combined data saved to {output_path}")

    print("All CSV files have been combined successfully.")

def rename_files(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename contains "_modified"
        if "_modified" in filename:
            # Create the new filename by removing "_modified"
            new_filename = filename.replace("_modified", "")
            # Form the full paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} to {new_file}")

def combine_csv_files():
    # Define the paths to the CSV files
    file1_path = r"C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\ProcessedNormalData\SYD1AIG01CURR01.csv"
    file2_path = r"C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\AbnormalData\SYD1AIG01CURR01.csv"
    output_path = r"C:\Users\Default.DESKTOP-646QQQ2\Downloads\Training\Data Analysis K1Water\Data2024\SYD1AIG01CURR01_combined.csv"

    # Read the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Combine the data frames
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Save the combined data to a new CSV file
    combined_df.to_csv(output_path, index=False)

    print(f"Combined CSV saved to {output_path}")


if __name__ == "__main__":

    combine_all_files()
    # Call the function to combine the CSV files
    # combine_csv_files()

