import sys
import pandas as pd

def check_id_in_accepted_list(id_to_check, csv_file):
    try:
        accepted_IDs = pd.read_csv(csv_file)['ID'].str.lower().unique()
        if id_to_check.lower() in accepted_IDs:
            print('yes')
        else:
            print('no')
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_id_in_list.py <id_to_check> <csv_file>")
    else:
        check_id_in_accepted_list(sys.argv[1], sys.argv[2])
