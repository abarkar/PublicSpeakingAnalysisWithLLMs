import sys
import pandas as pd

def read_id_from_csv(file_path, category, city, transcript_id):
    try:
        df = pd.read_csv(file_path)
        # Filter the DataFrame based on the criteria
        filtered_df = df[(df.iloc[:, 2] == category) & (df.iloc[:, 3] == city) & (df.iloc[:, 1] == transcript_id)]
        # print(filtered_df)
        if not filtered_df.empty:
            print(filtered_df.iloc[0, 4])  # Assuming the ID is in the fifth column
        else:
            print("No match found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python read_id_from_csv.py <file_path> <category> <city> <transcript_id>")
    else:
        read_id_from_csv(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
