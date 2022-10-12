import os
import argparse
import pandas as pd
import datetime as dt

def generate_dataset(f):
        df = pd.read_csv(f)
        df.columns = ['id', 'timestamp', 'long', 'lat']
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        for date, group in df.groupby(df['timestamp'].dt.date):
            print(str(date))
            print(group)


def open_file(input_file_path):
    with open(input_file_path) as f:
        generate_dataset(f)

def open_folder(input_folder_path):
    for filename in os.listdir(input_folder_path):
        with open(os.path.join(input_folder_path, filename), 'r') as f:

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-i", "--Input", help="Input File Path")
    parser.add_argument("-o", "--Output", help="Output File Path")

    # Read arguments from command line
    args = parser.parse_args()
    if args.Input:
        print("Displaying Output as: % s" % args.Input)

    # Iterating through the files to create new dataset
    open_file(args.Input)


if __name__ == '__main__':
    main()
